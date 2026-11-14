import json
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field, replace
from hashlib import sha256
from time import perf_counter
from types import MappingProxyType
from typing import Any, Generic, Literal, TypeVar

from pydantic import BaseModel

from halo.config.settings import ModelSettings, RuntimeSettings
from halo.core.errors import LimitError
from halo.llm.providers.base import (
    Message,
    ModelProvider,
    ModelRequest,
    ModelResponseError,
    serialize_request,
)
from halo.llm.validate import OutputValidationError, ValidationIssue
from halo.models.usage import TokenUsage
from halo.types import AgentRole

Input = TypeVar("Input")
Output = TypeVar("Output", bound="AgentOutput")
ValidationStatus = Literal["accepted", "partial", "rejected"]


def format_request(
    request: str, context: Mapping[str, Any], constraints: Mapping[str, Any]
) -> str:
    return json.dumps(
        {"REQUEST": request, "CONTEXT": context, "CONSTRAINTS": constraints},
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    )


class AgentOutput(BaseModel, extra="forbid"):
    pass


@dataclass(frozen=True, kw_only=True)
class Agent(Generic[Input, Output]):
    role: AgentRole
    prompt: str
    schema: type[Output]
    build_context: Callable[[Input], Mapping[str, Any]]
    verify: Callable[[Output, Input], None] | None = None
    max_attempts: int = 1

    def __post_init__(self) -> None:
        if type(self.max_attempts) is not int or self.max_attempts < 1:
            raise ValueError("max_attempts must be a positive integer.")


@dataclass(frozen=True)
class ModelCall:
    call_index: int
    agent_role: AgentRole
    model: str
    status: Literal["completed", "failed"]
    invocation_digest: str
    context_bytes: int
    elapsed_ms: float
    usage: TokenUsage | None
    output: dict[str, Any] | None
    request_json: str | None = field(default=None, repr=False)
    validation_status: ValidationStatus | None = None
    validation_issues: tuple[str, ...] | None = None


class ModelTrace:
    """The model call budget and reported usage for one operation."""

    def __init__(self, max_calls: int) -> None:
        if type(max_calls) is not int or max_calls < 1:
            raise ValueError("max_calls must be a positive integer.")
        self.max_calls = max_calls
        self._started = 0
        self._calls: dict[int, ModelCall] = {}

    @property
    def calls(self) -> tuple[ModelCall, ...]:
        return tuple(self._calls[call_index] for call_index in sorted(self._calls))

    @property
    def remaining(self) -> int:
        return self.max_calls - self._started

    def reserve(self) -> int:
        """Reserve one model call from the remaining budget."""
        if self.remaining <= 0:
            raise LimitError("The model call limit was reached.")
        self._started += 1
        return self._started

    def record_call(self, call: ModelCall) -> None:
        if not 1 <= call.call_index <= self._started:
            raise ValueError("A model call needs a reserved call index.")
        if call.call_index in self._calls:
            raise ValueError("The model call was already recorded.")
        self._calls[call.call_index] = call

    def usage(self, *, after: int = 0) -> TokenUsage:
        """Sum reported usage after a call index, including failed calls."""
        total = TokenUsage()
        for call in self._calls.values():
            if call.call_index > after and call.usage is not None:
                total += call.usage
        return total

    def record_validation(
        self,
        call_index: int,
        status: ValidationStatus,
        issues: Iterable[ValidationIssue] = (),
    ) -> None:
        self._calls[call_index] = replace(
            self._calls[call_index],
            validation_status=status,
            validation_issues=tuple(
                dict.fromkeys(issue.name.lower() for issue in issues)
            ),
        )


@dataclass(frozen=True, kw_only=True)
class ModelRunner:
    """A configured model with a provider and execution limits."""

    provider: ModelProvider
    settings: ModelSettings
    runtime: RuntimeSettings = field(default_factory=RuntimeSettings)
    roles: Mapping[AgentRole, ModelSettings] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "roles", MappingProxyType(dict(self.roles)))

    def new_trace(self) -> ModelTrace:
        return ModelTrace(max_calls=self.runtime.max_model_calls)

    async def run(
        self,
        schema: type[Output],
        *,
        role: AgentRole,
        prompt: str,
        request: str,
        context: Mapping[str, Any],
        constraints: Mapping[str, Any],
        trace: ModelTrace,
    ) -> tuple[Output, int]:
        content = format_request(request, context, constraints)
        context_bytes = len(content.encode())
        if context_bytes > self.runtime.max_context_bytes:
            raise LimitError("The selected context exceeds the model input limit.")
        settings = self.roles.get(role, self.settings)
        payload = ModelRequest(
            model=settings.model,
            messages=(
                Message(role="system", content=prompt),
                Message(role="user", content=content),
            ),
            max_tokens=settings.max_tokens,
            temperature=settings.temperature,
            reasoning_effort=settings.reasoning_effort,
        )
        request_json = serialize_request(self.provider.name, payload, schema)
        call_index = trace.reserve()
        start = perf_counter()
        usage = None
        output = None
        model = settings.model
        try:
            response = await self.provider.generate_structured(payload, schema)
        except ModelResponseError as exc:
            usage = exc.usage
            raise
        else:
            usage = response.usage
            model = response.model
            output = response.output.model_dump(mode="json")
            return response.output, call_index
        finally:
            trace.record_call(
                ModelCall(
                    call_index=call_index,
                    agent_role=role,
                    model=model,
                    status="completed" if output is not None else "failed",
                    invocation_digest=sha256(request_json.encode()).hexdigest(),
                    context_bytes=context_bytes,
                    elapsed_ms=(perf_counter() - start) * 1000,
                    usage=usage,
                    output=output,
                    request_json=request_json if self.runtime.capture_prompts else None,
                )
            )


async def run_agent(
    agent: Agent[Input, Output],
    runner: ModelRunner,
    inputs: Input,
    *,
    request: str,
    constraints: Mapping[str, Any],
    trace: ModelTrace,
) -> tuple[Output, int]:
    """Run an agent and retry rejected output within the call budget."""
    context = agent.build_context(inputs)
    for attempt in range(agent.max_attempts):
        response, call_index = await runner.run(
            agent.schema,
            role=agent.role,
            prompt=agent.prompt,
            request=request,
            context=context,
            constraints=constraints,
            trace=trace,
        )
        if agent.verify is None:
            return response, call_index
        try:
            agent.verify(response, inputs)
        except OutputValidationError as exc:
            issue = exc.issue
            trace.record_validation(call_index, "rejected", (issue,))
            if attempt + 1 == agent.max_attempts or trace.remaining <= 0:
                raise
            constraints = {
                **constraints,
                "validation": issue.value,
            }
        else:
            trace.record_validation(call_index, "accepted")
            return response, call_index
