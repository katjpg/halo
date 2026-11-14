import json
from dataclasses import dataclass
from enum import StrEnum
from typing import Generic, Literal, Protocol, TypeVar

from pydantic import BaseModel, Field

from halo.core.errors import ProviderError
from halo.models.usage import TokenUsage
from halo.types import ReasoningEffort


class ProviderName(StrEnum):
    OPENAI = "openai"
    OPENAI_COMPATIBLE = "openai_compatible"


class Message(BaseModel, frozen=True, extra="forbid"):
    role: Literal["system", "user", "assistant"]
    content: str


class ModelRequest(BaseModel, frozen=True, extra="forbid"):
    model: str = Field(min_length=1)
    messages: tuple[Message, ...] = Field(min_length=1)
    max_tokens: int | None = Field(default=None, gt=0, strict=True)
    temperature: float | None = Field(default=None, allow_inf_nan=False)
    reasoning_effort: ReasoningEffort | None = None
    cache_namespace: str | None = None


def serialize_request(
    provider: ProviderName, request: ModelRequest, schema: type[BaseModel]
) -> str:
    """Capture the model request and its declared output schema."""
    content = {
        "provider": provider,
        **request.model_dump(mode="json"),
        "output_schema": {
            "name": schema.__name__,
            "schema": schema.model_json_schema(),
        },
    }
    return json.dumps(content, ensure_ascii=False, allow_nan=False)


Output = TypeVar("Output")
T = TypeVar("T", bound=BaseModel)


@dataclass(frozen=True)
class ModelResponse(Generic[Output]):
    output: Output
    usage: TokenUsage
    model: str
    provider: ProviderName


class ModelResponseError(ProviderError):
    """An unusable model response with its reported token usage."""

    def __init__(self, message: str, *, usage: TokenUsage | None = None) -> None:
        super().__init__(message)
        self.usage = usage


def truncation_error(
    provider: ProviderName,
    usage: TokenUsage | None,
    limit: int | None,
) -> ModelResponseError:
    counts = (
        f"{usage.output_tokens} output, {usage.reasoning_tokens} reasoning, "
        if usage is not None
        else ""
    )
    return ModelResponseError(
        f"{provider.value} stopped at the output token limit "
        f"({counts}limit {limit}). Raise max_tokens or lower reasoning_effort.",
        usage=usage,
    )


class ModelProvider(Protocol):
    name: ProviderName

    async def generate(self, request: ModelRequest) -> ModelResponse[str]: ...

    async def generate_structured(
        self, request: ModelRequest, schema: type[T]
    ) -> ModelResponse[T]: ...

    async def aclose(self) -> None: ...
