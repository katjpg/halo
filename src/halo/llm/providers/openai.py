from collections.abc import Mapping
from hashlib import sha256
from typing import Any
from urllib.parse import urlsplit

from openai import AsyncOpenAI, OpenAIError
from pydantic import ValidationError

from halo.config.settings import OpenAISettings
from halo.core.errors import ConfigurationError, ProviderError
from halo.llm.providers.base import (
    Message,
    ModelRequest,
    ModelResponse,
    ModelResponseError,
    ProviderName,
    T,
    truncation_error,
)
from halo.models.usage import TokenUsage


def _prompt_cache_key(
    messages: tuple[Message, ...],
    cache_namespace: str | None,
) -> str | None:
    material = cache_namespace
    if material is None:
        material = next(
            (m.content for m in messages if m.role == "system"),
            "",
        )
    if not material:
        return None
    return sha256(material.encode()).hexdigest()[:40]


def _request(request: ModelRequest) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": request.model,
        "input": [message.model_dump() for message in request.messages],
    }
    if request.temperature is not None:
        payload["temperature"] = request.temperature
    if request.max_tokens is not None:
        payload["max_output_tokens"] = request.max_tokens
    if request.reasoning_effort is not None:
        payload["reasoning"] = {"effort": request.reasoning_effort}
    key = _prompt_cache_key(request.messages, request.cache_namespace)
    if key is not None:
        payload["prompt_cache_key"] = key
    return payload


def _field(value: Any, key: str) -> Any:
    # A failed SDK parse leaves the response as raw JSON.
    return value.get(key) if isinstance(value, Mapping) else getattr(value, key, None)


def _usage(response: Any) -> TokenUsage | None:
    usage = _field(response, "usage")
    if usage is None:
        return None
    input_details = _field(usage, "input_tokens_details")
    output_details = _field(usage, "output_tokens_details")
    reasoning_tokens = _field(output_details, "reasoning_tokens")
    cached_tokens = _field(input_details, "cached_tokens")
    try:
        counts = TokenUsage(
            input_tokens=_field(usage, "input_tokens"),
            output_tokens=_field(usage, "output_tokens"),
            reasoning_tokens=0 if reasoning_tokens is None else reasoning_tokens,
            cached_tokens=0 if cached_tokens is None else cached_tokens,
        )
    except ValidationError as exc:
        raise ModelResponseError("OpenAI returned invalid token usage.") from exc
    return counts


def _validated_usage(
    response: Any, max_tokens: int | None, provider: ProviderName
) -> TokenUsage:
    usage = _usage(response)
    status = _field(response, "status")
    if status == "incomplete":
        details = _field(response, "incomplete_details")
        if _field(details, "reason") == "max_output_tokens":
            raise truncation_error(provider, usage, max_tokens)
    if status not in (None, "completed"):
        raise ModelResponseError(
            f"OpenAI response ended with status {status!r}.", usage=usage
        )
    if any(
        _field(part, "type") == "refusal"
        for item in _field(response, "output") or ()
        if _field(item, "type") == "message"
        for part in _field(item, "content") or ()
    ):
        raise ModelResponseError("OpenAI refused the request.", usage=usage)
    if usage is None:
        raise ModelResponseError("OpenAI returned no token usage.")
    return usage


class OpenAIProvider:
    def __init__(
        self,
        settings: OpenAISettings | None = None,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        settings = settings or OpenAISettings()
        key = api_key if api_key is not None else settings.require_api_key()
        if not isinstance(key, str) or not key.strip():
            raise ConfigurationError("Provide an OpenAI API key.")
        resolved_url = base_url if base_url is not None else settings.base_url
        try:
            url = urlsplit(resolved_url or "https://api.openai.com/v1")
            port = url.port
        except ValueError as exc:
            raise ConfigurationError("Provide a valid model service URL.") from exc
        if (
            url.scheme not in {"https", "http"}
            or not url.hostname
            or url.username is not None
            or url.query
            or url.fragment
        ):
            raise ConfigurationError(
                "Provide a model service URL without credentials, query, or fragment."
            )
        self.name = (
            ProviderName.OPENAI
            if url.scheme == "https"
            and url.hostname == "api.openai.com"
            and port in (None, 443)
            and url.path.rstrip("/") == "/v1"
            else ProviderName.OPENAI_COMPATIBLE
        )
        self._client = AsyncOpenAI(
            api_key=key.strip(),
            base_url=resolved_url,
            max_retries=settings.max_retries,
            timeout=settings.timeout,
        )

    async def aclose(self) -> None:
        await self._client.close()

    async def generate(self, request: ModelRequest) -> ModelResponse[str]:
        try:
            response = await self._client.responses.create(**_request(request))
        except (OpenAIError, ValueError) as exc:
            raise ProviderError(f"OpenAI request failed: {exc}") from exc

        usage = _validated_usage(response, request.max_tokens, self.name)
        content = _field(response, "output_text")
        if not isinstance(content, str) or not content:
            raise ModelResponseError("OpenAI returned an empty response.", usage=usage)
        return ModelResponse(
            output=content,
            usage=usage,
            model=_field(response, "model") or request.model,
            provider=self.name,
        )

    async def generate_structured(
        self, request: ModelRequest, schema: type[T]
    ) -> ModelResponse[T]:
        try:
            raw = await self._client.responses.with_raw_response.parse(
                text_format=schema, **_request(request)
            )
        except (OpenAIError, ValueError) as exc:
            raise ProviderError(f"OpenAI request failed: {exc}") from exc

        try:
            response = raw.parse()
        except ValidationError as exc:
            usage = _validated_usage(
                raw.http_response.json(), request.max_tokens, self.name
            )
            raise ModelResponseError(
                f"OpenAI returned invalid {schema.__name__} output.", usage=usage
            ) from exc
        except (OpenAIError, ValueError) as exc:
            raise ProviderError("OpenAI returned an unreadable response.") from exc

        usage = _validated_usage(response, request.max_tokens, self.name)
        parsed = _field(response, "output_parsed")
        if not isinstance(parsed, schema):
            raise ModelResponseError(
                f"OpenAI returned no parsed {schema.__name__}.", usage=usage
            )
        return ModelResponse(
            output=parsed,
            usage=usage,
            model=_field(response, "model") or request.model,
            provider=self.name,
        )
