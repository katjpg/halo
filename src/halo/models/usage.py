from pydantic import BaseModel, Field, model_validator


class TokenUsage(BaseModel, frozen=True):
    """Cached tokens count as input. Reasoning tokens count as output."""

    input_tokens: int = Field(default=0, ge=0, strict=True)
    output_tokens: int = Field(default=0, ge=0, strict=True)
    reasoning_tokens: int = Field(default=0, ge=0, strict=True)
    cached_tokens: int = Field(default=0, ge=0, strict=True)

    @model_validator(mode="after")
    def check_token_counts(self) -> "TokenUsage":
        if self.cached_tokens > self.input_tokens:
            raise ValueError("Cached tokens cannot exceed input tokens.")
        if self.reasoning_tokens > self.output_tokens:
            raise ValueError("Reasoning tokens cannot exceed output tokens.")
        return self

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        if not isinstance(other, TokenUsage):
            return NotImplemented
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            reasoning_tokens=self.reasoning_tokens + other.reasoning_tokens,
            cached_tokens=self.cached_tokens + other.cached_tokens,
        )
