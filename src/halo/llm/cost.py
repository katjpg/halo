from collections.abc import Mapping
from dataclasses import dataclass

from pydantic import computed_field

from halo.config.settings import ModelRates
from halo.models.usage import TokenUsage


@dataclass(frozen=True)
class ModelCost:
    model: str
    input_usd: float
    cached_input_usd: float
    output_usd: float

    @computed_field
    @property
    def total_usd(self) -> float:
        return self.input_usd + self.cached_input_usd + self.output_usd


def estimate_cost(
    model: str, usage: TokenUsage, rates: Mapping[str, ModelRates]
) -> ModelCost | None:
    """Estimate short-context token cost in USD.

    Args:
        model: Model identifier.
        usage: Token counts. Reasoning tokens count as output.
        rates: Rates per million tokens, with a separate rate for cached input.

    Returns:
        The cost breakdown, or None when the model has no configured rate.
    """
    model_rates = rates.get(model)
    if model_rates is None:
        return None
    uncached = usage.input_tokens - usage.cached_tokens
    return ModelCost(
        model=model,
        input_usd=model_rates.input * uncached / 1_000_000,
        cached_input_usd=model_rates.cached_input * usage.cached_tokens / 1_000_000,
        output_usd=model_rates.output * usage.output_tokens / 1_000_000,
    )
