import math
from dataclasses import dataclass
from numbers import Real
from typing import Literal


@dataclass(frozen=True, slots=True)
class Metric:
    value: float
    unit: str
    method: str

    def __str__(self) -> str:
        return f"{self.value:.4g} {self.unit} ({self.method})"


@dataclass(frozen=True, slots=True)
class Failure:
    reason: str

    def __str__(self) -> str:
        return self.reason


MetricResult = Metric | Failure


@dataclass(frozen=True, slots=True)
class Prediction:
    """A property prediction, failed attempt, or unavailable result."""

    _result: MetricResult | None

    def __str__(self) -> str:
        result = self._result
        if isinstance(result, Metric):
            return str(result)
        if isinstance(result, Failure):
            return f"Prediction failed ({result.reason})."
        return "Prediction unavailable."

    @property
    def status(self) -> Literal["computed", "failed", "unavailable"]:
        if isinstance(self._result, Metric):
            return "computed"
        return "failed" if isinstance(self._result, Failure) else "unavailable"

    @property
    def value(self) -> float | None:
        return self._result.value if isinstance(self._result, Metric) else None

    @property
    def unit(self) -> str | None:
        return self._result.unit if isinstance(self._result, Metric) else None

    @property
    def method(self) -> str | None:
        return self._result.method if isinstance(self._result, Metric) else None

    @property
    def error(self) -> str | None:
        return self._result.reason if isinstance(self._result, Failure) else None


def metric_change(
    source: MetricResult | None, candidate: MetricResult | None
) -> float | None:
    """Return candidate minus source when both values share a unit and method."""
    if (
        not isinstance(source, Metric)
        or not isinstance(candidate, Metric)
        or not source.method
        or (source.unit, source.method) != (candidate.unit, candidate.method)
    ):
        return None
    delta = candidate.value - source.value
    return delta if math.isfinite(delta) else None


def to_metric(value: object, *, unit: str, method: str) -> MetricResult:
    """Convert a finite real number or numeric string to a metric.

    Return a Failure for missing, nonnumeric, or nonfinite values.
    """
    if value is None:
        return Failure(f"{method} returned no value")
    if isinstance(value, bool) or not isinstance(value, (Real, str)):
        return Failure(f"{method} returned a non-numeric value")
    try:
        number = float(value)
    except (ValueError, OverflowError):
        return Failure(f"{method} returned a non-numeric value")
    if not math.isfinite(number):
        return Failure(f"{method} returned {number}")
    return Metric(number, unit, method)
