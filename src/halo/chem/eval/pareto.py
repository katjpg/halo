import math
from collections.abc import Mapping, Sequence, Set
from dataclasses import dataclass

from halo.chem.eval.objectives import CriterionDirection, Direction
from halo.chem.results import Metric, MetricResult, metric_change
from halo.core.errors import ParseError
from halo.types import MoleculeId


@dataclass(frozen=True, slots=True)
class MoleculeValues:
    id: MoleculeId
    properties: Mapping[str, float]


@dataclass(frozen=True, slots=True)
class ParetoEvaluation:
    """A candidate's objective values, changes from baseline, and Pareto flags.

    Delta is candidate minus baseline in the property's unit. An
    improvement requires a strict change in the preferred direction.
    """

    molecule_id: MoleculeId
    values: Mapping[str, float]
    delta: Mapping[str, float]
    improved: Mapping[str, bool]
    pareto_optimal: bool
    dominates_baseline: bool

    @property
    def n_improved(self) -> int:
        return sum(self.improved.values())


def evaluate_pareto(
    *,
    baseline: MoleculeValues,
    candidates: Sequence[MoleculeValues],
    objectives: Mapping[str, Direction],
) -> list[ParetoEvaluation]:
    """Compare distinct candidates with the baseline in input order.

    Compute the Pareto front among candidates, excluding the baseline.
    All objective values must be finite and repeated IDs must agree.
    """
    if not objectives:
        raise ParseError("objectives cannot be empty")
    if any(
        direction not in ("increase", "decrease") for direction in objectives.values()
    ):
        raise ParseError("objectives must use 'increase' or 'decrease'")
    keys = tuple(objectives)
    signs = tuple(1 if objectives[key] == "increase" else -1 for key in keys)

    base = _read_values(baseline, keys)
    values: dict[MoleculeId, tuple[float, ...]] = {baseline.id: base}
    order: list[MoleculeId] = []
    for mol in candidates:
        row = _read_values(mol, keys)
        if mol.id in values:
            if values[mol.id] != row:
                raise ParseError(f"molecule {mol.id!r} has conflicting values")
            continue
        values[mol.id] = row
        order.append(mol.id)

    gains = {
        mid: tuple(val * sign for val, sign in zip(row, signs, strict=True))
        for mid, row in values.items()
    }
    front = _pareto_front({gains[mid] for mid in order})
    base_gains = gains[baseline.id]
    evaluations = []
    for mid in order:
        row = values[mid]
        delta = {key: val - ref for key, val, ref in zip(keys, row, base, strict=True)}
        evaluations.append(
            ParetoEvaluation(
                molecule_id=mid,
                values=dict(zip(keys, row, strict=True)),
                delta=delta,
                improved={
                    key: delta[key] * sign > 0
                    for key, sign in zip(keys, signs, strict=True)
                },
                pareto_optimal=gains[mid] in front,
                dominates_baseline=_dominates(gains[mid], base_gains),
            )
        )
    return evaluations


def _read_values(mol: MoleculeValues, keys: Sequence[str]) -> tuple[float, ...]:
    row = []
    for key in keys:
        if key not in mol.properties:
            raise ParseError(f"{mol.id!r} has no {key!r} value")
        val = mol.properties[key]
        if not math.isfinite(val):
            raise ParseError(f"{mol.id!r} has a non-finite {key!r} value")
        row.append(float(val))
    return tuple(row)


def _pareto_front(vectors: Set[tuple[float, ...]]) -> set[tuple[float, ...]]:
    """Find nondominated vectors when larger values are preferred."""
    front: list[tuple[float, ...]] = []
    # Only a lexicographically larger vector can dominate, so visit largest first.
    # A discarded vector's dominator also dominates anything it could dominate.
    for vector in sorted(vectors, reverse=True):
        if not any(_dominates(other, vector) for other in front):
            front.append(vector)
    return set(front)


def _dominates(a: Sequence[float], b: Sequence[float]) -> bool:
    better = False
    for x, y in zip(a, b, strict=True):
        if x < y:
            return False
        better |= x > y
    return better


def prediction_front(
    source_id: MoleculeId,
    source: Mapping[str, MetricResult],
    candidates: Mapping[MoleculeId, Mapping[str, MetricResult]],
    criteria: Mapping[str, CriterionDirection],
) -> tuple[MoleculeId, ...]:
    """Rank comparable predictions by signed change or distance from the source."""
    baseline = {
        key: metric for key in criteria if isinstance(metric := source.get(key), Metric)
    }
    if not criteria or len(baseline) != len(criteria):
        return ()
    vectors = []
    for mid, predictions in candidates.items():
        if mid == source_id:
            continue
        scores = _prediction_scores(baseline, predictions, criteria)
        if scores is not None:
            vectors.append(MoleculeValues(mid, scores))
    front = evaluate_pareto(
        baseline=MoleculeValues(source_id, dict.fromkeys(criteria, 0.0)),
        candidates=vectors,
        objectives=dict.fromkeys(criteria, "increase"),
    )
    return tuple(row.molecule_id for row in front if row.pareto_optimal)


def _prediction_scores(
    baseline: Mapping[str, Metric],
    predictions: Mapping[str, MetricResult],
    criteria: Mapping[str, CriterionDirection],
) -> dict[str, float] | None:
    scores = {}
    for key, direction in criteria.items():
        delta = metric_change(baseline[key], predictions.get(key))
        if delta is None:
            return None
        scores[key] = (
            delta
            if direction == "increase"
            else -delta
            if direction == "decrease"
            else -abs(delta)
        )
    return scores
