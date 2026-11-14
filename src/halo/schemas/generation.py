from collections.abc import Mapping
from datetime import datetime
from math import isfinite
from typing import Annotated, Literal, Self
from uuid import UUID

from pydantic import BaseModel, BeforeValidator, Field, model_validator

from halo.chem.eval.objectives import ADMET_UNITS, OBJECTIVES
from halo.chem.results import Metric, Prediction, metric_change
from halo.llm.cost import ModelCost
from halo.models.generation import (
    Candidate,
    DesignConfig,
    RGroupEdit,
    Run,
    RunStatus,
    SampleConfig,
)
from halo.models.usage import TokenUsage
from halo.schemas.molecule import (
    ComputedMetric,
    MetricValue,
    ProfileResponse,
    metric_value,
    prediction_result,
    profile_response,
)
from halo.types import (
    DEFAULT_BATCH_SIZE,
    RECORD_ID,
    BatchSize,
    Criteria,
    CriterionDirection,
    DesignCriteria,
    FiniteFloat,
    MoleculeId,
    OptimizationTask,
)

GoalOutcome = Literal["met", "missed", "unavailable", "unassessed"]


class CandidateResponse(ProfileResponse):
    ordinal: int
    transformation: RGroupEdit | None = None
    rationale: str | None = None
    evidence_ids: tuple[str, ...] = ()
    prediction_snapshot: dict[str, MetricValue] | None = None
    prediction_changes: dict[OptimizationTask, float | None] | None = Field(
        default=None,
        description="Candidate prediction minus source, in the saved unit. Null means a comparison is unavailable.",
    )
    goal_outcomes: dict[OptimizationTask, GoalOutcome] = Field(
        default_factory=dict,
        description="Whether each requested goal was met by the saved result.",
    )

    def run_prediction(self, name: str) -> Prediction:
        """Read the prediction saved when this candidate was generated."""
        return prediction_result((self.prediction_snapshot or {}).get(name))


class OptimizationRequest(BaseModel, extra="forbid"):
    type: Literal["design"]
    source_molecule_id: Annotated[MoleculeId, RECORD_ID]
    criteria: DesignCriteria = Field(
        description="Directions relative to the source predictions."
    )
    n: BatchSize = DEFAULT_BATCH_SIZE
    request_id: UUID | None = None


class SamplingRequest(BaseModel, extra="forbid"):
    type: Literal["sample"]
    source_molecule_id: Annotated[MoleculeId, RECORD_ID]
    tasks: list[OptimizationTask] = Field(min_length=1)
    targets: dict[OptimizationTask, FiniteFloat] | None = None
    request_id: UUID | None = None

    @model_validator(mode="after")
    def check_objectives(self) -> Self:
        unexpected = set(self.targets or {}) - set(self.tasks)
        if unexpected:
            raise ValueError(
                f"targets name properties that are not being optimized: "
                f"{sorted(unexpected)}"
            )
        return self


def tag_run_request(value: object) -> object:
    """Choose the request type for clients that omit it."""
    if isinstance(value, dict) and "type" not in value:
        sample = "tasks" in value or ("criteria" not in value and "targets" in value)
        return value | {"type": "sample" if sample else "design"}
    return value


GenerationRunRequest = Annotated[
    OptimizationRequest | SamplingRequest,
    Field(discriminator="type"),
    BeforeValidator(tag_run_request),
]


class GenerationRunResponse(BaseModel):
    id: str
    status: RunStatus
    source_molecule_id: int
    model: str
    objectives: dict[str, float | None]
    usage: TokenUsage
    cost: ModelCost | None
    failure_reason: str | None
    created_at: datetime
    finished_at: datetime | None
    candidates: list[CandidateResponse]
    criteria: Criteria | None = None
    core_smiles: str | None = None
    n: int | None = None
    project_id: str | None = None
    parent_generation_run_id: str | None = None
    source_predictions: dict[str, ComputedMetric] | None = None
    request_id: UUID | None = None
    expires_at: datetime | None = None


def run_response(run: Run, cost: ModelCost | None) -> GenerationRunResponse:
    design = run.config if isinstance(run.config, DesignConfig) else None
    source_predictions = run.config.source_predictions
    targets = run.config.targets if isinstance(run.config, SampleConfig) else {}
    criteria: Mapping[OptimizationTask, CriterionDirection] = (
        design.criteria
        if design
        else {name: OBJECTIVES[name].direction for name in targets}
    )
    return GenerationRunResponse(
        id=str(run.id),
        status=run.status,
        source_molecule_id=run.source.molecule_id,
        model=run.model,
        objectives=dict(targets),
        usage=run.usage,
        cost=cost,
        failure_reason=run.failure_reason,
        created_at=run.created_at,
        finished_at=run.finished_at,
        request_id=run.request_id,
        expires_at=run.expires_at,
        criteria=design.criteria if design else None,
        core_smiles=design.core_smiles if design else None,
        n=design.n if design else None,
        project_id=run.project_id,
        parent_generation_run_id=run.source.run_id,
        source_predictions={
            name: ComputedMetric(
                status="computed",
                value=metric.value,
                unit=metric.unit,
                method=metric.method,
            )
            for name, metric in source_predictions.items()
        }
        if source_predictions is not None
        else None,
        candidates=[
            candidate_response(candidate, source_predictions, criteria, targets)
            for candidate in run.candidates
        ],
    )


def candidate_response(
    candidate: Candidate,
    source_predictions: Mapping[OptimizationTask, Metric] | None,
    criteria: Mapping[OptimizationTask, CriterionDirection],
    targets: Mapping[OptimizationTask, float | None],
) -> CandidateResponse:
    predictions = candidate.run_predictions
    changes = (
        {
            name: metric_change(source_predictions.get(name), predictions.get(name))
            for name in criteria
        }
        if source_predictions is not None and predictions is not None
        else None
    )
    outcomes: dict[OptimizationTask, GoalOutcome] = {}
    for name, direction in criteria.items():
        target = targets.get(name)
        delta = changes.get(name) if changes is not None else None
        if target is not None:
            metric = predictions.get(name) if predictions is not None else None
            if (
                isinstance(metric, Metric)
                and isfinite(metric.value)
                and metric.unit == ADMET_UNITS.get(name, metric.unit)
            ):
                met = (
                    metric.value >= target
                    if direction == "increase"
                    else metric.value <= target
                )
                outcomes[name] = "met" if met else "missed"
            else:
                outcomes[name] = "unavailable"
        elif delta is None:
            outcomes[name] = "unavailable"
        elif direction == "preserve":
            outcomes[name] = "met" if delta == 0 else "unassessed"
        else:
            met = delta > 0 if direction == "increase" else delta < 0
            outcomes[name] = "met" if met else "missed"
    return CandidateResponse(
        ordinal=candidate.ordinal,
        transformation=candidate.transformation,
        rationale=candidate.rationale,
        evidence_ids=candidate.evidence_ids,
        prediction_snapshot={
            name: metric_value(value) for name, value in predictions.items()
        }
        if predictions is not None
        else None,
        prediction_changes=changes,
        goal_outcomes=outcomes,
        **dict(profile_response(candidate.molecule_id, candidate.profile)),
    )
