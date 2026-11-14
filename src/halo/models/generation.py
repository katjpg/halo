from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import Annotated, Literal

from pydantic import AliasChoices, Field

from halo.chem.results import Metric, MetricResult
from halo.models.molecule import MoleculeProfile
from halo.models.usage import TokenUsage
from halo.types import (
    DEFAULT_BATCH_SIZE,
    RECORD_ID,
    UUID_STRING,
    ADMETProperty,
    BatchSize,
    Criteria,
    DecisionId,
    DesignCriteria,
    FiniteFloat,
    GenerationRunId,
    Message,
    MoleculeId,
    OptimizationTask,
    ProjectId,
    UserId,
)

RunStatus = Literal["running", "completed", "failed", "limit_exhausted"]
RunOutcome = Literal["completed", "limit_exhausted"]


@dataclass(frozen=True, kw_only=True)
class ModelSettings:
    max_tokens: int | None = None
    temperature: float | None = None
    reasoning_effort: str | None = None


@dataclass(frozen=True, kw_only=True)
class DesignConfig:
    type: Literal["design"] = "design"
    criteria: DesignCriteria
    n: BatchSize = DEFAULT_BATCH_SIZE
    instructions: Annotated[
        str | None, Field(validation_alias=AliasChoices("instructions", "request"))
    ] = None
    core_smiles: str | None = None
    decision_ids: tuple[DecisionId, ...] = ()
    source_predictions: Mapping[ADMETProperty, Metric] | None = None
    model_settings: ModelSettings | None = None


@dataclass(frozen=True, kw_only=True)
class SampleConfig:
    type: Literal["sample"] = "sample"
    samples: Annotated[int, Field(gt=0)] | None = None
    targets: Mapping[OptimizationTask, FiniteFloat | None] = field(default_factory=dict)
    model_settings: ModelSettings | None = None
    source_predictions: Mapping[OptimizationTask, Metric] | None = None


RunConfig = Annotated[DesignConfig | SampleConfig, Field(discriminator="type")]


@dataclass(frozen=True)
class Occurrence:
    molecule_id: Annotated[MoleculeId, RECORD_ID]
    run_id: Annotated[
        Annotated[GenerationRunId, UUID_STRING] | None,
        Field(
            validation_alias=AliasChoices("generation_run_id", "run_id"),
            serialization_alias="generation_run_id",
        ),
    ] = None


@dataclass(frozen=True)
class GenerationBatch:
    source: Occurrence
    molecule_ids: tuple[MoleculeId, ...]
    criteria: Criteria


@dataclass(frozen=True)
class RGroupEdit:
    site: int
    from_fragment_smiles: str
    to_fragment_smiles: str
    reaction_smarts: str | None


@dataclass(frozen=True, kw_only=True)
class DesignRequest:
    instructions: Message
    criteria: DesignCriteria
    n: BatchSize
    core_smiles: str | None = None


@dataclass(frozen=True)
class CandidateDesign:
    smiles: str
    rationale: str
    evidence_ids: tuple[str, ...]
    transformation: RGroupEdit | None


@dataclass(frozen=True, kw_only=True)
class PreparedCandidate:
    profile: MoleculeProfile
    design: CandidateDesign | None = None

    def __post_init__(self) -> None:
        if self.design and self.design.smiles != self.profile.canonical_smiles:
            raise ValueError("The design and profile describe different structures.")


@dataclass(frozen=True)
class PreparedGeneration:
    source: MoleculeProfile
    candidates: tuple[PreparedCandidate, ...]

    @property
    def status(self) -> RunOutcome:
        return "completed" if self.candidates else "limit_exhausted"


@dataclass(frozen=True)
class GenerationWrite:
    source: Occurrence
    model: str
    config: DesignConfig
    prepared: PreparedGeneration
    usage: TokenUsage
    preference_ids: tuple[DecisionId, ...] = ()


@dataclass(frozen=True, slots=True)
class Candidate:
    """A generated molecule with a zero-based position in its run."""

    molecule_id: MoleculeId
    ordinal: int
    profile: MoleculeProfile
    transformation: RGroupEdit | None = None
    rationale: str | None = None
    evidence_ids: tuple[str, ...] = ()
    run_predictions: Mapping[OptimizationTask, MetricResult] | None = None


@dataclass(frozen=True)
class RunStart:
    id: GenerationRunId
    replay: bool


@dataclass(frozen=True, kw_only=True)
class Run:
    """A stored generation run with candidates in proposal order."""

    id: GenerationRunId
    user_id: UserId
    source: Occurrence
    status: RunStatus
    model: str
    usage: TokenUsage
    failure_reason: str | None
    created_at: datetime
    finished_at: datetime | None
    candidates: tuple[Candidate, ...]
    config: RunConfig
    project_id: ProjectId | None = None
    request_id: str | None = None
    expires_at: datetime | None = None
