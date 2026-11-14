from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Annotated, Literal

from pydantic import AliasChoices, AwareDatetime, Field, PlainSerializer, TypeAdapter

from halo.models.analysis import (
    ClusterAnalysis,
    MoleculeAnnotation,
    ResultValue,
    ScatterPlot,
)
from halo.models.generation import Occurrence, RGroupEdit
from halo.models.research import Citation, Passage, Response, SourceExclusion
from halo.types import (
    DEFAULT_BATCH_SIZE,
    MAX_RECALL,
    MAX_SELECTION,
    RECORD_ID,
    ApplicationErrorCode,
    BatchSize,
    Criteria,
    DecisionId,
    DesignCriteria,
    GenerationRunId,
    Message,
    MoleculeId,
    Note,
    ProjectId,
    SessionId,
    TurnId,
    UserId,
)

RoutedTask = Literal[
    "profile",
    "retrieve",
    "design",
    "cluster",
    "plot",
    "refine_plot",
    "strategy",
    "converge",
    "activity",
    "assay_results",
    "recall",
    "clarify",
]
Task = Literal[RoutedTask, "select", "decide", "remember"]
Scope = Literal["lead", "parent", "batch", "all", "selected"]
RecallScope = Literal["all", "acceptances", "preferences", "selected_preferences"]
PreferenceScope = Literal["next_run", "ongoing"]
TurnStatus = Literal["running", "completed", "failed", "cancelled"]
DecisionType = Literal["accept_proposal", "set_preference"]
TurnErrorCode = Literal[ApplicationErrorCode, "timeout", "cancelled"]


@dataclass(frozen=True)
class Project:
    id: ProjectId
    user_id: UserId
    lead_id: Annotated[
        MoleculeId,
        Field(
            validation_alias=AliasChoices("source_molecule_id", "lead_id"),
            serialization_alias="source_molecule_id",
        ),
    ]
    created_at: datetime


@dataclass(frozen=True)
class SessionState:
    parent: Occurrence
    selected_ids: tuple[MoleculeId, ...] = ()
    plot_turn_id: TurnId | None = None


@dataclass(frozen=True)
class Session:
    id: SessionId
    project_id: ProjectId
    state: SessionState
    created_at: datetime


@dataclass(frozen=True)
class ExperimentProposal:
    parent: Occurrence
    comparison_ids: tuple[MoleculeId, ...]
    hypothesis: str
    evidence_ids: tuple[str, ...]
    result_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class Decision:
    id: DecisionId
    turn_id: TurnId
    decided_at: datetime
    note: str
    type: Annotated[
        DecisionType, Field(validation_alias=AliasChoices("decision_type", "type"))
    ]
    scope: PreferenceScope | None = None
    proposal_turn_id: TurnId | None = None
    supersedes_id: DecisionId | None = None
    proposal: ExperimentProposal | None = None
    applied: bool = False


@dataclass(frozen=True, kw_only=True)
class DesignHistory:
    preferences: tuple[Decision, ...] = ()
    decisions: tuple[Decision, ...] = ()
    previous_edits: tuple[RGroupEdit, ...] = ()


@dataclass(frozen=True, kw_only=True)
class AcceptProposal:
    type: Literal["accept_proposal"]
    proposal_turn_id: Annotated[TurnId, RECORD_ID]
    reason: Note
    supersedes_id: Annotated[DecisionId, RECORD_ID] | None = None


@dataclass(frozen=True, kw_only=True)
class SetPreference:
    type: Literal["set_preference"]
    text: Note
    scope: PreferenceScope
    supersedes_id: Annotated[DecisionId, RECORD_ID] | None = None


@dataclass(frozen=True, kw_only=True)
class Select:
    type: Literal["select"]
    molecule_ids: Annotated[
        tuple[Annotated[MoleculeId, RECORD_ID], ...], Field(max_length=MAX_SELECTION)
    ]
    parent: Occurrence | None = None


@dataclass(frozen=True, kw_only=True)
class Design:
    type: Literal["design"]
    criteria: DesignCriteria
    n: BatchSize = DEFAULT_BATCH_SIZE
    core_smiles: str | None = None
    instructions: Message = "Propose a new batch from the active parent."


@dataclass(frozen=True, kw_only=True)
class Research:
    type: Literal["research"]
    molecule_id: Annotated[MoleculeId, RECORD_ID]
    question: Message


@dataclass(frozen=True, kw_only=True)
class Recall:
    type: Literal["recall"]
    decision_ids: Annotated[
        tuple[Annotated[DecisionId, RECORD_ID], ...], Field(max_length=MAX_RECALL)
    ] = ()
    # preserve the timestamp spelling used in stored requests for replay
    as_of: (
        Annotated[
            AwareDatetime,
            PlainSerializer(datetime.isoformat, return_type=str, when_used="json"),
        ]
        | None
    ) = None
    scope: RecallScope | Literal["applicable_preferences"] = "all"


Action = Annotated[
    AcceptProposal | SetPreference | Select | Design | Research | Recall,
    Field(discriminator="type"),
]
ACTION = TypeAdapter(Action)


@dataclass(frozen=True)
class TurnRecord:
    id: TurnId
    session_id: SessionId
    turn_index: int
    status: TurnStatus
    request: dict
    output: TurnResult | None
    trace: dict | None
    started_at: datetime
    finished_at: datetime | None
    request_id: str | None = None


@dataclass(frozen=True, kw_only=True)
class RecentTurn:
    task: Task | None
    request: str
    response: str
    molecule_ids: tuple[MoleculeId, ...]


@dataclass(frozen=True)
class StartedTurn:
    turn: TurnRecord
    project: Project
    state: SessionState


@dataclass(frozen=True)
class TurnResult:
    """Result of a turn. Include the fields produced by its task."""

    task: Task | None = None
    message: str | None = None
    response: Response | None = None
    passages: tuple[Passage, ...] = ()
    evidence: tuple[Citation, ...] = ()
    values: tuple[ResultValue, ...] = ()
    annotations: tuple[MoleculeAnnotation, ...] = ()
    unavailable: tuple[str, ...] = ()
    proposal: ExperimentProposal | None = None
    plot: ScatterPlot | None = None
    clusters: ClusterAnalysis | None = None
    candidate_ids: tuple[MoleculeId, ...] = ()
    baseline_id: MoleculeId | None = None
    criteria: Criteria | None = None
    generated_count: int | None = None
    generation_run_id: GenerationRunId | None = None
    decision_id: DecisionId | None = None
    error: str | None = None
    error_code: TurnErrorCode | None = None
    source_exclusions: tuple[SourceExclusion, ...] = ()


STATE = TypeAdapter(SessionState)
TURN_RESULT = TypeAdapter(TurnResult)
