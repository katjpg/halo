from datetime import datetime
from typing import Annotated, Self
from uuid import UUID

from pydantic import AliasChoices, BaseModel, Field, model_validator

from halo.models.research import Response
from halo.models.session import (
    Action,
    Task,
    TurnErrorCode,
    TurnRecord,
    TurnResult,
    TurnStatus,
)
from halo.types import (
    RECORD_ID,
    DecisionId,
    GenerationRunId,
    Message,
    MoleculeId,
    SessionId,
    TurnId,
)


class ProjectRequest(BaseModel, extra="forbid"):
    source_molecule_id: Annotated[MoleculeId, RECORD_ID]


class TurnRequest(BaseModel, extra="forbid"):
    request_id: UUID | None = None
    message: Message | None = None
    action: Action | None = None

    @model_validator(mode="after")
    def check_request(self) -> Self:
        if (self.message is None) == (self.action is None):
            raise ValueError("Provide a message or an action.")
        return self


class TurnMetadata(BaseModel, from_attributes=True):
    id: TurnId
    session_id: SessionId
    turn_index: int
    status: TurnStatus
    request: dict
    request_id: UUID | None
    started_at: datetime
    finished_at: datetime | None

    @classmethod
    def from_record(cls, turn: TurnRecord) -> Self:
        return cls.model_validate(turn)


class TurnSummary(BaseModel, from_attributes=True):
    task: Task | None = None
    message: str | None = None
    response: Response | None = None
    generation_run_id: GenerationRunId | None = None
    decision_id: DecisionId | None = None
    error: str | None = None
    error_code: TurnErrorCode | None = None


class TurnSummaryResponse(TurnMetadata):
    result: TurnSummary | None = Field(
        validation_alias=AliasChoices("result", "output")
    )


class TurnResponse(TurnMetadata):
    result: TurnResult | None = Field(validation_alias=AliasChoices("result", "output"))
    trace: dict | None
