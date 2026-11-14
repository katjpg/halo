import json
from typing import Annotated

from pydantic import BaseModel, Field, JsonValue, StringConstraints, field_validator

from halo.models.analysis import AssayRelation
from halo.models.research import Response, SourceSearch
from halo.types import (
    RECORD_ID,
    AssayEndpoint,
    AssayName,
    DecisionId,
    Message,
    MoleculeId,
)


class ResearchRequest(BaseModel, extra="forbid"):
    molecule_id: Annotated[MoleculeId, RECORD_ID]
    question: Message


class ResearchResponse(BaseModel):
    sources: SourceSearch
    response: Response


class AssayResultRequest(BaseModel, extra="forbid"):
    molecule_id: Annotated[MoleculeId, RECORD_ID]
    assay_id: AssayName = Field(serialization_alias="assay")
    endpoint: AssayEndpoint
    relation: AssayRelation = "="
    value: float = Field(strict=True, allow_inf_nan=False)
    unit: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1, max_length=40)
    ]
    conditions: dict[str, JsonValue]
    source_ref: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1, max_length=256)
    ]
    decision_id: Annotated[DecisionId, RECORD_ID] | None = None

    @field_validator("conditions")
    @classmethod
    def check_conditions(cls, conditions: dict[str, JsonValue]) -> dict[str, JsonValue]:
        encoded = json.dumps(conditions, ensure_ascii=False, allow_nan=False).encode()
        if len(encoded) > 1024:
            raise ValueError("Keep assay conditions within 1024 bytes.")
        return conditions
