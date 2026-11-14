from collections.abc import Callable, Mapping
from typing import Annotated, Literal, NewType
from uuid import UUID

from pydantic import AfterValidator, Field, StringConstraints

MAX_RECORD_ID = 2**63 - 1
MAX_BATCH = 6
DEFAULT_BATCH_SIZE = 4
MAX_SELECTION = 60
MAX_RECALL = 4

UserId = NewType("UserId", str)
MoleculeId = NewType("MoleculeId", int)
GenerationRunId = NewType("GenerationRunId", str)
ProjectId = NewType("ProjectId", str)
SessionId = NewType("SessionId", str)
TurnId = NewType("TurnId", int)
DecisionId = NewType("DecisionId", int)
AssayResultId = NewType("AssayResultId", int)

AgentRole = Literal["orchestrator", "researcher", "strategist", "analyst", "optimizer"]
Progress = Callable[[str], None]

RECORD_ID = Field(strict=True, gt=0, le=MAX_RECORD_ID)
BatchSize = Annotated[int, Field(strict=True, ge=1, le=MAX_BATCH)]
FiniteFloat = Annotated[float, Field(allow_inf_nan=False)]


def nonblank(text: str) -> str:
    if not text.strip():
        raise ValueError("Provide nonempty text.")
    return text


def validate_uuid(value: str) -> str:
    UUID(value)
    return value


Note = Annotated[
    str, StringConstraints(min_length=1, max_length=2000), AfterValidator(nonblank)
]
Message = Annotated[
    str, StringConstraints(min_length=1, max_length=3000), AfterValidator(nonblank)
]
AssayName = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=128)
]
AssayEndpoint = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=80)
]
UUID_STRING = AfterValidator(validate_uuid)

Direction = Literal["increase", "decrease"]
CriterionDirection = Literal[Direction, "preserve"]
ADMETProperty = Literal[
    "bbbp",
    "carcinogenicity",
    "dili",
    "herg",
    "hia",
    "mutagenicity",
    "pampa",
    "solubility",
]
Criteria = Mapping[ADMETProperty, CriterionDirection]
DesignCriteria = Annotated[Criteria, Field(min_length=1)]
OptimizationTask = Literal[ADMETProperty, "drd2", "penalized_logp", "qed"]
ApplicationErrorCode = Literal[
    "internal_error",
    "configuration_error",
    "parse_error",
    "unauthenticated",
    "provider_error",
    "limit_exceeded",
    "not_found",
    "conflict",
]

ReasoningEffort = Literal["none", "minimal", "low", "medium", "high", "xhigh", "max"]
