import json
from dataclasses import dataclass
from datetime import datetime
from typing import Annotated, Literal

from pydantic import AliasChoices, Field, computed_field, model_validator

from halo.types import ADMETProperty, AssayResultId, DecisionId, MoleculeId, ProjectId

AssayRelation = Literal["=", "<", "<=", ">", ">="]
ResultRelation = Literal[AssayRelation, "~", "<<", ">>"]
MCSStatus = Literal["found", "none", "timeout", "skipped"]


@dataclass(frozen=True)
class AssayResult:
    """One reported measurement for a molecule in a project."""

    id: AssayResultId
    project_id: ProjectId
    molecule_id: MoleculeId
    assay: Annotated[
        str,
        Field(
            validation_alias=AliasChoices("assay", "assay_id"),
            serialization_alias="assay_id",
        ),
    ]
    endpoint: str
    relation: AssayRelation
    value: float
    unit: str
    conditions: dict[str, object]
    source_ref: str
    recorded_at: datetime
    decision_id: DecisionId | None = None


@dataclass(frozen=True, kw_only=True)
class ResultValue:
    id: str
    molecule_id: MoleculeId
    endpoint: str
    value: float
    unit: str
    method: str
    source_type: Literal["predicted", "experimental"]
    relation: ResultRelation = "="
    assay_id: str | None = None
    target: str | None = None
    assay_description: str | None = None
    source_ref: str | None = None
    conditions: dict[str, object] | None = None


def assay_signature(result: ResultValue) -> tuple[str, ...] | None:
    """Identify exact measurements that can be compared under recorded conditions."""
    if (
        result.source_type != "experimental"
        or result.relation != "="
        or not result.assay_id
        or not result.conditions
    ):
        return None
    return (
        result.assay_id,
        result.endpoint.casefold(),
        result.method,
        result.unit,
        json.dumps(result.conditions, sort_keys=True),
    )


@dataclass(frozen=True)
class PlotSpec:
    x: ADMETProperty
    y: ADMETProperty
    selected_only: bool = False

    def __post_init__(self) -> None:
        if self.x == self.y:
            raise ValueError("Choose different properties for the two axes.")


@dataclass(frozen=True)
class PlotPoint:
    molecule_id: MoleculeId
    x: float
    y: float
    result_ids: tuple[str, str]


@dataclass(frozen=True, kw_only=True)
class Axis:
    field: ADMETProperty
    unit: str
    method: str
    lo: float
    hi: float


@dataclass(frozen=True)
class ScatterPlot:
    spec: PlotSpec
    source: PlotPoint
    points: tuple[PlotPoint, ...]
    visible_ids: tuple[MoleculeId, ...]
    x_axis: Annotated[Axis, Field(exclude=True)]
    y_axis: Annotated[Axis, Field(exclude=True)]

    def __post_init__(self) -> None:
        if (self.x_axis.field, self.y_axis.field) != (self.spec.x, self.spec.y):
            raise ValueError("The plot axes must match its properties.")

    @model_validator(mode="before")
    @classmethod
    def read_axes(cls, value: object) -> object:
        if not isinstance(value, dict) or "axes" not in value:
            return value
        x, y = (
            {"field": field, "unit": unit, "method": method, "lo": lo, "hi": hi}
            for field, unit, method, (lo, hi) in zip(
                (value["spec"]["x"], value["spec"]["y"]),
                value["units"],
                value["methods"],
                value["axes"],
                strict=True,
            )
        )
        return value | {"x_axis": x, "y_axis": y}

    @computed_field
    @property
    def axes(self) -> tuple[tuple[float, float], tuple[float, float]]:
        return (self.x_axis.lo, self.x_axis.hi), (self.y_axis.lo, self.y_axis.hi)

    @computed_field
    @property
    def methods(self) -> tuple[str, str]:
        return self.x_axis.method, self.y_axis.method

    @computed_field
    @property
    def units(self) -> tuple[str, str]:
        return self.x_axis.unit, self.y_axis.unit


@dataclass(frozen=True)
class ClusterGroup:
    molecule_ids: tuple[MoleculeId, ...]
    representative_id: MoleculeId
    mcs_smarts: str | None
    mcs_status: MCSStatus
    fragments: dict[int, tuple[str, ...]]

    def __post_init__(self) -> None:
        if (self.mcs_status == "found") != bool(self.mcs_smarts):
            raise ValueError("A found core needs SMARTS.")

    @model_validator(mode="before")
    @classmethod
    def read_status(cls, value: object) -> object:
        if not isinstance(value, dict) or "mcs_status" in value:
            return value
        if value.get("mcs_complete"):
            status = "found" if value.get("mcs_smarts") else "none"
        else:
            status = "timeout" if len(value["molecule_ids"]) > 1 else "skipped"
        return value | {"mcs_status": status}

    @computed_field
    @property
    def mcs_complete(self) -> bool:
        return self.mcs_status in ("found", "none")


@dataclass(frozen=True)
class ClusterAnalysis:
    groups: tuple[ClusterGroup, ...]
    fingerprint_method: str
    cutoff: float


@dataclass(frozen=True)
class MoleculeAnnotation:
    molecule_id: MoleculeId
    groups: tuple[str, ...]
    alerts: dict[str, tuple[str, ...]]
