from typing import Annotated, Literal

from pydantic import AfterValidator, BaseModel, Field, StringConstraints

from halo.chem.results import Failure, Metric, MetricResult, Prediction
from halo.models.molecule import Descriptors, MoleculeProfile
from halo.types import MoleculeId, nonblank


class ComputedMetric(BaseModel):
    status: Literal["computed"]
    value: float
    unit: str
    method: str


class FailedMetric(BaseModel):
    status: Literal["failed"]
    reason: str


MetricValue = Annotated[ComputedMetric | FailedMetric, Field(discriminator="status")]


class ScreeningValues(BaseModel):
    qed: float
    lipinski_violations: int
    sa_score: MetricValue
    pains_alerts: list[str]
    pains_pass: bool


class ProfileResponse(BaseModel):
    molecule_id: int
    smiles: str
    descriptors: Descriptors
    screening: ScreeningValues
    predictions: dict[str, MetricValue]

    def prediction(self, name: str) -> Prediction:
        """Read the prediction saved on the molecular profile."""
        return prediction_result(self.predictions.get(name))


class MoleculeRequest(BaseModel, extra="forbid"):
    smiles: Annotated[
        str, StringConstraints(min_length=1, max_length=2000), AfterValidator(nonblank)
    ]


def metric_value(result: MetricResult) -> MetricValue:
    if isinstance(result, Metric):
        return ComputedMetric(
            status="computed",
            value=result.value,
            unit=result.unit,
            method=result.method,
        )
    return FailedMetric(status="failed", reason=result.reason)


def prediction_result(value: MetricValue | None) -> Prediction:
    if isinstance(value, ComputedMetric):
        return Prediction(Metric(value.value, value.unit, value.method))
    return Prediction(Failure(value.reason) if value is not None else None)


def profile_response(
    molecule_id: MoleculeId, profile: MoleculeProfile
) -> ProfileResponse:
    screening = profile.screening
    return ProfileResponse(
        molecule_id=int(molecule_id),
        smiles=profile.canonical_smiles,
        descriptors=profile.descriptors,
        screening=ScreeningValues(
            qed=screening.qed,
            lipinski_violations=screening.lipinski_violations,
            sa_score=metric_value(screening.sa_score),
            pains_alerts=list(screening.pains_alerts),
            pains_pass=not screening.pains_alerts,
        ),
        predictions={
            name: metric_value(result) for name, result in profile.predictions.items()
        },
    )
