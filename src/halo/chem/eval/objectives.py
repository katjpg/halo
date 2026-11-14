from dataclasses import dataclass
from math import isfinite

from halo.chem.results import Metric, MetricResult
from halo.models.molecule import MoleculeProfile
from halo.types import ADMETProperty as ADMETProperty
from halo.types import Criteria as Criteria
from halo.types import CriterionDirection as CriterionDirection
from halo.types import Direction as Direction
from halo.types import OptimizationTask as OptimizationTask


@dataclass(frozen=True, slots=True)
class Objective:
    label: str
    direction: Direction


ADMET_UNITS: dict[ADMETProperty, str] = {
    "bbbp": "probability",
    "carcinogenicity": "probability",
    "dili": "probability",
    "herg": "probability",
    "hia": "probability",
    "mutagenicity": "probability",
    "pampa": "probability",
    "solubility": "log10 mol/L",
}


OBJECTIVES: dict[OptimizationTask, Objective] = {
    "bbbp": Objective("predicted probability of BBB permeability", "increase"),
    "carcinogenicity": Objective(
        "predicted probability of carcinogenicity", "decrease"
    ),
    "dili": Objective("predicted probability of drug-induced liver injury", "decrease"),
    "drd2": Objective("predicted probability of DRD2 activity", "increase"),
    "herg": Objective("predicted probability of hERG block", "decrease"),
    "hia": Objective("predicted probability of intestinal absorption", "increase"),
    "mutagenicity": Objective("predicted probability of mutagenicity", "decrease"),
    "pampa": Objective("predicted probability of high PAMPA permeability", "increase"),
    "penalized_logp": Objective("penalized logP score", "increase"),
    "qed": Objective("QED", "increase"),
    "solubility": Objective("predicted aqueous solubility", "increase"),
}


def is_current(
    key: ADMETProperty, result: MetricResult | None, *, method: str | None = None
) -> bool:
    return (
        isinstance(result, Metric)
        and result.unit == ADMET_UNITS[key]
        and (method is None or result.method == method)
        and isfinite(result.value)
        and (result.unit != "probability" or 0 <= result.value <= 1)
    )


def extract_source_values(profile: MoleculeProfile) -> dict[OptimizationTask, float]:
    """Use screening QED and the available predictions for other objectives."""
    return {
        task: result.value
        for task in OBJECTIVES
        if isinstance(result := objective_metric(profile, task), Metric)
    }


def objective_metric(
    profile: MoleculeProfile, task: OptimizationTask
) -> MetricResult | None:
    """Read the profile value used to evaluate an objective."""
    if task == "qed":
        return Metric(profile.screening.qed, "score", "RDKit QED")
    return profile.predictions.get(task)
