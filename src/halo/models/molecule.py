from collections.abc import Mapping
from dataclasses import dataclass

from halo.chem.results import MetricResult, Prediction
from halo.types import OptimizationTask


@dataclass(frozen=True, slots=True)
class Descriptors:
    """RDKit descriptors after removing ordinary hydrogens.

    Uses average molecular weight, Wildman–Crippen logP, and Ertl TPSA for N and O.
    """

    molecular_weight_g_mol: float
    logp: float
    tpsa_angstrom_sq: float
    heavy_atom_count: int
    h_bond_acceptors: int
    h_bond_donors: int
    rotatable_bonds: int
    ring_count: int
    aromatic_rings: int


@dataclass(frozen=True, slots=True)
class Screening:
    """Drug-likeness scores and PAINS alerts.

    Attributes:
        qed: Drug-likeness from 0 to 1, with higher values preferred.
        lipinski_violations: Failed rules, from 0 to 4. Counts use N + O and NH + OH.
        sa_score: Synthetic accessibility from 1 (easy) to 10 (hard).
        pains_alerts: Distinct alert names in catalog order.
    """

    qed: float
    lipinski_violations: int
    sa_score: MetricResult
    pains_alerts: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class MoleculeProfile:
    """Descriptors, screening scores, and predictions for one structure.

    Missing properties have no entry. A Failure records an unsuccessful prediction.
    """

    canonical_smiles: str
    descriptors: Descriptors
    screening: Screening
    predictions: Mapping[OptimizationTask, MetricResult]

    def prediction(self, name: OptimizationTask) -> Prediction:
        """Read a prediction with its status, value, and method."""
        return Prediction(self.predictions.get(name))
