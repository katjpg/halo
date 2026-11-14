from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from time import monotonic

from rdkit import Chem, rdBase

from halo.chem.cluster import ClusterResult, cluster_butina
from halo.chem.draw import draw_molecule as draw_png
from halo.chem.eval.pareto import MoleculeValues
from halo.chem.eval.pareto import evaluate_pareto as evaluate_values
from halo.chem.eval.profile import profile_molecule as profile_molecule
from halo.chem.eval.profile import profile_molecules as profile_molecules
from halo.chem.fingerprints import Fingerprints
from halo.chem.io import read_smiles as read_smiles
from halo.chem.mol import to_mol
from halo.chem.properties import DESCRIPTORS
from halo.chem.results import Failure, Metric, metric_change
from halo.chem.sar.fragment import fragment_molecule as fragment_molecule
from halo.chem.sar.mcs import MCS_TIMEOUT_S
from halo.chem.sar.mcs import find_mcs as search_mcs
from halo.chem.sar.rgroup import (
    RGroupDecomposition,
    RGroupReplacement,
    decompose_molecule,
)
from halo.chem.sar.rgroup import decompose_rgroups as decompose_series
from halo.core.format import format_table
from halo.core.errors import ParseError
from halo.models.molecule import MoleculeProfile
from halo.types import Direction, MoleculeId

DESCRIPTOR_METRICS = {
    item.name: (item.unit, item.method)
    for item in DESCRIPTORS
    if item.unit is not None and item.method is not None
}


@dataclass(frozen=True)
class CommonCore:
    """An MCS query and the input molecules that match it."""

    smarts: str
    atom_count: int
    matched_indices: tuple[int, ...]
    molecule_count: int

    @property
    def coverage(self) -> float:
        return len(self.matched_indices) / self.molecule_count

    def __str__(self) -> str:
        return (
            f"{self.smarts}\n{self.atom_count} atoms · "
            f"{len(self.matched_indices)}/{self.molecule_count} molecules"
        )


def find_mcs(
    molecules: Sequence[str | Chem.Mol],
    *,
    min_coverage: float = 1.0,
) -> CommonCore | Failure:
    """Find the largest common substructure at the requested coverage.

    Args:
        molecules: At least two molecules as SMILES or RDKit objects.
        min_coverage: Minimum fraction of matching molecules in (0, 1].

    Returns:
        The common core and matching input indices. Return a Failure on
        timeout or when no core matches at least two molecules at that coverage.
    """
    if isinstance(molecules, str):
        raise ParseError("Provide a sequence of at least two molecules.")
    if len(molecules) < 2 or not 0 < min_coverage <= 1:
        raise ParseError("Provide at least two molecules and min_coverage in (0, 1].")
    mols = [to_mol(mol) for mol in molecules]
    n = len(mols)
    required = next(k for k in range(2, n + 1) if k / n >= min_coverage)
    deadline = monotonic() + MCS_TIMEOUT_S
    best = None
    best_size = (0, 0)
    # rdkit can stop at an outlier with no seeds, so search the qualifying subsets
    for subset in combinations(mols, required):
        result = search_mcs(subset, deadline_s=deadline)
        if isinstance(result, Failure):
            return result
        size = (result.numBonds, result.numAtoms)
        if result.queryMol is None or size <= best_size:
            continue
        matched = tuple(
            i for i, mol in enumerate(mols) if mol.HasSubstructMatch(result.queryMol)
        )
        if len(matched) >= required:
            best = CommonCore(result.smartsString, result.numAtoms, matched, n)
            best_size = size
    return best if best is not None else Failure("No core met the requested coverage.")


@dataclass(frozen=True, slots=True)
class CandidateEvaluation:
    """A candidate's values, changes from the source, and Pareto membership."""

    smiles: str
    values: Mapping[str, float]
    delta: Mapping[str, float]
    improved: Mapping[str, bool]
    pareto_optimal: bool
    dominates_source: bool


@dataclass(frozen=True, slots=True)
class ParetoComparison:
    """Objective directions and comparable values for the source and candidates."""

    source_smiles: str
    source_metrics: Mapping[str, Metric]
    objectives: Mapping[str, Direction]
    candidates: tuple[CandidateEvaluation, ...]

    def __str__(self) -> str:
        fields = tuple(self.objectives)
        columns = [
            "SMILES",
            *(f"Δ {field} ({self.source_metrics[field].unit})" for field in fields),
            "Pareto front",
        ]
        rows = [
            [
                candidate.smiles,
                *(f"{candidate.delta[field]:+.3g}" for field in fields),
                "yes" if candidate.pareto_optimal else "no",
            ]
            for candidate in self.candidates
        ]
        return f"Baseline  {self.source_smiles}\n" + format_table(columns, rows)


def cluster_molecules(
    smiles: Sequence[str],
    *,
    min_similarity: float = 0.70,
    include_chirality: bool = True,
) -> ClusterResult:
    """Group molecules by Tanimoto similarity to their Butina seeds.

    Cluster indices refer to the input sequence, including repeats.
    """
    if isinstance(smiles, str) or len(smiles) == 0:
        raise ParseError("Provide a nonempty sequence of SMILES.")
    if not 0 <= min_similarity <= 1:
        raise ParseError("min_similarity must be from 0 to 1")
    fps = Fingerprints.from_mols(
        [to_mol(value) for value in smiles], include_chirality=include_chirality
    )
    return cluster_butina(fps, cutoff=1 - min_similarity)


def evaluate_pareto(
    *,
    source: str | MoleculeProfile,
    candidates: Sequence[str | MoleculeProfile],
    objectives: Mapping[str, Direction],
) -> ParetoComparison:
    """Compare candidates with a source and find the Pareto front.

    Args:
        source: Baseline molecule or profile.
        candidates: Molecules or profiles to compare. Profile SMILES locally.
        objectives: Properties and their preferred directions.

    Returns:
        Distinct candidates in input order, with the source excluded from the
        front. Deltas are candidate minus source in the source metric's units.
    """
    if isinstance(candidates, str):
        raise ParseError("Provide a sequence of candidates, such as ['CCO'].")
    inputs = (source, *candidates)
    computed = iter(
        profile_molecules([item for item in inputs if isinstance(item, str)])
    )
    baseline, *profiles = [
        next(computed) if isinstance(item, str) else item for item in inputs
    ]
    source_metrics = {name: _objective_metric(baseline, name) for name in objectives}
    structures = tuple(
        dict.fromkeys(profile.canonical_smiles for profile in (baseline, *profiles))
    )
    ids = {smiles: MoleculeId(i) for i, smiles in enumerate(structures, start=1)}
    rows = []
    for profile in profiles:
        values = {}
        for name, ref in source_metrics.items():
            metric = _objective_metric(profile, name)
            if metric_change(ref, metric) is None:
                raise ParseError(
                    f"Cannot compare {name!r} for {profile.canonical_smiles}: "
                    "use finite values with the same unit and method as the source."
                )
            values[name] = metric.value
        rows.append(MoleculeValues(ids[profile.canonical_smiles], values))

    evaluations = evaluate_values(
        baseline=MoleculeValues(
            ids[baseline.canonical_smiles],
            {name: metric.value for name, metric in source_metrics.items()},
        ),
        candidates=rows,
        objectives=objectives,
    )
    return ParetoComparison(
        baseline.canonical_smiles,
        source_metrics,
        dict(objectives),
        tuple(
            CandidateEvaluation(
                structures[row.molecule_id - 1],
                row.values,
                row.delta,
                row.improved,
                row.pareto_optimal,
                row.dominates_baseline,
            )
            for row in evaluations
        ),
    )


def _objective_metric(profile: MoleculeProfile, name: str) -> Metric:
    if name in DESCRIPTOR_METRICS:
        unit, method = DESCRIPTOR_METRICS[name]
        return Metric(float(getattr(profile.descriptors, name)), unit, method)
    if name == "qed":
        return Metric(profile.screening.qed, "dimensionless", "rdkit_qed")
    if name == "lipinski_violations":
        return Metric(profile.screening.lipinski_violations, "count", "rdkit_lipinski")
    metric = (
        profile.screening.sa_score
        if name == "sa_score"
        else profile.predictions.get(name)
    )
    if isinstance(metric, Metric):
        return metric
    reason = metric.reason if isinstance(metric, Failure) else "no value in the profile"
    raise ParseError(
        f"Cannot evaluate {name!r} for {profile.canonical_smiles}: {reason}."
    )


def decompose_rgroups(
    smiles: Sequence[str], core: str | Chem.Mol | CommonCore
) -> RGroupDecomposition:
    """Decompose a series on a SMILES core or a substructure query.

    Retain the input indices and report molecules that cannot be rebuilt.
    """
    if isinstance(core, CommonCore):
        core = to_mol(core.smarts, notation="smarts")
    elif isinstance(core, str):
        with rdBase.BlockLogs():
            try:
                core = to_mol(core, allow_dummies=True)
            except ParseError:
                core = to_mol(core, notation="smarts")
    return decompose_series(
        smiles, core, core_notation="smarts" if core.HasQuery() else "smiles"
    )


def replace_rgroup(
    *, smiles: str, core: str, site: int, fragment: str
) -> RGroupReplacement:
    """Replace one mapped R-group and verify the protected structure.

    Use a connected fragment with one attachment point, such as `CCO*`.
    """
    return decompose_molecule(smiles, core).replace(site=site, fragment=fragment)


def draw_molecule(
    value: str | Chem.Mol | CommonCore,
    *,
    width: int = 300,
    height: int = 300,
    path: str | Path | None = None,
) -> bytes:
    """Draw a SMILES, InChI, molfile text, or RDKit molecule as PNG.

    Width and height are in pixels. An existing file at path is replaced.
    """
    notation = "smiles"
    if isinstance(value, CommonCore):
        notation = "smarts"
        value = value.smarts
    elif isinstance(value, str):
        if value.lstrip().startswith("InChI="):
            notation = "inchi"
        elif "\nM  END" in value:
            notation = "molblock"
    elif value.HasQuery():
        notation = "smarts"
    image = draw_png(value, notation=notation, width=width, height=height)
    if path is not None:
        Path(path).write_bytes(image)
    return image
