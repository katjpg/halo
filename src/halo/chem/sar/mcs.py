from collections.abc import Sequence, Set
from dataclasses import dataclass
from time import monotonic

from rdkit import Chem
from rdkit.Chem import rdFMCS

from halo.chem.results import Failure
from halo.core.errors import ParseError
from halo.types import MoleculeId

MCS_TIMEOUT_S = 10


@dataclass(frozen=True, slots=True)
class CoreCandidate:
    smiles: str
    member_ids: frozenset[MoleculeId]
    heavy_atom_count: int

    @property
    def member_count(self) -> int:
        return len(self.member_ids)


@dataclass(frozen=True, slots=True)
class CoreAssignment:
    """A selected scaffold with its newly assigned molecule IDs in sorted order."""

    core_smiles: str
    member_ids: tuple[MoleculeId, ...]


def find_mcs(
    mols: Sequence[Chem.Mol],
    *,
    timeout_s: int = MCS_TIMEOUT_S,
    threshold: float = 1.0,
    deadline_s: float | None = None,
) -> rdFMCS.MCSResult | Failure:
    """Find the maximum common substructure of at least two molecules.

    Match atoms by element and bonds by order, with complete rings.
    Threshold is the fraction of molecules that must match, in (0, 1].
    A monotonic deadline can share the budget across searches. Return a
    Failure if the search times out.
    """
    return _find_mcs(mols, timeout_s, threshold, deadline_s=deadline_s)


class _Deadline(rdFMCS.MCSProgress):
    def __init__(self, deadline_s: float) -> None:
        super().__init__()
        self.deadline_s = deadline_s

    def __call__(
        self, progress: rdFMCS.MCSProgressData, parameters: rdFMCS.MCSParameters
    ) -> bool:
        return monotonic() < self.deadline_s


def _find_mcs(
    mols: Sequence[Chem.Mol],
    timeout_s: int,
    threshold: float,
    *,
    deadline_s: float | None = None,
    store_all: bool = False,
) -> rdFMCS.MCSResult | Failure:
    if len(mols) < 2 or any(not mol.GetNumAtoms() for mol in mols):
        raise ParseError("mols must contain at least two nonempty molecules")
    if timeout_s < 1:
        raise ParseError("timeout_s must be at least 1")
    if not 0 < threshold <= 1:
        raise ParseError("threshold must be in (0, 1]")
    failure = Failure(f"MCS search exceeded the {timeout_s} s budget.")
    if deadline_s is not None and monotonic() >= deadline_s:
        return failure
    params = rdFMCS.MCSParameters()
    params.Timeout = timeout_s
    params.Threshold = threshold
    params.StoreAll = store_all
    params.AtomTyper = rdFMCS.AtomCompare.CompareElements
    params.BondTyper = rdFMCS.BondCompare.CompareOrder
    params.BondCompareParameters.RingMatchesRingOnly = True
    params.BondCompareParameters.CompleteRingsOnly = True
    if deadline_s is not None:
        params.ProgressCallback = _Deadline(deadline_s)
    result = rdFMCS.FindMCS(list(mols), params)
    return failure if result.canceled else result


def find_mcs_degenerate(
    mols: Sequence[Chem.Mol],
    *,
    timeout_s: int = MCS_TIMEOUT_S,
    threshold: float = 1.0,
    max_results: int = 50,
) -> list[str] | Failure:
    """Return sorted SMARTS for distinct maximum matches reported by RDKit.

    Cap the list at max_results. Return a Failure on timeout.
    """
    if max_results < 1:
        raise ParseError("max_results must be positive")
    result = _find_mcs(mols, timeout_s, threshold, store_all=True)
    if isinstance(result, Failure):
        return result
    if not result.numAtoms:
        return []
    return sorted(result.degenerateSmartsQueryMolDict)[:max_results]


def select_covering_cores(
    candidates: Sequence[CoreCandidate],
    molecule_ids: Set[MoleculeId],
    *,
    max_cores: int = 3,
    min_additional_coverage: int = 2,
) -> list[CoreAssignment]:
    """Pick the scaffold that covers the most unassigned molecules at each step.

    Ties favor the first candidate. Stop at max_cores or when the best
    scaffold adds fewer than min_additional_coverage molecules.
    """
    if max_cores < 0 or min_additional_coverage < 1:
        raise ParseError("invalid core-selection limits")
    coverage = [candidate.member_ids & molecule_ids for candidate in candidates]
    uncovered = set(molecule_ids)
    selected: list[CoreAssignment] = []
    while uncovered and len(selected) < max_cores:
        best_idx, best_gain = -1, set()
        for i, members in enumerate(coverage):
            gain = members & uncovered
            if len(gain) > len(best_gain):
                best_idx, best_gain = i, gain
        if len(best_gain) < min_additional_coverage:
            break
        selected.append(
            CoreAssignment(candidates[best_idx].smiles, tuple(sorted(best_gain)))
        )
        uncovered -= best_gain
    return selected
