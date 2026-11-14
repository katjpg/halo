from collections.abc import Sequence
from dataclasses import dataclass
from time import monotonic
from typing import Literal

from rdkit import Chem, DataStructs
from rdkit.Chem.Scaffolds import MurckoScaffold

from halo.chem.fingerprints import Fingerprints
from halo.chem.mol import canonical_structure, to_mol
from halo.chem.results import Failure
from halo.chem.sar.mcs import find_mcs
from halo.chem.sar.sites import MAX_CORE_MATCHES
from halo.core.errors import ParseError

SiteChange = Literal["addition", "deletion", "substitution"]
TransformationType = Literal[
    "no_change",
    "addition",
    "deletion",
    "substitution",
    "complex",
    "no_common_core",
    "unmapped_change",
]


@dataclass(frozen=True, slots=True)
class MCSCounts:
    atom_count: int
    bond_count: int


@dataclass(frozen=True, slots=True)
class StructuralComparison:
    """Fingerprint similarity and common substructure of two molecules.

    Tanimoto ranges from 0 to 1. Atom counts exclude ordinary hydrogens.
    The MCS result carries a Failure when the search times out.
    """

    source_smiles: str
    candidate_smiles: str
    source_atom_count: int
    candidate_atom_count: int
    tanimoto: float
    fingerprint_method: str
    mcs: MCSCounts | Failure

    @property
    def mcs_complete(self) -> bool:
        return isinstance(self.mcs, MCSCounts)

    @property
    def mcs_atom_count(self) -> int | None:
        return self.mcs.atom_count if isinstance(self.mcs, MCSCounts) else None

    @property
    def mcs_bond_count(self) -> int | None:
        return self.mcs.bond_count if isinstance(self.mcs, MCSCounts) else None

    @property
    def source_unmatched_atoms(self) -> int | None:
        return (
            self.source_atom_count - self.mcs.atom_count
            if isinstance(self.mcs, MCSCounts)
            else None
        )

    @property
    def candidate_unmatched_atoms(self) -> int | None:
        return (
            self.candidate_atom_count - self.mcs.atom_count
            if isinstance(self.mcs, MCSCounts)
            else None
        )


@dataclass(frozen=True, slots=True)
class FragmentChange:
    """The fragments before and after a change at a common-core site.

    Site is the core atom index plus one. Hydrogen is represented as
    "[*:n][H]" with n set to the site label.
    """

    site: int
    change_type: SiteChange
    source: str
    target: str


@dataclass(frozen=True, slots=True)
class Transformation:
    """Changes at a shared core and their overall type.

    Complex covers mixed changes and sites that cannot be split into
    separate fragments. Unmapped changes leave the mapped heavy-atom
    fragments unchanged. Ambiguous marks a reached match limit.
    """

    transformation_type: TransformationType
    changes: tuple[FragmentChange, ...]
    ambiguous: bool = False


TransformationResult = Transformation | Failure


def protected_cores(source_smiles: str) -> tuple[Chem.Mol, ...]:
    """Build chiral queries for the source framework.

    Keep the ring scaffold and any paths to specified stereocenters.
    For acyclic structures, allow one terminal atom outside a stereo group
    to change when at least two heavy atoms remain.
    """
    source, _ = canonical_structure(to_mol(source_smiles))
    if len(Chem.GetMolFrags(source)) != 1:
        raise ParseError("Choose a connected source molecule for generation.")

    stereo_atoms = set()
    for atom in source.GetAtoms():
        if atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED:
            stereo_atoms.add(atom.GetIdx())
            stereo_atoms.update(neighbor.GetIdx() for neighbor in atom.GetNeighbors())
    for bond in source.GetBonds():
        if stereo := tuple(bond.GetStereoAtoms()):
            stereo_atoms.update((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), *stereo))

    mapped = Chem.Mol(source)
    for atom in mapped.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx() + 1)
    core = {
        atom.GetAtomMapNum() - 1
        for atom in MurckoScaffold.GetScaffoldForMol(mapped).GetAtoms()
    }
    if core:
        root = min(core)
        for idx in stereo_atoms - core:
            core.update(Chem.GetShortestPath(source, root, idx))
        cores = [core]
    elif source.GetNumHeavyAtoms() > 2:
        atoms = set(range(source.GetNumAtoms()))
        cores = [
            atoms - {atom.GetIdx()}
            for atom in source.GetAtoms()
            if atom.GetAtomicNum() > 1
            and atom.GetDegree() == 1
            and atom.GetIdx() not in stereo_atoms
        ] or [atoms]
    else:
        cores = [set(range(source.GetNumAtoms()))]

    return tuple(
        to_mol(
            Chem.MolFragmentToSmarts(source, sorted(indices), isomericSmarts=True),
            notation="smarts",
        )
        for indices in cores
    )


def verify_analog(smiles: str, cores: Sequence[Chem.Mol]) -> str:
    """Return a connected analog's canonical SMILES if a protected core matches."""
    product, canonical = canonical_structure(to_mol(smiles))
    if len(Chem.GetMolFrags(product)) != 1:
        raise ParseError("An analog needs a single connected structure.")
    if not any(product.HasSubstructMatch(core, useChirality=True) for core in cores):
        raise ParseError("The analog changed the source scaffold or stereochemistry.")
    return canonical


def compare_structures(
    source_smiles: str,
    candidate_smiles: Sequence[str],
    *,
    timeout_s: int = 2,
) -> list[StructuralComparison]:
    """Compare each candidate with the source using Morgan fingerprints and MCS.

    Compare each distinct structure once. All searches share timeout_s
    seconds. Return results in input order, including repeats.
    """
    if timeout_s < 1:
        raise ParseError("timeout_s must be at least 1")
    source = to_mol(source_smiles)
    key_by_text: dict[str, str] = {}
    structures: dict[str, Chem.Mol] = {}
    for text in dict.fromkeys(candidate_smiles):
        mol, key = canonical_structure(to_mol(text))
        key_by_text[text] = key
        structures.setdefault(key, mol)
    if not structures:
        return []

    source, source_key = canonical_structure(source)
    fps = Fingerprints.from_mols([source, *structures.values()], prepared=True)
    similarities = DataStructs.BulkTanimotoSimilarity(fps.bits[0], fps.bits[1:])
    comparisons = {}
    deadline_s = monotonic() + timeout_s
    for (key, mol), similarity in zip(structures.items(), similarities, strict=True):
        mcs = find_mcs([source, mol], timeout_s=timeout_s, deadline_s=deadline_s)
        comparisons[key] = StructuralComparison(
            source_smiles=source_key,
            candidate_smiles=key,
            source_atom_count=source.GetNumAtoms(),
            candidate_atom_count=mol.GetNumAtoms(),
            tanimoto=float(similarity),
            fingerprint_method=fps.method,
            mcs=mcs
            if isinstance(mcs, Failure)
            else MCSCounts(mcs.numAtoms, mcs.numBonds),
        )
    return [comparisons[key_by_text[text]] for text in candidate_smiles]


def compare_fragments(
    source_smiles: str, target_smiles: str, *, timeout_s: int = 3
) -> TransformationResult:
    """Compare attachment sites on the maximum common substructure.

    Use the target orientation with the fewest changes. Ties keep the
    first match. Return a Failure on MCS timeout.
    """
    if timeout_s < 1:
        raise ParseError("timeout_s must be at least 1")
    source, source_key = canonical_structure(to_mol(source_smiles))
    target, target_key = canonical_structure(to_mol(target_smiles))
    if source_key == target_key:
        return Transformation("no_change", ())
    mcs = find_mcs([source, target], timeout_s=timeout_s)
    if isinstance(mcs, Failure):
        return mcs
    core = Chem.MolFromSmarts(mcs.smartsString) if mcs.smartsString else None
    if core is None:
        return Transformation("no_common_core", ())

    before = _site_fragments(source, core, source.GetSubstructMatch(core))
    if before is None:
        return Transformation("complex", ())
    matches = target.GetSubstructMatches(
        core, uniquify=False, maxMatches=MAX_CORE_MATCHES
    )
    ambiguous = len(matches) == MAX_CORE_MATCHES
    best: list[FragmentChange] | None = None
    for match in matches:
        after = _site_fragments(target, core, match)
        if after is None:
            continue
        changes = _site_changes(before, after)
        if not changes:
            return Transformation("unmapped_change", (), ambiguous)
        if best is None or len(changes) < len(best):
            best = changes
    if best is None:
        return Transformation("complex", (), ambiguous)
    change_types = {change.change_type for change in best}
    transformation_type = change_types.pop() if len(change_types) == 1 else "complex"
    return Transformation(transformation_type, tuple(best), ambiguous)


def _site_fragments(
    mol: Chem.Mol, core: Chem.Mol, match: Sequence[int]
) -> dict[int, tuple[str, int]] | None:
    """Map core sites to fragment SMILES and heavy-atom counts.

    Return None for a fragment spanning sites or a site with several fragments.
    """
    replaced = Chem.ReplaceCore(mol, core, match, labelByIndex=True)
    if replaced is None:
        return None
    for atom in replaced.GetAtoms():
        if atom.GetAtomicNum() == 0:  # isotope = core atom index; site = index + 1
            atom.SetAtomMapNum(atom.GetIsotope() + 1)
            atom.SetIsotope(0)
    sites: dict[int, tuple[str, int]] = {}
    for fragment in Chem.GetMolFrags(replaced, asMols=True, sanitizeFrags=False):
        dummies = [atom for atom in fragment.GetAtoms() if atom.GetAtomicNum() == 0]
        if len(dummies) != 1 or dummies[0].GetAtomMapNum() in sites:
            return None
        sites[dummies[0].GetAtomMapNum()] = (
            Chem.MolToSmiles(fragment),
            fragment.GetNumHeavyAtoms(),
        )
    return sites


def _site_changes(
    before: dict[int, tuple[str, int]], after: dict[int, tuple[str, int]]
) -> list[FragmentChange]:
    changes = []
    for site in sorted(before.keys() | after.keys()):
        hydrogen = (f"[*:{site}][H]", 0)
        (source, source_heavy), (target, target_heavy) = (
            before.get(site, hydrogen),
            after.get(site, hydrogen),
        )
        if source == target or not (source_heavy or target_heavy):
            continue  # unchanged, or a hydrogen-only difference such as an isotope
        if not source_heavy:
            change_type: SiteChange = "addition"
        elif not target_heavy:
            change_type = "deletion"
        else:
            change_type = "substitution"
        changes.append(FragmentChange(site, change_type, source, target))
    return changes
