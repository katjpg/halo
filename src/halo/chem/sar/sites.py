from bisect import bisect_left
from collections import defaultdict, deque
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from rdkit import Chem

from halo.chem.mol import attachment_atoms, to_mol
from halo.core.errors import ParseError

MAX_CORE_MATCHES = 1000


@dataclass(frozen=True, slots=True)
class AtomSelection:
    """Atom and bond indices, both in the atom order of RGroupMatch.molfile."""

    atoms: tuple[int, ...]
    bonds: tuple[int, ...]


@dataclass(frozen=True)
class RGroupMatch:
    """A core match and its site selections in molfile atom order.

    Sites include bonds to the core. Unassigned atoms can include
    counterions. Ambiguous marks tied mappings or a reached match limit.
    """

    core: AtomSelection
    sites: Mapping[int, AtomSelection]
    molfile: str
    unassigned: AtomSelection
    ambiguous: bool


def match_rgroup_atoms(
    molecule: str | Chem.Mol,
    core: str,
    *,
    core_notation: Literal["smiles", "smarts"] = "smiles",
    max_matches: int = MAX_CORE_MATCHES,
) -> RGroupMatch:
    """Select core and substituent atoms for highlighting.

    Prefer the match with the most occupied labeled sites. Ties keep
    the first. Number unlabeled sites from 1 in atom order. With a
    labeled core, substituents at unlabeled sites stay unassigned.

    Selections use molfile atom order. With no match, every atom is
    unassigned. Shared attachment anchors can raise ParseError.
    """
    if max_matches < 1:
        raise ParseError("max_matches must be positive")
    mol = to_mol(molecule)
    query, anchors = _strip_dummies(
        to_mol(core, notation=core_notation, allow_dummies=True)
    )
    if not query.GetNumAtoms():
        raise ParseError("core must contain atoms beyond its attachment points")
    matches = mol.GetSubstructMatches(
        query, uniquify=False, useChirality=True, maxMatches=max_matches
    )
    molfile = Chem.MolToMolBlock(mol)
    if not matches:
        return RGroupMatch(
            core=AtomSelection((), ()),
            sites={},
            molfile=molfile,
            unassigned=AtomSelection(
                tuple(range(mol.GetNumAtoms())), tuple(range(mol.GetNumBonds()))
            ),
            ambiguous=False,
        )

    selected, tied = _best_match(mol, matches, anchors)
    core_atoms = set(selected)
    sites, unassigned = _assign_sites(
        mol, core_atoms, {label: selected[i] for label, i in anchors.items()}
    )
    core_bonds, site_bonds = _selection_bonds(
        mol, core_atoms, sites | {None: unassigned}
    )
    return RGroupMatch(
        core=AtomSelection(tuple(sorted(core_atoms)), tuple(core_bonds)),
        sites={
            label: AtomSelection(tuple(sorted(atoms)), tuple(site_bonds.get(label, ())))
            for label, atoms in sorted(sites.items())
        },
        molfile=molfile,
        unassigned=AtomSelection(
            tuple(sorted(unassigned)), tuple(site_bonds.get(None, ()))
        ),
        ambiguous=tied or len(matches) == max_matches,
    )


def _best_match(
    mol: Chem.Mol, matches: tuple[tuple[int, ...], ...], anchors: Mapping[int, int]
) -> tuple[tuple[int, ...], bool]:
    def filled_sites(match: tuple[int, ...]) -> int:
        matched = set(match)
        return sum(
            any(
                neighbor.GetIdx() not in matched
                for neighbor in mol.GetAtomWithIdx(match[anchor]).GetNeighbors()
            )
            for anchor in anchors.values()
        )

    scores = [filled_sites(match) for match in matches]
    best = max(range(len(matches)), key=scores.__getitem__)
    ordered_anchors = tuple(idx for _, idx in sorted(anchors.items()))
    signatures = {
        (frozenset(match), tuple(match[idx] for idx in ordered_anchors))
        for match, score in zip(matches, scores, strict=True)
        if score == scores[best]
    }
    return matches[best], len(signatures) > 1


def _assign_sites(
    mol: Chem.Mol, core_atoms: set[int], anchors: Mapping[int, int]
) -> tuple[dict[int, set[int]], set[int]]:
    labels_by_atom: dict[int, set[int]] = defaultdict(set)
    for label, idx in anchors.items():
        labels_by_atom[idx].add(label)

    sites: dict[int, set[int]] = defaultdict(set)
    unassigned: set[int] = set()
    next_label = 1
    remaining = set(range(mol.GetNumAtoms())) - core_atoms
    for component in _connected_components(mol, remaining):
        connected = {
            idx
            for atom_idx in component
            for neighbor in mol.GetAtomWithIdx(atom_idx).GetNeighbors()
            if (idx := neighbor.GetIdx()) in core_atoms
        }
        labels: set[int] = set()
        for idx in connected:
            at_anchor = labels_by_atom[idx]
            if len(at_anchor) > 1:
                raise ParseError("R-group sites share an ambiguous attachment anchor")
            labels.update(at_anchor)
        if not labels:
            if anchors or not connected:
                unassigned.update(component)
                continue
            labels = {next_label}
            next_label += 1
        for label in labels:
            sites[label].update(component)

    return sites, unassigned


def _strip_dummies(core: Chem.Mol) -> tuple[Chem.Mol, dict[int, int]]:
    """Remove attachment atoms; return the query and each label's anchor index."""
    attachments = attachment_atoms(core)
    if any(len(indices) != 1 for indices in attachments.values()):
        raise ParseError("core must have one attachment atom per label")
    anchors = {
        label: core.GetAtomWithIdx(idx).GetNeighbors()[0].GetIdx()
        for label, (idx,) in attachments.items()
    }
    removed = sorted(idx for (idx,) in attachments.values())
    if any(idx in removed for idx in anchors.values()):
        raise ParseError("core attachments must connect to atoms in the core")
    rw = Chem.RWMol(core)
    for idx in reversed(removed):
        rw.RemoveAtom(idx)
    shifted = {label: idx - bisect_left(removed, idx) for label, idx in anchors.items()}
    for idx in shifted.values():
        atom = rw.GetAtomWithIdx(idx)
        atom.SetNoImplicit(False)
        atom.SetNumExplicitHs(0)
    return rw.GetMol(), shifted


def _connected_components(mol: Chem.Mol, atoms: set[int]) -> list[set[int]]:
    """Split atoms into components connected within the set, by lowest index."""
    seen: set[int] = set()
    components: list[set[int]] = []
    for start in sorted(atoms):
        if start in seen:
            continue
        seen.add(start)
        queue = deque([start])
        component: set[int] = set()
        while queue:
            idx = queue.popleft()
            component.add(idx)
            for neighbor in mol.GetAtomWithIdx(idx).GetNeighbors():
                nxt = neighbor.GetIdx()
                if nxt in atoms and nxt not in seen:
                    seen.add(nxt)
                    queue.append(nxt)
        components.append(component)
    return components


def _selection_bonds(
    mol: Chem.Mol, core: set[int], sites: Mapping[int | None, set[int]]
) -> tuple[list[int], dict[int | None, list[int]]]:
    labels_by_atom: dict[int, set[int | None]] = defaultdict(set)
    for label, atoms in sites.items():
        for idx in atoms:
            labels_by_atom[idx].add(label)
    core_bonds = []
    site_bonds: dict[int | None, list[int]] = defaultdict(list)
    for bond in mol.GetBonds():
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        idx = bond.GetIdx()
        if a in core and b in core:
            core_bonds.append(idx)
            continue
        if a in core:
            labels = labels_by_atom[b]
        elif b in core:
            labels = labels_by_atom[a]
        else:
            labels = labels_by_atom[a] & labels_by_atom[b]
        for label in labels:
            site_bonds[label].append(idx)
    return core_bonds, site_bonds
