import itertools
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from rdkit import Chem
from rdkit.Chem import rdChemReactions, rdMMPA, rdRGroupDecomposition

from halo.chem.mol import attachment_atoms, to_mol, to_smiles
from halo.chem.results import Failure
from halo.core.format import format_table
from halo.core.errors import ParseError
from halo.models.generation import RGroupEdit


@dataclass(frozen=True, slots=True)
class RGroupReplacement:
    """The canonical product SMILES and the edit that produced it."""

    product_smiles: str
    edit: RGroupEdit

    def __str__(self) -> str:
        edit = self.edit
        return (
            f"{self.product_smiles}\n"
            f"R{edit.site}  {edit.from_fragment_smiles} → {edit.to_fragment_smiles}"
        )


@dataclass(frozen=True, slots=True)
class Decomposition:
    """A labeled core and its fragments for one input molecule."""

    molecule_index: int
    core_smiles: str
    groups: Mapping[int, str]

    def replace(self, *, site: int, fragment: str) -> RGroupReplacement:
        """Replace a fragment and assign a lone unlabeled attachment to site."""
        product, edit = replace_rgroup(
            self, site=site, fragment_smiles=fragment, label_attachment=True
        )
        return RGroupReplacement(product, edit)


@dataclass(frozen=True)
class RGroupDecomposition:
    """A mapped core and verified R-groups for a molecular series."""

    core_smiles: str
    smiles: tuple[str, ...]
    rows: tuple[Decomposition, ...]
    unmatched: tuple[int, ...]

    def __str__(self) -> str:
        sites = sorted({site for row in self.rows for site in row.groups})
        columns = ["Index", "SMILES", *(f"R{site}" for site in sites)]
        rows = [
            [
                str(row.molecule_index),
                self.smiles[row.molecule_index],
                *(row.groups.get(site, "—") for site in sites),
            ]
            for row in self.rows
        ]
        lines = [f"Core  {self.core_smiles}", format_table(columns, rows)]
        if self.unmatched:
            lines.append(f"Unmatched indices  {', '.join(map(str, self.unmatched))}")
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class RGroupOption:
    """An observed fragment with its member count and example input indices."""

    smiles: str
    count: int
    example_indices: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class RGroupSite:
    """An attachment label and its observed fragments, most frequent first."""

    label: int
    options: tuple[RGroupOption, ...]


@dataclass(frozen=True)
class RGroupLibrary:
    """Observed fragments that rebuild on the same labeled core.

    Sites are in label order. Options include observed hydrogen and list
    bridging fragments under their lowest label. Unmatched lists input
    indices of molecules that could not be decomposed and rebuilt.
    """

    core_smiles: str
    sites: tuple[RGroupSite, ...]
    member_count: int
    unmatched: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class EnumeratedMolecule:
    smiles: str
    groups: Mapping[int, str]


def infer_rgroup_edit(source_smiles: str, product_smiles: str) -> RGroupEdit | Failure:
    """Find a verified single-cut edit, or report why it could not be inferred."""
    source, product = to_mol(source_smiles), to_mol(product_smiles)
    if len(Chem.GetMolFrags(source)) != 1 or len(Chem.GetMolFrags(product)) != 1:
        return Failure("A single-cut edit needs two connected molecules.")

    before, after = _single_cuts(source), _single_cuts(product)
    choices = [
        (
            to_mol(core, allow_dummies=True).GetNumHeavyAtoms(),
            core,
            next(iter(after[core])),
        )
        for core in before.keys() & after.keys()
        if len(before[core]) == len(after[core]) == 1 and before[core] != after[core]
    ]
    if not choices:
        return Failure("No single-cut fragment exchange shares a core.")
    size, core, fragment = max(choices, key=lambda choice: choice[0])
    if sum(choice[0] == size for choice in choices) > 1:
        return Failure("Several single-cut edits share the largest core size.")
    try:
        rebuilt, edit = replace_rgroup(
            decompose_molecule(source_smiles, core), site=1, fragment_smiles=fragment
        )
    except ParseError as exc:
        return Failure(f"Could not rebuild the single-cut edit: {exc}")
    if rebuilt != to_smiles(product):
        return Failure("The inferred edit rebuilt a different product.")
    return edit


def _single_cuts(mol: Chem.Mol) -> dict[str, set[str]]:
    fragments: dict[str, set[str]] = defaultdict(set)
    for _, chains in rdMMPA.FragmentMol(
        mol,
        minCuts=1,
        maxCuts=1,
        maxCutBonds=30,
        pattern="[!#1]!@!=!#[!#1]",
        resultsAsMols=False,
    ):
        core, group = chains.split(".")
        fragments[core].add(group)
        fragments[group].add(core)
    return fragments


def replace_rgroup(
    source: Decomposition,
    *,
    site: int,
    fragment_smiles: str,
    label_attachment: bool = False,
) -> tuple[str, RGroupEdit]:
    """Replace a single mapped fragment and verify the rebuilt structure."""
    if site not in source.groups:
        raise ParseError("The source has no fragment at the requested site.")
    before = to_mol(source.groups[site], allow_dummies=True)
    fragment = to_mol(fragment_smiles, allow_dummies=True)
    if label_attachment:
        dummies = [atom for atom in fragment.GetAtoms() if atom.GetAtomicNum() == 0]
        if len(dummies) == 1 and not (
            dummies[0].GetAtomMapNum() or dummies[0].GetIsotope()
        ):
            dummies[0].SetAtomMapNum(site)
    labels = attachment_atoms(fragment)
    if (
        set(attachment_atoms(before)) != {site}
        or set(labels) != {site}
        or len(labels[site]) != 1
        or len(Chem.GetMolFrags(fragment)) != 1
        or any(
            atom.GetAtomMapNum() for atom in fragment.GetAtoms() if atom.GetAtomicNum()
        )
    ):
        raise ParseError("The edit needs one connected fragment at the requested site.")
    dummy = fragment.GetAtomWithIdx(labels[site][0])
    dummy.SetAtomMapNum(site)
    dummy.SetIsotope(0)
    replacement = to_smiles(fragment, keep_atom_maps=True)
    groups = dict(source.groups) | {site: replacement}
    assembler = RGroupAssembler(source.core_smiles)
    parent, product = assembler.join(source.groups), assembler.join(groups)
    if parent == product:
        raise ParseError("The edit returns the source structure.")
    verified = decompose_molecule(product, source.core_smiles)
    if dict(verified.groups) != groups or verified.core_smiles != source.core_smiles:
        raise ParseError("The edit changed a protected attachment or the core.")
    smarts = edit_reaction_smarts(source, site, replacement, parent, product)
    return product, RGroupEdit(site, source.groups[site], replacement, smarts)


def edit_reaction_smarts(
    source: Decomposition, site: int, replacement: str, parent: str, product: str
) -> str | None:
    """Return a mapped template only when it replays the verified product."""
    retained = dict.fromkeys(
        text for label, text in sorted(source.groups.items()) if label != site
    )
    constant = Chem.Mol()
    atom_map = 0
    for text in (source.core_smiles, *retained):
        mol = to_mol(text, allow_dummies=True)
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 0:
                atom.SetIsotope(atom.GetAtomMapNum() or atom.GetIsotope())
                atom.SetAtomMapNum(0)
            else:
                atom_map += 1
                atom.SetAtomMapNum(atom_map)
        constant = Chem.CombineMols(constant, mol)
    params = Chem.MolzipParams()
    params.label = Chem.MolzipLabel.Isotope
    try:
        sides = []
        for text in (source.groups[site], replacement):
            fragment = to_mol(text, allow_dummies=True)
            for atom in fragment.GetAtoms():
                if atom.GetAtomicNum() == 0:
                    atom.SetIsotope(site)
                atom.SetAtomMapNum(0)
            mapped = Chem.molzip(Chem.CombineMols(constant, fragment), params)
            if mapped is None:
                return None
            sides.append(to_smiles(mapped, keep_atom_maps=True))
        smarts = ">>".join(sides)
        reaction = rdChemReactions.ReactionFromSmarts(smarts)
        if reaction is None:
            return None
        reaction.GetSubstructParams().useChirality = True
        outcomes = reaction.RunReactants((to_mol(parent),), maxProducts=128)
        if not 0 < len(outcomes) < 128 or any(
            len(row) != 1 or to_smiles(row[0]) != product for row in outcomes
        ):
            return None
    except (RuntimeError, ValueError):
        return None
    return smarts


def decompose_molecule(
    smiles: str,
    core: str,
    *,
    core_notation: Literal["smiles", "smarts"] = "smiles",
) -> Decomposition:
    """Split a molecule into a labeled core and R-groups.

    Verify that the fragments rebuild the original stereochemistry.
    """
    rows, _ = _decompose(
        [smiles], to_mol(core, notation=core_notation, allow_dummies=True)
    )
    if not rows:
        raise ParseError("molecule is invalid, does not match, or cannot be rebuilt")
    return rows[0]


def decompose_rgroups(
    smiles: Sequence[str],
    core: str | Chem.Mol,
    *,
    core_notation: Literal["smiles", "smarts"] = "smiles",
) -> RGroupDecomposition:
    """Decompose a series on one core and retain the original input indices.

    Unmatched indices include invalid structures and rows that cannot be
    rebuilt with their stereochemistry. Matching rows must share one mapped core.
    """
    if isinstance(smiles, str) or len(smiles) == 0:
        raise ParseError("Provide a nonempty sequence of SMILES.")
    rows, unmatched = _decompose(
        smiles, to_mol(core, notation=core_notation, allow_dummies=True)
    )
    if not rows:
        raise ParseError("no molecules matched the core")
    labeled = rows[0].core_smiles
    if any(row.core_smiles != labeled for row in rows):
        raise ParseError("R-group decomposition produced incompatible cores")
    return RGroupDecomposition(labeled, tuple(smiles), tuple(rows), unmatched)


def build_rgroup_library(
    member_smiles: Sequence[str],
    core: str,
    *,
    core_notation: Literal["smiles", "smarts"] = "smiles",
    max_sites: int = 10,
    top_k: int = 20,
    max_examples: int = 5,
) -> RGroupLibrary:
    """Collect R-groups by site, ordered by frequency and then SMILES.

    Keep the lowest site labels up to max_sites. Enumeration caps
    omitted sites with hydrogen. Limits must be positive and accepted
    members must share a labeled core.
    """
    if not member_smiles or min(max_sites, top_k, max_examples) < 1:
        raise ParseError("members and positive limits are required")
    decomposition = decompose_rgroups(member_smiles, core, core_notation=core_notation)

    counts: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    examples: dict[tuple[int, str], list[int]] = defaultdict(list)
    for row in decomposition.rows:
        for label, smiles in row.groups.items():
            counts[label][smiles] += 1
            if len(examples[label, smiles]) < max_examples:
                examples[label, smiles].append(row.molecule_index)
    sites = []
    for label in sorted(counts)[:max_sites]:
        options = (
            RGroupOption(smiles, count, tuple(examples[label, smiles]))
            for smiles, count in counts[label].items()
        )
        top = sorted(options, key=lambda option: (-option.count, option.smiles))[:top_k]
        sites.append(RGroupSite(label, tuple(top)))
    return RGroupLibrary(
        decomposition.core_smiles,
        tuple(sites),
        len(decomposition.rows),
        decomposition.unmatched,
    )


def _decompose(
    member_smiles: Sequence[str], core: Chem.Mol
) -> tuple[list[Decomposition], tuple[int, ...]]:
    """Return reconstructable decompositions and rejected input indices."""
    params = rdRGroupDecomposition.RGroupDecompositionParameters()
    params.removeAllHydrogenRGroups = False
    params.removeAllHydrogenRGroupsAndLabels = False
    params.onlyMatchAtRGroups = any(
        atom.GetAtomicNum() == 0 and (atom.GetAtomMapNum() or atom.GetIsotope())
        for atom in core.GetAtoms()
    )
    try:
        rgd = rdRGroupDecomposition.RGroupDecomposition([core], params)
    except ValueError as exc:
        raise ParseError(f"invalid R-group core: {exc}") from exc
    accepted: dict[int, tuple[int, str]] = {}
    rejected: set[int] = set()
    for i, smiles in enumerate(member_smiles):
        try:
            mol = to_mol(smiles)
        except ParseError:
            rejected.add(i)
            continue
        row_idx = rgd.Add(mol)
        if row_idx < 0:
            rejected.add(i)
        else:
            accepted[row_idx] = i, to_smiles(mol)
    if not accepted:
        return [], tuple(sorted(rejected))
    if not rgd.Process():
        raise RuntimeError("R-group decomposition did not complete")
    rows = rgd.GetRGroupsAsRows(asSmiles=False)
    if len(rows) != len(accepted):
        raise RuntimeError("R-group rows do not match accepted molecule indices")

    result = []
    assemblers: dict[str, RGroupAssembler] = {}
    for row_idx, row in enumerate(rows):
        groups: dict[int, str] = {}
        owners: dict[int, _Fragment] = {}
        for key, fragment in sorted(row.items()):
            if not (key.startswith("R") and key[1:].isdigit()):
                continue
            labels = attachment_atoms(fragment)
            if not labels:
                raise RuntimeError("decomposed fragment has no labeled attachments")
            text = to_smiles(fragment, keep_atom_maps=True)
            claim = _claim_fragment(owners, _Fragment(fragment, labels, text))
            if claim == "conflict":
                raise RuntimeError("decomposed fragments claim the same attachment")
            if claim == "added":
                groups[min(labels)] = text
        labeled = to_smiles(row["Core"], keep_atom_maps=True)
        member_idx, member = accepted[row_idx]
        if labeled not in assemblers:
            assemblers[labeled] = RGroupAssembler(labeled)
        try:
            rebuilt = assemblers[labeled].join(groups)
        except ParseError:
            rejected.add(member_idx)
            continue
        if rebuilt != member:
            rejected.add(member_idx)  # lost structure or stereochemistry
            continue
        result.append(Decomposition(member_idx, labeled, groups))
    return result, tuple(sorted(rejected))


@dataclass(frozen=True, slots=True)
class _Fragment:
    mol: Chem.Mol
    labels: Mapping[int, tuple[int, ...]]
    smiles: str


def _claim_fragment(
    owners: dict[int, _Fragment], fragment: _Fragment
) -> Literal["added", "repeat", "conflict"]:
    """Track which fragment fills each label. A bridge can fill several."""
    shared = owners.keys() & fragment.labels.keys()
    if shared:
        if any(owners[label].smiles != fragment.smiles for label in shared):
            return "conflict"
        return "repeat"
    owners.update(dict.fromkeys(fragment.labels, fragment))
    return "added"


class RGroupAssembler:
    """A labeled core for joining and enumerating R-groups."""

    def __init__(self, core_smiles: str) -> None:
        self._core = to_mol(core_smiles, allow_dummies=True)
        if len(Chem.GetMolFrags(self._core)) != 1:
            raise ParseError("core must be connected")
        self._labels = attachment_atoms(self._core)
        if any(len(indices) != 1 for indices in self._labels.values()):
            raise ParseError("core must have one dummy per attachment label")

    def join(self, groups: Mapping[int, str]) -> str:
        """Join fragments keyed by attachment label and return canonical SMILES.

        Cap unused sites with hydrogen. A bridging fragment can appear
        under any or all of its labels. Conflicting labels raise ParseError.
        """
        return self._join(
            {label: self._parse_fragment(s) for label, s in groups.items()}
        )

    def enumerate(
        self,
        sites: Sequence[RGroupSite],
        *,
        max_molecules: int = 1000,
        max_attempts: int | None = None,
        constraints: Mapping[int, str] | None = None,
    ) -> list[EnumeratedMolecule]:
        """Return distinct valid joins in site and option order.

        Constraints fix a fragment at a site. Invalid joins count toward
        max_attempts, which defaults to ten times max_molecules.
        """
        if max_molecules < 1:
            raise ParseError("max_molecules must be positive")
        limit = max_attempts if max_attempts is not None else 10 * max_molecules
        if limit < 1:
            raise ParseError("max_attempts must be positive")
        fixed = dict(constraints or {})
        labels = [site.label for site in sites]
        if len(set(labels)) != len(labels):
            raise ParseError("site labels must be unique")
        if not set(labels) <= self._labels.keys() or not fixed.keys() <= set(labels):
            raise ParseError("sites or constraints reference unknown attachment labels")

        parsed: dict[str, _Fragment] = {}
        choices: list[list[tuple[int, _Fragment]]] = []
        for site in sites:
            texts = (
                [fixed[site.label]]
                if site.label in fixed
                else [option.smiles for option in site.options]
            )
            options: dict[str, _Fragment] = {}
            for text in texts:
                if text not in parsed:
                    parsed[text] = self._parse_fragment(text)
                fragment = parsed[text]
                if site.label not in fragment.labels:
                    raise ParseError(f"fragment does not contain site {site.label}")
                options[fragment.smiles] = fragment
            choices.append([(site.label, fragment) for fragment in options.values()])

        seen: set[str] = set()
        result: list[EnumeratedMolecule] = []
        for combo in itertools.islice(itertools.product(*choices), limit):
            selected = _select_compatible(combo, fixed)
            if selected is None:
                continue
            try:
                smiles = self._join(selected)
            except ParseError:
                continue
            if smiles in seen:
                continue
            seen.add(smiles)
            groups = {label: fragment.smiles for label, fragment in selected.items()}
            result.append(EnumeratedMolecule(smiles, groups))
            if len(result) == max_molecules:
                break
        return result

    def _parse_fragment(self, smiles: str) -> _Fragment:
        mol = to_mol(smiles, allow_dummies=True)
        labels = attachment_atoms(mol)
        if not labels or not labels.keys() <= self._labels.keys():
            raise ParseError("fragment has missing or unknown attachment labels")
        dummy_ids = {idx for indices in labels.values() for idx in indices}
        if any(not dummy_ids.intersection(part) for part in Chem.GetMolFrags(mol)):
            raise ParseError("every fragment component must contain an attachment")
        for label, indices in labels.items():
            for idx in indices:
                atom = mol.GetAtomWithIdx(idx)
                atom.SetAtomMapNum(label)
                atom.SetIsotope(0)
        return _Fragment(mol, labels, to_smiles(mol, keep_atom_maps=True))

    def _join(self, groups: Mapping[int, _Fragment]) -> str:
        owners: dict[int, _Fragment] = {}
        fragments: list[_Fragment] = []
        for site, fragment in sorted(groups.items()):
            if site not in fragment.labels:
                raise ParseError(f"fragment at site {site} does not contain that label")
            claim = _claim_fragment(owners, fragment)
            if claim == "conflict":
                raise ParseError("fragments claim the same attachment site")
            if claim == "added":
                fragments.append(fragment)

        combined, tags = self._tag_core(owners)
        for fragment in fragments:
            combined = Chem.CombineMols(combined, _tag_fragment(fragment, tags))
        for label in self._labels.keys() - owners.keys():
            cap = Chem.MolFromSmiles(f"[{tags[label][0]}*][H]", sanitize=False)
            combined = Chem.CombineMols(combined, cap)

        params = Chem.MolzipParams()
        params.label = Chem.MolzipLabel.Isotope
        try:
            joined = Chem.molzip(combined, params)
        except (RuntimeError, ValueError) as exc:
            raise ParseError("cannot join the supplied R-groups") from exc
        if joined is None or any(
            atom.GetAtomicNum() == 0 for atom in joined.GetAtoms()
        ):
            raise ParseError("unmatched attachment atoms")
        if len(Chem.GetMolFrags(joined)) != 1:
            raise ParseError("joined molecule is disconnected")
        return to_smiles(joined)

    def _tag_core(
        self, owners: Mapping[int, _Fragment]
    ) -> tuple[Chem.Mol, dict[int, tuple[int, ...]]]:
        # Pair dummies by isotope tag. Hydrogen caps sites without a fragment.
        core = Chem.RWMol(self._core)
        tags: dict[int, tuple[int, ...]] = {}
        next_tag = 1
        for label, (idx,) in sorted(self._labels.items()):
            dummy = core.GetAtomWithIdx(idx)
            anchor = dummy.GetNeighbors()[0].GetIdx()
            bond = core.GetBondBetweenAtoms(idx, anchor).GetBondType()
            count = len(owners[label].labels[label]) if label in owners else 1
            tags[label] = tuple(range(next_tag, next_tag + count))
            next_tag += count
            dummy.SetIsotope(tags[label][0])
            dummy.SetAtomMapNum(0)
            for tag in tags[label][1:]:
                added = core.AddAtom(Chem.Atom(0))
                core.GetAtomWithIdx(added).SetIsotope(tag)
                core.AddBond(anchor, added, bond)
        return core.GetMol(), tags


def _tag_fragment(fragment: _Fragment, tags: Mapping[int, tuple[int, ...]]) -> Chem.Mol:
    mol = Chem.Mol(fragment.mol)
    for label, indices in fragment.labels.items():
        for idx, tag in zip(indices, tags[label], strict=True):
            atom = mol.GetAtomWithIdx(idx)
            atom.SetAtomMapNum(0)
            atom.SetIsotope(tag)
    return mol


def _select_compatible(
    combo: Sequence[tuple[int, _Fragment]], fixed: Mapping[int, str]
) -> dict[int, _Fragment] | None:
    """Keep one fragment per site, or None when two fragments claim one site."""
    selected: dict[int, _Fragment] = {}
    owners: dict[int, _Fragment] = {}
    for label, fragment in combo:
        if label in owners:  # already filled by a bridging fragment
            if label in fixed and owners[label].smiles != fragment.smiles:
                return None
            continue
        if _claim_fragment(owners, fragment) != "added":
            return None
        selected[label] = fragment
    return selected


def join_rgroups(core_smiles: str, groups: Mapping[int, str]) -> str:
    """Join a labeled core with fragments, capping unused sites with hydrogen."""
    return RGroupAssembler(core_smiles).join(groups)


def enumerate_rgroups(
    library: RGroupLibrary,
    *,
    max_molecules: int = 1000,
    max_attempts: int | None = None,
    constraints: Mapping[int, str] | None = None,
) -> list[EnumeratedMolecule]:
    """Enumerate distinct valid joins in the library's site and option order."""
    return RGroupAssembler(library.core_smiles).enumerate(
        library.sites,
        max_molecules=max_molecules,
        max_attempts=max_attempts,
        constraints=constraints,
    )
