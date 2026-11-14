from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

from rdkit import Chem

from halo.chem.mol import canonicalize_smiles, to_mol, to_smiles
from halo.core.errors import ParseError

DEFAULT_GROUPS = ("rings_in_drugs", "basic_groups", "extended_groups")
DEFAULT_CATALOGS = ("pains", "brenk", "nibr", "reactive_unstable_toxic")


@dataclass(frozen=True, slots=True)
class GroupMatch:
    """A medchem group with matches in the annotation's atom order."""

    name: str
    group: str
    hierarchies: tuple[str, ...]
    smarts: str
    atom_matches: tuple[tuple[int, ...], ...]

    @property
    def match_count(self) -> int:
        return len(self.atom_matches)


@dataclass(frozen=True, slots=True)
class StructuralAlert:
    """An alert with the atom matches reported by its catalog."""

    catalog: str
    description: str
    properties: tuple[tuple[str, str], ...]
    atom_matches: tuple[tuple[int, ...], ...] = ()


@dataclass(frozen=True, slots=True)
class StructuralAnnotation:
    """The groups and alerts found in one molecule.

    Match indices refer to molblock if provided, otherwise to the order
    in the canonical SMILES. Overlapping matches stay separate.
    """

    smiles: str
    molblock: str | None
    groups: tuple[GroupMatch, ...]
    alerts: tuple[StructuralAlert, ...]


class StructuralAnnotator:
    """Compiled medicinal chemistry groups and alerts shared across batches."""

    def __init__(
        self,
        *,
        groups: Sequence[str] = DEFAULT_GROUPS,
        catalogs: Sequence[str] = DEFAULT_CATALOGS,
    ) -> None:
        from medchem.catalogs import NamedCatalogs, list_named_catalogs
        from medchem.groups import ChemicalGroup, list_default_chemical_groups

        group_names = tuple(dict.fromkeys(groups))
        catalog_names = tuple(dict.fromkeys(catalogs))
        if not group_names:
            raise ParseError("select at least one chemical group collection")
        unknown = set(group_names) - set(list_default_chemical_groups())
        if unknown:
            raise ParseError(f"unknown chemical group collections: {sorted(unknown)}")
        unknown = set(catalog_names) - set(list_named_catalogs())
        if unknown:
            raise ParseError(f"unknown alert catalogs: {sorted(unknown)}")
        # Each match handles one molecule, so avoid starting a worker pool.
        self._groups = ChemicalGroup(groups=list(group_names), n_jobs=1)
        self._catalogs = {
            name: getattr(NamedCatalogs, name)() for name in catalog_names
        }
        paths: dict[tuple[str, str, str], set[str]] = defaultdict(set)
        for row in self._groups.dataframe.to_dict(orient="records"):
            hierarchy = row.get("hierarchy")
            if isinstance(hierarchy, str) and hierarchy:
                paths[(row["group"], row["name"], row["smarts"])].add(hierarchy)
        self._hierarchies = {
            key: tuple(sorted(values)) for key, values in paths.items()
        }

    def annotate(self, value: str | Chem.Mol) -> StructuralAnnotation:
        """Find groups and alerts, with a molblock in the input's atom order.

        Groups sort by collection and name.
        """
        mol = to_mol(value)
        return self._annotate(mol, to_smiles(mol), include_molblock=True)

    def _annotate(
        self, mol: Chem.Mol, smiles: str, *, include_molblock: bool
    ) -> StructuralAnnotation:
        hits = self._groups.get_matches(mol, use_smiles=False)
        if hits is None:
            raise RuntimeError("medchem returned no match table for a valid molecule")
        if not {"name", "group", "smarts", "matches"}.issubset(hits.columns):
            raise RuntimeError("the medchem match table is missing required columns")
        matches: dict[tuple[str, str, str], set[tuple[int, ...]]] = defaultdict(set)
        n_atoms = mol.GetNumAtoms()
        for row in hits.to_dict(orient="records"):
            key = (row["group"], row["name"], row["smarts"])
            for match in row["matches"]:
                indices = tuple(int(index) for index in match)
                if any(not 0 <= index < n_atoms for index in indices):
                    raise RuntimeError(
                        "medchem returned atom indices outside the molecule"
                    )
                matches[key].add(indices)
        groups = tuple(
            GroupMatch(
                name=name,
                group=group,
                hierarchies=self._hierarchies.get((group, name, smarts), ()),
                smarts=smarts,
                atom_matches=tuple(sorted(indices)),
            )
            for (group, name, smarts), indices in sorted(matches.items())
        )
        alerts = tuple(
            StructuralAlert(
                catalog=name,
                description=entry.GetDescription(),
                properties=tuple(
                    sorted((key, entry.GetProp(key)) for key in entry.GetPropList())
                ),
                atom_matches=tuple(
                    tuple(pair.target for pair in match.atomPairs)
                    for match in entry.GetFilterMatches(mol)
                    if match.atomPairs
                ),
            )
            for name, catalog in self._catalogs.items()
            for entry in catalog.GetMatches(mol)
        )
        return StructuralAnnotation(
            smiles=smiles,
            molblock=Chem.MolToMolBlock(mol) if include_molblock else None,
            groups=groups,
            alerts=alerts,
        )

    def annotate_many(
        self, values: Sequence[str | Chem.Mol], *, include_molblock: bool = False
    ) -> list[StructuralAnnotation]:
        """Annotate each canonical structure once and return results in input order.

        Use canonical SMILES atom order for matches and optional molblocks.
        """
        if isinstance(values, str):
            raise ParseError("Provide a sequence of molecules, such as ['CCO'].")
        keys: dict[str, str] = {}
        done: dict[str, StructuralAnnotation] = {}
        out: list[StructuralAnnotation] = []
        for value in values:
            if isinstance(value, str):
                if value not in keys:
                    keys[value] = canonicalize_smiles(value)
                smiles = keys[value]
            else:
                smiles = to_smiles(to_mol(value))
            if smiles not in done:
                done[smiles] = self._annotate(
                    to_mol(smiles), smiles, include_molblock=include_molblock
                )
            out.append(done[smiles])
        return out
