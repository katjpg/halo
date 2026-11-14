from dataclasses import dataclass

from rdkit import Chem
from rdkit.Chem import BRICS

from halo.chem.mol import canonical_structure, to_mol, to_smiles
from halo.core.errors import ParseError
from halo.core.format import format_table


@dataclass(frozen=True)
class Fragmentation:
    """Distinct fragment pairs from single BRICS cuts of the source."""

    source_smiles: str
    cuts: tuple[tuple[str, str], ...]

    @property
    def fragments(self) -> tuple[str, ...]:
        return tuple(sorted({fragment for pair in self.cuts for fragment in pair}))

    def __str__(self) -> str:
        header = f"Single BRICS cuts of {self.source_smiles}"
        if not self.cuts:
            return f"{header}\nNo BRICS bonds found."
        return header + "\n" + format_table(("Fragment", "Partner"), self.cuts)


def fragment_molecule(value: str | Chem.Mol) -> Fragmentation:
    """Cut each BRICS bond once and retain the two labeled fragments.

    Collapse symmetric cuts that give the same pair. A source with no
    eligible bonds returns an empty result.
    """
    mol, source = canonical_structure(to_mol(value))
    if len(Chem.GetMolFrags(mol)) != 1:
        raise ParseError("Fragmentation needs one connected molecule.")
    cuts = set()
    for bond in BRICS.FindBRICSBonds(mol):
        parts = Chem.GetMolFrags(BRICS.BreakBRICSBonds(mol, [bond]), asMols=True)
        first, second = sorted(to_smiles(fragment) for fragment in parts)
        cuts.add((first, second))
    return Fragmentation(source, tuple(sorted(cuts)))
