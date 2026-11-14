from collections.abc import Mapping
from typing import Literal

from rdkit import Chem
from rdkit.Chem import rdCoordGen, rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D

from halo.chem.mol import Notation, to_mol
from halo.chem.sar.sites import match_rgroup_atoms
from halo.core.errors import ParseError

Color = tuple[float, float, float, float]

_SITE_COLORS: tuple[Color, ...] = (
    (0.30, 0.52, 0.90, 0.35),
    (0.30, 0.75, 0.45, 0.35),
    (0.95, 0.60, 0.20, 0.35),
    (0.65, 0.40, 0.85, 0.35),
    (0.90, 0.35, 0.35, 0.35),
    (0.20, 0.75, 0.75, 0.35),
)
_CORE_COLOR: Color = (0.6, 0.6, 0.6, 0.15)


def draw_png(
    mol: Chem.Mol,
    *,
    width: int = 300,
    height: int = 300,
    atom_colors: Mapping[int, Color] | None = None,
    bond_colors: Mapping[int, Color] | None = None,
) -> bytes:
    """Draw a molecule or query as PNG from a copy.

    Reuse any 2D coordinates. Dimensions are positive pixel counts.
    Highlight colors are RGBA channels from 0 to 1, keyed by atom or bond index.
    """
    if width <= 0 or height <= 0:
        raise ParseError("image dimensions must be positive")
    mol = Chem.Mol(mol)
    if not mol.GetNumConformers() or mol.GetConformer().Is3D():
        try:
            rdDepictor.Compute2DCoords(
                mol, useRingTemplates=True, canonOrient=True, forceRDKit=True
            )
        except (ValueError, RuntimeError):
            rdCoordGen.AddCoords(mol)
    atoms = dict(atom_colors or {})
    bonds = dict(bond_colors or {})
    drawer = rdMolDraw2D.MolDraw2DCairo(width, height)
    drawer.drawOptions().padding = 0.02
    rdMolDraw2D.PrepareAndDrawMolecule(
        drawer,
        mol,
        highlightAtoms=list(atoms),
        highlightAtomColors=atoms,
        highlightBonds=list(bonds),
        highlightBondColors=bonds,
    )
    drawer.FinishDrawing()
    return bytes(drawer.GetDrawingText())


def draw_molecule(
    value: str | Chem.Mol,
    *,
    notation: Notation = "smiles",
    width: int = 300,
    height: int = 300,
) -> bytes:
    """Draw a structure or SMARTS query as PNG."""
    mol = to_mol(value, notation=notation, allow_dummies=True)
    return draw_png(mol, width=width, height=height)


def draw_rgroups(
    molecule: str | Chem.Mol,
    core: str,
    *,
    core_notation: Literal["smiles", "smarts"] = "smiles",
    width: int = 300,
    height: int = 300,
) -> bytes:
    """Draw the selected core in gray and each matched site in color."""
    mol = to_mol(molecule)
    match = match_rgroup_atoms(mol, core, core_notation=core_notation)
    atoms = dict.fromkeys(match.core.atoms, _CORE_COLOR)
    bonds = dict.fromkeys(match.core.bonds, _CORE_COLOR)
    for label, selection in match.sites.items():
        color = _SITE_COLORS[(label - 1) % len(_SITE_COLORS)]
        atoms.update(dict.fromkeys(selection.atoms, color))
        bonds.update(dict.fromkeys(selection.bonds, color))
    return draw_png(
        mol, width=width, height=height, atom_colors=atoms, bond_colors=bonds
    )
