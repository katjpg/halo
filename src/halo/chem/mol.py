import re
from collections import defaultdict
from collections.abc import Callable
from typing import Literal

from rdkit import Chem

from halo.core.errors import ParseError

Notation = Literal["smiles", "smarts", "molblock", "inchi"]

IDENTITY_POLICY = "rdkit:canonical_smiles"
_INCHIKEY = re.compile(r"[A-Z]{14}-[A-Z]{10}-[A-Z]")


def to_mol(
    value: str | Chem.Mol,
    *,
    notation: Notation = "smiles",
    allow_dummies: bool = False,
) -> Chem.Mol:
    """Parse a structure or copy an RDKit molecule.

    Args:
        value: An RDKit molecule or one structure string without a name.
        notation: Input format. Use SMARTS for substructure queries.
        allow_dummies: Allow attachment atoms and hydrogen-only fragments.

    Returns:
        A molecule with its original atom order and explicit hydrogens.
        Concrete molecules are sanitized.
    """
    if notation not in ("smiles", "smarts", "molblock", "inchi"):
        raise ParseError(f"unsupported notation: {notation!r}")
    query = notation == "smarts"
    if isinstance(value, Chem.Mol):
        mol = Chem.Mol(value)
    else:
        text = value.strip()
        if not text:
            raise ParseError("molecular input is empty")
        if notation != "molblock" and any(ch.isspace() for ch in text):
            raise ParseError("expected one structure without whitespace or text")
        if notation == "molblock" and "$$$$" in value:
            raise ParseError("expected a molfile, not an SDF record")
        if notation == "inchi" and not text.startswith("InChI="):
            raise ParseError("expected an InChI string")
        try:
            mol = _parse(value, text, notation)
        except (ValueError, RuntimeError) as exc:
            raise ParseError(f"cannot parse {notation} input") from exc

    if mol is None or not mol.GetNumAtoms():
        raise ParseError(f"invalid or empty {notation} structure")
    if not query:
        if any(a.HasQuery() for a in mol.GetAtoms()) or any(
            b.HasQuery() for b in mol.GetBonds()
        ):
            raise ParseError("a concrete molecule is required")
        if not allow_dummies and (
            not mol.GetNumHeavyAtoms()
            or any(atom.GetAtomicNum() == 0 for atom in mol.GetAtoms())
        ):
            raise ParseError("expected a heavy-atom molecule without attachment atoms")
        if isinstance(value, Chem.Mol):
            try:
                Chem.SanitizeMol(mol)
            except ValueError as exc:
                raise ParseError("invalid molecular graph") from exc
    return mol


def _parse(value: str, text: str, notation: Notation) -> Chem.Mol | None:
    if notation == "smiles":
        params = Chem.SmilesParserParams()
        params.parseName = False
        params.allowCXSMILES = False
        params.removeHs = False
        return Chem.MolFromSmiles(text, params)
    if notation == "smarts":
        params = Chem.SmartsParserParams()
        params.parseName = False
        params.allowCXSMILES = False
        return Chem.MolFromSmarts(text, params)
    if notation == "molblock":
        # leading blank lines belong to the molfile header, so parse the raw text
        return Chem.MolFromMolBlock(value, removeHs=False, strictParsing=True)
    return Chem.MolFromInchi(text, removeHs=False)


def to_smiles(mol: Chem.Mol, *, keep_atom_maps: bool = False) -> str:
    """Convert a molecule to canonical isomeric SMILES.

    Args:
        mol: A concrete molecule.
        keep_atom_maps: Preserve atom map numbers.

    Returns:
        SMILES with isotopes, stereochemistry, charges, and all components.
        Ordinary hydrogens are removed and atom order can change.

    """
    return canonical_structure(mol, keep_atom_maps=keep_atom_maps)[1]


def canonical_structure(
    mol: Chem.Mol, *, keep_atom_maps: bool = False
) -> tuple[Chem.Mol, str]:
    """Return a normalized copy and its canonical isomeric SMILES."""
    if (
        not mol.GetNumAtoms()
        or any(atom.HasQuery() for atom in mol.GetAtoms())
        or any(bond.HasQuery() for bond in mol.GetBonds())
    ):
        raise ParseError("expected a nonempty concrete molecule")
    copy = Chem.Mol(mol)
    if not keep_atom_maps:
        for atom in copy.GetAtoms():
            atom.SetAtomMapNum(0)
    try:
        copy = Chem.RemoveHs(copy, sanitize=False)
        Chem.SanitizeMol(copy)
        return copy, Chem.MolToSmiles(copy, canonical=True, isomericSmiles=True)
    except (ValueError, RuntimeError) as exc:
        raise ParseError("cannot serialize molecular graph") from exc


def canonicalize_smiles(text: str) -> str:
    """Return HALO's structure key for a valid heavy-atom SMILES.

    Stereo, isotopes, charges, salt components, and tautomers affect
    identity. Atom order, atom maps, and kekulization are normalized.
    """
    return to_smiles(to_mol(text))


def normalize_inchikey(value: str) -> str:
    """Normalize an InChIKey and check its format."""
    key = value.strip().upper()
    if not _INCHIKEY.fullmatch(key):
        raise ParseError("invalid InChIKey")
    return key


def attachment_atoms(mol: Chem.Mol) -> dict[int, tuple[int, ...]]:
    """Map each attachment label to its dummy atom indices.

    A label comes from the atom map or, if unset, the isotope.

    Each attachment needs a positive label and exactly one bond.
    Atom map and isotope labels must agree when both are set.
    """
    labels: dict[int, list[int]] = defaultdict(list)
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 0:
            continue
        label = atom.GetAtomMapNum() or atom.GetIsotope()
        if label < 1 or atom.GetDegree() != 1:
            raise ParseError("attachment atoms need a positive label and one neighbor")
        if atom.GetAtomMapNum() and atom.GetIsotope() not in (0, label):
            raise ParseError("attachment atom has conflicting map and isotope labels")
        labels[label].append(atom.GetIdx())
    return {label: tuple(indices) for label, indices in labels.items()}


def resolve_smiles(identifier: str, *, resolver: Callable[[str], str]) -> str:
    """Resolve a chemical identifier to canonical SMILES."""
    if not identifier.strip():
        raise ParseError("identifier cannot be empty")
    return to_smiles(to_mol(resolver(identifier.strip())))


def edit_molecule(
    molecule: str | Chem.Mol, edit: Callable[[Chem.RWMol], None]
) -> Chem.Mol:
    """Apply an edit to a copy of the molecule and discard its old conformers."""
    rw = Chem.RWMol(to_mol(molecule))
    edit(rw)
    result = rw.GetMol()
    result.RemoveAllConformers()
    result = to_mol(result)
    Chem.AssignStereochemistry(result, cleanIt=True, force=True)
    return result


def embed_conformer(smiles: str, *, seed: int = 42) -> str | None:
    """Return an ETKDG 3D molfile with hydrogens, or None if embedding fails.

    Try MMFF refinement when parameters are available. Use a seed from
    0 to 2**31 - 1.
    """
    from rdkit.Chem import AllChem  # slow to import, and only this function uses it

    if not 0 <= seed < 2**31:
        raise ParseError("seed must be from 0 to 2**31 - 1")
    mol = Chem.AddHs(to_mol(smiles))
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    if AllChem.EmbedMolecule(mol, params) < 0:
        return None
    if AllChem.MMFFHasAllMoleculeParams(mol):
        AllChem.MMFFOptimizeMolecule(mol)
    return Chem.MolToMolBlock(mol)
