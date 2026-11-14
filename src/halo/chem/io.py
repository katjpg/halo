from csv import DictReader
from pathlib import Path

from halo.core.errors import ParseError


def read_smiles(path: str | Path, *, column: str = "smiles") -> list[str]:
    """Read SMILES from a CSV column or SDF in input order, including repeats.

    SDF records return canonical isomeric SMILES.
    """
    path = Path(path)
    if path.suffix.lower() == ".sdf":
        from rdkit import Chem

        from halo.chem.mol import to_mol, to_smiles

        smiles = []
        with path.open("rb") as file:
            for i, mol in enumerate(
                Chem.ForwardSDMolSupplier(file, removeHs=False), start=1
            ):
                if mol is None:
                    raise ParseError(f"Invalid SDF record {i}.")
                smiles.append(to_smiles(to_mol(mol)))
        return smiles
    with path.open(encoding="utf-8-sig", newline="") as file:
        reader = DictReader(file)
        if (reader.fieldnames or []).count(column) != 1:
            raise ParseError(f"CSV needs exactly one {column!r} column.")
        smiles = []
        for row in reader:
            value = (row[column] or "").strip()
            if not value:
                raise ParseError(f"CSV line {reader.line_num} needs a SMILES value.")
            smiles.append(value)
    return smiles
