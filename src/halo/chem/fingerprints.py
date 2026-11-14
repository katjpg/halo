from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Self

import numpy as np
from numpy.typing import NDArray
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

from halo.chem.mol import to_mol
from halo.core.errors import ParseError

MORGAN_RADIUS = 2
MORGAN_BITS = 2048
MORGAN_CHIRALITY = False


def prepare_mols(mols: Sequence[Chem.Mol]) -> list[Chem.Mol]:
    """Return sanitized copies without atom maps or ordinary hydrogens.

    Preserve salts, charges, protonation states, and tautomers.
    """
    prepared = []
    for i, mol in enumerate(mols):
        try:
            copy = to_mol(mol)
        except ParseError as exc:
            raise ParseError(f"molecule {i}: {exc}") from exc
        for atom in copy.GetAtoms():
            atom.SetAtomMapNum(0)
        prepared.append(Chem.RemoveHs(copy))
    return prepared


@dataclass(frozen=True)
class Fingerprints:
    """Equal-length binary fingerprints in input order, with their method."""

    bits: tuple[DataStructs.ExplicitBitVect, ...]
    method: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "bits", tuple(self.bits))
        for i, fp in enumerate(self.bits):
            if not isinstance(fp, DataStructs.ExplicitBitVect):
                raise ParseError(f"fingerprint {i} is not an ExplicitBitVect")
        widths = {fp.GetNumBits() for fp in self.bits}
        if 0 in widths or len(widths) > 1:
            raise ParseError("fingerprints must share one nonzero length")

    @classmethod
    def from_mols(
        cls,
        mols: Sequence[Chem.Mol],
        *,
        radius: int = MORGAN_RADIUS,
        n_bits: int = MORGAN_BITS,
        include_chirality: bool = MORGAN_CHIRALITY,
        prepared: bool = False,
    ) -> Self:
        """Compute Morgan fingerprints for each molecule.

        Set prepared=True only for sanitized molecules without ordinary
        hydrogens or atom maps.
        """
        if not 0 <= radius < 2**32 or not 0 < n_bits < 2**32:
            raise ParseError(
                "Morgan fingerprints need a nonnegative radius and a positive size"
            )
        generator = rdFingerprintGenerator.GetMorganGenerator(
            radius=radius, fpSize=n_bits, includeChirality=include_chirality
        )
        return cls(
            generator.GetFingerprints(mols if prepared else prepare_mols(mols)),
            f"rdkit_morgan(radius={radius}, n_bits={n_bits}, "
            f"chirality={include_chirality})",
        )

    def __len__(self) -> int:
        return len(self.bits)

    def __iter__(self) -> Iterator[DataStructs.ExplicitBitVect]:
        return iter(self.bits)

    def __getitem__(
        self, index: int | slice
    ) -> DataStructs.ExplicitBitVect | tuple[DataStructs.ExplicitBitVect, ...]:
        return self.bits[index]

    def condensed_distances(self) -> NDArray[np.float64]:
        """Return the lower triangle of Tanimoto distances in Butina order."""
        n = len(self)
        distances = np.empty(n * (n - 1) // 2)
        for i in range(1, n):
            start = i * (i - 1) // 2  # Butina indexes distance (i, j) at start + j
            distances[start : start + i] = DataStructs.BulkTanimotoSimilarity(
                self.bits[i], self.bits[:i], returnDistance=True
            )
        return distances

    def medoid(self, indices: Sequence[int]) -> int:
        """Choose the member with the smallest total distance, then lowest input index."""
        if not len(indices) or any(not 0 <= i < len(self) for i in indices):
            raise ParseError("a medoid needs at least one valid fingerprint index")
        members = tuple(self.bits[i] for i in indices)
        return min(
            indices,
            key=lambda i: (
                sum(
                    DataStructs.BulkTanimotoSimilarity(
                        self.bits[i], members, returnDistance=True
                    )
                ),
                i,
            ),
        )

    def distances(self) -> NDArray[np.float64]:
        """Compute pairwise Tanimoto distances, 1 - similarity, as an (n, n) matrix.

        Allocates 8 * n * n bytes.
        """
        n = len(self.bits)
        dist = np.zeros((n, n))
        for i in range(1, n):
            row = DataStructs.BulkTanimotoSimilarity(
                self.bits[i], self.bits[:i], returnDistance=True
            )
            dist[i, :i] = row
            dist[:i, i] = row
        return dist
