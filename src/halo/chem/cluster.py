from collections.abc import Sequence
from dataclasses import dataclass

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.ML.Cluster import Butina
from rdkit.SimDivFilters import rdSimDivPickers

from halo.chem.fingerprints import Fingerprints, prepare_mols
from halo.core.format import format_table
from halo.core.errors import ParseError


@dataclass(frozen=True, slots=True)
class Cluster:
    """Input indices in one cluster and the index of its Butina seed."""

    indices: tuple[int, ...]
    representative: int


@dataclass(frozen=True, slots=True)
class ClusterResult:
    """Butina clusters in input order and their Tanimoto distance cutoff."""

    clusters: tuple[Cluster, ...]
    fingerprint_method: str
    cutoff: float

    def __str__(self) -> str:
        rows = [
            [str(i), str(cluster.representative), ", ".join(map(str, cluster.indices))]
            for i, cluster in enumerate(self.clusters, start=1)
        ]
        return (
            f"Butina clusters (minimum Tanimoto similarity {1 - self.cutoff:.3g})\n"
            + format_table(("Cluster", "Seed index", "Member indices"), rows)
        )

    @property
    def n_clusters(self) -> int:
        return len(self.clusters)


@dataclass(frozen=True, slots=True)
class ScaffoldGroup:
    """Input indices sharing one Murcko scaffold; smiles is None when acyclic."""

    smiles: str | None
    indices: tuple[int, ...]


def cluster_butina(fps: Fingerprints, *, cutoff: float) -> ClusterResult:
    """Group molecules within an inclusive Tanimoto distance of a Butina seed.

    Choose the seed with the most unassigned neighbors. Ties favor later
    inputs. Keep singletons. The cutoff is between 0 and 1.
    """
    if not 0 <= cutoff <= 1:
        raise ParseError("cutoff must be from 0 to 1")
    groups = Butina.ClusterData(
        fps.condensed_distances(), len(fps), cutoff, isDistData=True, reordering=True
    )
    clusters = [Cluster(tuple(sorted(group)), group[0]) for group in groups]
    return ClusterResult(
        tuple(sorted(clusters, key=lambda cluster: cluster.indices[0])),
        fps.method,
        cutoff,
    )


def pick_diverse(fps: Fingerprints, n_pick: int, *, seed: int = 0) -> tuple[int, ...]:
    """Select input indices by MaxMin Tanimoto diversity, in picking order.

    Use a nonnegative seed and a count from zero to the input size.
    """
    if not 0 <= n_pick <= len(fps):
        raise ParseError("n_pick must be from 0 to the number of fingerprints")
    if seed < 0:  # RDKit treats a negative seed as "pick at random"
        raise ParseError("seed must be nonnegative")
    if n_pick == 0:  # RDKit returns one pick when asked for none
        return ()
    picker = rdSimDivPickers.MaxMinPicker()
    return tuple(picker.LazyBitVectorPick(fps.bits, len(fps), n_pick, seed=seed))


def group_scaffolds(
    mols: Sequence[Chem.Mol], *, include_chirality: bool = False
) -> tuple[ScaffoldGroup, ...]:
    """Group by Murcko scaffold SMILES, in order of first appearance.

    Give each acyclic molecule its own group. Drop acyclic fragments and
    keep charges on ring atoms.
    """
    scaffolded: dict[str, list[int]] = {}
    acyclic = []
    for i, mol in enumerate(prepare_mols(mols)):
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(
            mol=mol, includeChirality=include_chirality
        )
        if scaffold:
            scaffolded.setdefault(scaffold, []).append(i)
        else:
            acyclic.append(ScaffoldGroup(None, (i,)))
    groups = [
        ScaffoldGroup(smiles, tuple(indices)) for smiles, indices in scaffolded.items()
    ]
    return tuple(sorted(groups + acyclic, key=lambda group: group.indices[0]))
