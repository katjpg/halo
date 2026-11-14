from collections.abc import Mapping
from dataclasses import dataclass

from halo.chem.cluster import cluster_butina
from halo.chem.eval.pareto import prediction_front
from halo.chem.fingerprints import Fingerprints
from halo.chem.medchem import StructuralAnnotation, StructuralAnnotator
from halo.chem.mol import to_mol
from halo.chem.operations import CommonCore, decompose_rgroups, find_mcs
from halo.chem.results import Failure, Metric
from halo.chem.sar.rgroup import RGroupDecomposition
from halo.core.errors import ParseError
from halo.core.format import format_table
from halo.refs import molecule_label
from halo.schemas.generation import GenerationRunResponse
from halo.schemas.molecule import ComputedMetric, MetricValue
from halo.types import MoleculeId


@dataclass(frozen=True)
class ClusterReview:
    """One cluster's medoid, common core, and verified decomposition."""

    member_ids: tuple[MoleculeId, ...]
    representative_id: MoleculeId
    core: CommonCore | None
    rgroups: RGroupDecomposition | None
    note: str | None


@dataclass(frozen=True)
class RunReview:
    """Saved predictions and local structural analysis for one generation run."""

    run: GenerationRunResponse
    source_smiles: str
    clusters: tuple[ClusterReview, ...]
    annotations: Mapping[MoleculeId, StructuralAnnotation]
    pareto_ids: tuple[MoleculeId, ...]
    min_similarity: float
    min_core_coverage: float

    def tables(self) -> dict[str, list[dict[str, str]]]:
        """Return rows for the candidate, cluster, edit, and annotation tables."""
        source_id = MoleculeId(self.run.source_molecule_id)
        criteria = self.run.criteria or {}
        fields = criteria or self.run.objectives
        predictions = self.run.source_predictions or {}
        candidates, edits = [], []
        for candidate in self.run.candidates:
            mid = MoleculeId(candidate.molecule_id)
            changes = candidate.prediction_changes or {}
            comparable = (
                bool(criteria)
                and mid != source_id
                and all(changes.get(name) is not None for name in criteria)
            )
            pareto = "—"
            if comparable:
                pareto = "yes" if mid in self.pareto_ids else "no"
            row = {
                "Molecule": molecule_label(mid),
                "SMILES": candidate.smiles,
                "Pareto": pareto,
            }
            for name in fields:
                label = f"Δ {name}"
                if name in predictions:
                    label += f" ({predictions[name].unit})"
                delta = changes.get(name)
                row[label] = f"{delta:+.3g}" if delta is not None else "—"
            for name in fields:
                row[f"{name} goal"] = candidate.goal_outcomes.get(name, "—")
            candidates.append(row)
            edit = candidate.transformation
            edit_label = "Unavailable"
            reaction_smarts = "—"
            if edit is not None:
                edit_label = (
                    "Mapped replacement"
                    if self.run.core_smiles
                    else "Inferred single cut"
                )
                reaction_smarts = edit.reaction_smarts or "—"
            edits.append(
                {
                    "Molecule": molecule_label(mid),
                    "Edit": edit_label,
                    "Site": f"R{edit.site}" if edit else "—",
                    "From": edit.from_fragment_smiles if edit else "—",
                    "To": edit.to_fragment_smiles if edit else "—",
                    "Reaction SMARTS": reaction_smarts,
                }
            )
        clusters, rgroups = [], []
        for i, cluster in enumerate(self.clusters, start=1):
            core, groups = cluster.core, cluster.rgroups
            matched = (
                tuple(cluster.member_ids[j] for j in core.matched_indices)
                if core
                else ()
            )
            clusters.append(
                {
                    "Cluster": str(i),
                    "Medoid": molecule_label(cluster.representative_id),
                    "Members": ", ".join(map(molecule_label, cluster.member_ids)),
                    "Core matches": f"{len(matched)}/{len(cluster.member_ids)}"
                    if core
                    else "—",
                    "Core": core.smarts if core else "—",
                    "Mapped core": groups.core_smiles if groups else "—",
                    "R-group rows": f"{len(groups.rows)}/{len(matched)}"
                    if groups
                    else "—",
                    "Note": cluster.note or "",
                }
            )
            if groups is not None:
                rgroups.extend(
                    {
                        "Cluster": str(i),
                        "Molecule": molecule_label(matched[row.molecule_index]),
                        "Site": f"R{site}",
                        "Fragment": fragment,
                    }
                    for row in groups.rows
                    for site, fragment in sorted(row.groups.items())
                )
        source_alerts = alert_names(self.annotations[source_id])
        annotations = []
        for mid, annotation in self.annotations.items():
            alerts = alert_names(annotation)
            annotations.append(
                {
                    "Molecule": molecule_label(mid),
                    "Groups": ", ".join(
                        sorted({group.name for group in annotation.groups})
                    )
                    or "—",
                    "Alerts": "; ".join(sorted(alerts)) or "—",
                    "Added alerts": "; ".join(sorted(alerts - source_alerts)) or "—",
                    "Removed alerts": "; ".join(sorted(source_alerts - alerts)) or "—",
                }
            )
        return {
            "Candidates": candidates,
            "Clusters": clusters,
            "R-groups": rgroups,
            "Transformations": edits,
            "Medchem annotations": annotations,
        }

    def __str__(self) -> str:
        baseline = molecule_label(MoleculeId(self.run.source_molecule_id))
        sections = [
            f"Run {self.run.id} · {self.run.status.replace('_', ' ')}",
            f"Baseline: {baseline}  {self.source_smiles}",
        ]
        if self.run.failure_reason:
            sections.append(self.run.failure_reason)
        if self.clusters:
            sections.append(
                f"Minimum Tanimoto similarity: {self.min_similarity:.0%} · "
                f"Minimum core coverage per cluster: {self.min_core_coverage:.0%}"
            )
        for name, rows in self.tables().items():
            if rows:
                sections.append(
                    name
                    + "\n"
                    + format_table(
                        tuple(rows[0]), [tuple(row.values()) for row in rows]
                    )
                )
        return "\n\n".join(sections)


def alert_names(annotation: StructuralAnnotation) -> set[str]:
    return {f"{hit.catalog}: {hit.description}" for hit in annotation.alerts}


def computed_metrics(values: Mapping[str, MetricValue] | None) -> dict[str, Metric]:
    return {
        name: Metric(item.value, item.unit, item.method)
        for name, item in (values or {}).items()
        if isinstance(item, ComputedMetric)
    }


def inspect_run(
    run: GenerationRunResponse,
    source_smiles: str,
    *,
    min_similarity: float,
    min_core_coverage: float,
) -> RunReview:
    """Analyze the saved batch relative to its recorded source."""
    if not 0 <= min_similarity <= 1:
        raise ParseError("min_similarity must be from 0 to 1.")
    if not 0 < min_core_coverage <= 1:
        raise ParseError("min_core_coverage must be greater than 0 and at most 1.")
    source_id = MoleculeId(run.source_molecule_id)
    structures = {MoleculeId(row.molecule_id): row.smiles for row in run.candidates}
    member_ids, smiles = tuple(structures), tuple(structures.values())
    all_smiles = {source_id: source_smiles} | structures
    annotations = dict(
        zip(
            all_smiles,
            StructuralAnnotator().annotate_many(tuple(all_smiles.values())),
            strict=True,
        )
    )
    clusters = []
    if smiles:
        mols = [to_mol(text) for text in smiles]
        fps = Fingerprints.from_mols(mols, include_chirality=True)
        for cluster in cluster_butina(fps, cutoff=1 - min_similarity).clusters:
            indices = cluster.indices
            ids = tuple(member_ids[i] for i in indices)
            representative = member_ids[fps.medoid(indices)]
            if len(indices) < 2:
                clusters.append(
                    ClusterReview(ids, representative, None, None, "Singleton.")
                )
                continue
            core = find_mcs(
                [mols[i] for i in indices],
                min_coverage=min_core_coverage,
            )
            if isinstance(core, Failure):
                clusters.append(
                    ClusterReview(ids, representative, None, None, core.reason)
                )
                continue
            matched_smiles = [smiles[indices[i]] for i in core.matched_indices]
            try:
                groups = decompose_rgroups(matched_smiles, core)
            except ParseError as exc:
                clusters.append(
                    ClusterReview(ids, representative, core, None, str(exc))
                )
            else:
                clusters.append(ClusterReview(ids, representative, core, groups, None))
    front = prediction_front(
        source_id,
        computed_metrics(run.source_predictions),
        {
            MoleculeId(row.molecule_id): computed_metrics(row.prediction_snapshot)
            for row in run.candidates
        },
        run.criteria or {},
    )
    return RunReview(
        run,
        source_smiles,
        tuple(clusters),
        annotations,
        front,
        min_similarity,
        min_core_coverage,
    )
