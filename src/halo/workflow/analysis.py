import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, replace
from math import isfinite

from halo.chem.cluster import cluster_butina
from halo.chem.eval.objectives import ADMET_UNITS
from halo.chem.fingerprints import Fingerprints, prepare_mols
from halo.chem.mol import to_mol
from halo.chem.results import Failure, Metric
from halo.chem.sar.comparison import Transformation, compare_fragments
from halo.chem.sar.mcs import find_mcs
from halo.chem.sar.rgroup import build_rgroup_library, decompose_molecule
from halo.core.errors import ParseError
from halo.models.analysis import (
    Axis,
    ClusterAnalysis,
    ClusterGroup,
    PlotPoint,
    PlotSpec,
    ResultValue,
    ScatterPlot,
    assay_signature,
)
from halo.models.molecule import MoleculeProfile
from halo.refs import admet_ref
from halo.types import MAX_SELECTION, ADMETProperty, MoleculeId

CLUSTER_CUTOFF = 0.3
ADMET_FIELDS = tuple(ADMET_UNITS)
FIELD_ALIASES: dict[str, ADMETProperty] = {field: field for field in ADMET_FIELDS} | {
    "aqueous solubility": "solubility",
    "logs": "solubility",
    "intestinal absorption": "hia",
    "herg inhibition": "herg",
    "bbb permeability": "bbbp",
}
FIELD_PATTERN = re.compile(
    r"(?<!\w)(?:"
    + "|".join(
        re.escape(alias) for alias in sorted(FIELD_ALIASES, key=len, reverse=True)
    )
    + r")(?!\w)"
)


def parse_requested_fields(request: str) -> tuple[ADMETProperty, ...]:
    """Return supported fields in first-mention order."""
    return tuple(
        dict.fromkeys(
            FIELD_ALIASES[match[0]]
            for match in FIELD_PATTERN.finditer(request.casefold())
        )
    )


def prediction_values(
    profiles: Mapping[MoleculeId, MoleculeProfile],
    fields: Sequence[ADMETProperty] = ADMET_FIELDS,
) -> tuple[ResultValue, ...]:
    return tuple(
        ResultValue(
            id=admet_ref(mid, field),
            molecule_id=mid,
            endpoint=field,
            value=metric.value,
            unit=metric.unit,
            method=metric.method,
            source_type="predicted",
        )
        for mid, profile in profiles.items()
        for field in fields
        if isinstance(metric := profile.predictions.get(field), Metric)
        and isfinite(metric.value)
    )


def matched_assay_values(
    values: Sequence[ResultValue], molecule_ids: Sequence[MoleculeId]
) -> tuple[ResultValue, ...]:
    """Choose one result per molecule from the newest comparable group."""
    groups: dict[tuple[str, ...], dict[MoleculeId, ResultValue]] = {}
    for value in values:
        if signature := assay_signature(value):
            groups.setdefault(signature, {}).setdefault(value.molecule_id, value)
    for group in groups.values():
        if all(mid in group for mid in molecule_ids):
            return tuple(group[mid] for mid in molecule_ids)
    return ()


def describe_changes(source: str, target: str, core: str | None) -> str:
    """Describe changes at mapped sites or in fragments of a shared core."""
    if core:
        try:
            before, after = (
                decompose_molecule(smiles, core) for smiles in (source, target)
            )
        except ParseError:
            pass
        else:
            changes = [
                f"R{site}: {fragment} → {after.groups[site]}"
                for site, fragment in sorted(before.groups.items())
                if site in after.groups and fragment != after.groups[site]
            ]
            if (
                changes
                and before.core_smiles == after.core_smiles
                and before.groups.keys() == after.groups.keys()
            ):
                return (
                    "Shared mapped core and stereochemistry. Changed fragments: "
                    + "; ".join(changes)
                )
    change = compare_fragments(source, target)
    return str(asdict(change)) if isinstance(change, Transformation) else change.reason


def parse_plot_spec(request: str, previous_spec: PlotSpec | None = None) -> PlotSpec:
    """Parse scatter axes or change the selection on an existing plot."""
    request = request.casefold()
    field_matches = tuple(FIELD_PATTERN.finditer(request))
    selected_only = bool(re.search(r"\b(?:selected|selection)\b", request))
    if previous_spec is not None:
        if field_matches or re.search(
            r"\b(?:replace|switch|remove|drop|without|instead|size|sized|scale)\b|[<>≤≥]",
            request,
        ):
            raise ParseError("Start a new plot to change the axes or encoding.")
        all_candidates = bool(
            re.search(
                r"\ball (?:candidates|molecules|compounds|analog(?:ue)?s)\b", request
            )
        )
        return replace(
            previous_spec,
            selected_only=not all_candidates
            and (selected_only or previous_spec.selected_only),
        )
    axes = {}
    for match in field_matches:
        axis = re.match(r"\s+on\s+(?:the\s+)?([xy])\b", request[match.end() :])
        if axis:
            axes[axis[1]] = FIELD_ALIASES[match[0]]
    fields = tuple(dict.fromkeys(FIELD_ALIASES[match[0]] for match in field_matches))
    if set(axes) == {"x", "y"}:
        x, y = axes["x"], axes["y"]
    elif len(fields) == 2 and not axes:
        x, y = (
            reversed(fields)
            if re.search(r"\b(?:vs\.?|versus|against)\b", request)
            else fields
        )
    else:
        raise ParseError("Name two supported ADMET properties for the scatter.")
    try:
        return PlotSpec(x, y, selected_only)
    except ValueError as exc:
        raise ParseError(str(exc)) from exc


def build_scatter_plot(
    spec: PlotSpec,
    source_id: MoleculeId,
    profiles: Mapping[MoleculeId, MoleculeProfile],
    selected_ids: Sequence[MoleculeId] = (),
) -> ScatterPlot:
    """Build a scatter with predictions comparable to the source on both axes."""
    values = {
        (item.molecule_id, item.endpoint): item
        for item in prediction_values(profiles, (spec.x, spec.y))
    }
    x, y = values.get((source_id, spec.x)), values.get((source_id, spec.y))
    if x is None or y is None:
        raise ParseError("The source needs predictions for both plot axes.")
    source = PlotPoint(source_id, x.value, y.value, (x.id, y.id))
    points = []
    for mid in profiles:
        x_value, y_value = values.get((mid, spec.x)), values.get((mid, spec.y))
        if (
            x_value is not None
            and y_value is not None
            and (x_value.unit, x_value.method) == (x.unit, x.method)
            and (y_value.unit, y_value.method) == (y.unit, y.method)
        ):
            points.append(
                PlotPoint(mid, x_value.value, y_value.value, (x_value.id, y_value.id))
            )

    def axis(
        field: ADMETProperty, metric: ResultValue, values: Sequence[float]
    ) -> Axis:
        lo, hi = min(values), max(values)
        padding = (hi - lo) * 0.08 if hi > lo else max(abs(lo) * 0.08, 0.01)
        return Axis(
            field=field,
            unit=metric.unit,
            method=metric.method,
            lo=lo - padding,
            hi=hi + padding,
        )

    plot = ScatterPlot(
        spec,
        source,
        tuple(points),
        (),
        axis(spec.x, x, [point.x for point in points]),
        axis(spec.y, y, [point.y for point in points]),
    )
    return filter_plot(plot, spec, selected_ids)


def filter_plot(
    plot: ScatterPlot, spec: PlotSpec, selected_ids: Sequence[MoleculeId]
) -> ScatterPlot:
    """Keep the source and axis limits when filtering points."""
    visible_ids = tuple(
        point.molecule_id
        for point in plot.points
        if not spec.selected_only or point.molecule_id in selected_ids
    )
    if not visible_ids:
        raise ParseError("Choose molecules with comparable predictions for both axes.")
    return replace(plot, spec=spec, visible_ids=visible_ids)


def analyze_clusters(
    molecules: Mapping[MoleculeId, str], *, core_smiles: str | None = None
) -> ClusterAnalysis:
    """Cluster structures and choose the medoid of each group."""
    if not 1 <= len(molecules) <= MAX_SELECTION:
        raise ParseError(f"Select one to {MAX_SELECTION} molecules to cluster.")
    ids = tuple(sorted(molecules))
    mols = prepare_mols([to_mol(molecules[mid]) for mid in ids])
    fps = Fingerprints.from_mols(mols, include_chirality=True, prepared=True)
    groups = []
    for group in cluster_butina(fps, cutoff=CLUSTER_CUTOFF).clusters:
        member_ids = tuple(ids[i] for i in group.indices)
        medoid = fps.medoid(group.indices)
        member_smiles = [molecules[mid] for mid in member_ids]
        mcs = (
            find_mcs([mols[i] for i in group.indices], timeout_s=2)
            if len(member_smiles) > 1
            else None
        )
        if mcs is None:
            core_smarts, mcs_status = None, "skipped"
        elif isinstance(mcs, Failure):
            core_smarts, mcs_status = None, "timeout"
        else:
            core_smarts = mcs.smartsString or None
            mcs_status = "found" if core_smarts else "none"
        fragments = {}
        if core := core_smiles or core_smarts:
            try:
                library = build_rgroup_library(
                    member_smiles,
                    core,
                    core_notation="smiles" if core_smiles else "smarts",
                )
            except ParseError:
                pass
            else:
                fragments = {
                    site.label: tuple(option.smiles for option in site.options)
                    for site in library.sites
                }
        groups.append(
            ClusterGroup(member_ids, ids[medoid], core_smarts, mcs_status, fragments)
        )
    return ClusterAnalysis(tuple(groups), fps.method, CLUSTER_CUTOFF)
