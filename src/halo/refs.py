import re
from typing import Literal

from halo.models.research import Citation
from halo.models.session import Decision
from halo.types import ADMETProperty, AssayResultId, MoleculeId

MOLECULE_PREFIX = "HALO-"
DECISION_PREFIX = "DECISION/"
ASSAY_PREFIX = "ASSAY/"
MOLECULE_LABEL = re.compile(rf"\b{MOLECULE_PREFIX}(\d+)\b", re.IGNORECASE)
DECISION_LABEL = re.compile(rf"\b{DECISION_PREFIX}(\d+)\b", re.IGNORECASE)
DECISION_EXCERPT_CHARS = 350

PROFILE = "PROFILE"
PARETO = "PARETO"
CLUSTERS = "CLUSTERS"
COMPARISON = "COMPARISON"


def molecule_label(molecule_id: MoleculeId) -> str:
    return f"{MOLECULE_PREFIX}{molecule_id}"


def admet_ref(molecule_id: MoleculeId, field: ADMETProperty) -> str:
    return f"ADMET/{molecule_label(molecule_id)}/{field}"


def chembl_activity_ref(activity_id: int) -> str:
    return f"CHEMBL/activity/{activity_id}"


def assay_ref(result_id: AssayResultId) -> str:
    return f"{ASSAY_PREFIX}{result_id}"


def passage_ref(
    source_type: Literal["paper", "patent"],
    publication_id: str,
    idx: int,
    *,
    paragraph_index: int | None = None,
) -> str:
    if paragraph_index is not None:
        return (
            f"{source_type.upper()}/{publication_id}/full-text-{paragraph_index}-{idx}"
        )
    return f"{source_type.upper()}/{publication_id}/abstract-{idx}"


def medchem_ref(molecule_id: MoleculeId) -> str:
    return f"MEDCHEM/{molecule_label(molecule_id)}"


def decision_citation(item: Decision, *, max_text_chars: int | None = None) -> Citation:
    text = item.note
    if max_text_chars is not None and len(text) > max_text_chars:
        text = text[:max_text_chars].rsplit(" ", 1)[0] + "…"
    if item.type == "set_preference":
        text = f"Scientist preference for design, scope {item.scope}: {text}"
    else:
        text = f"The scientist accepted the linked proposal. Their reason: {text}"
        if item.proposal:
            labels = " vs ".join(map(molecule_label, item.proposal.comparison_ids))
            text += f" Comparison: {labels}."
    return Citation(f"{DECISION_PREFIX}{item.id}", text)
