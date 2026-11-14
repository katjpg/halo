import re
from collections.abc import Sequence

from halo.models.session import Decision, RecallScope
from halo.refs import decision_citation
from halo.types import MAX_RECALL, DecisionId, MoleculeId


def select_preferences(records: Sequence[Decision]) -> tuple[Decision, ...]:
    return tuple(
        item
        for item in records
        if item.type == "set_preference"
        and (item.scope == "ongoing" or not item.applied)
    )[:MAX_RECALL]


def design_history(
    records: Sequence[Decision], parent_id: MoleculeId
) -> tuple[Decision, ...]:
    return tuple(
        item
        for item in records
        if item.type == "accept_proposal"
        and item.proposal is not None
        and parent_id in item.proposal.comparison_ids
    )[:MAX_RECALL]


def recall_records(
    records: Sequence[Decision],
    scope: RecallScope,
    ids: tuple[DecisionId, ...] = (),
    *,
    query: str = "",
    molecule_ids: tuple[MoleculeId, ...] = (),
) -> tuple[Decision, ...]:
    if ids:
        return tuple(item for item in records if item.id in ids)[:MAX_RECALL]
    if scope == "selected_preferences":
        return select_preferences(records)
    choices = [
        item
        for item in records
        if (
            not molecule_ids
            or item.proposal is None
            or set(molecule_ids).intersection(item.proposal.comparison_ids)
        )
        and (
            scope == "all"
            or (scope == "acceptances" and item.type == "accept_proposal")
            or (scope == "preferences" and item.type == "set_preference")
        )
    ]
    if query:
        words = set(re.findall(r"[a-z][a-z0-9]{3,}", query.casefold()))
        choices.sort(
            key=lambda item: len(
                words.intersection(
                    re.findall(
                        r"[a-z][a-z0-9]{3,}", decision_citation(item).text.casefold()
                    )
                )
            ),
            reverse=True,
        )
    return tuple(choices[:MAX_RECALL])
