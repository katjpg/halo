import re
from dataclasses import asdict, dataclass
from typing import Annotated, get_args

from pydantic import Field

from halo.llm.prompts.orchestrator import SYSTEM_PROMPT
from halo.llm.runtime import Agent, AgentOutput, ModelRunner, ModelTrace, run_agent
from halo.llm.validate import ValidationIssue
from halo.models.session import (
    STATE,
    RecallScope,
    RecentTurn,
    RoutedTask,
    Scope,
    SessionState,
)
from halo.types import (
    MAX_SELECTION,
    RECORD_ID,
    ADMETProperty,
    AssayEndpoint,
    AssayName,
    BatchSize,
    CriterionDirection,
    MoleculeId,
)

TASKS = {
    "profile": "Read computed or predicted properties.",
    "retrieve": "Find research papers and patent excerpts for a structure.",
    "design": "Generate analogs from the active parent.",
    "cluster": "Group molecules by structure and choose representatives.",
    "plot": "Plot predicted properties.",
    "refine_plot": "Update the selection in the current plot.",
    "strategy": "Propose a next experiment.",
    "converge": "Select candidates for follow-up.",
    "activity": "Look up public experimental activity in ChEMBL for an exact structure.",
    "assay_results": "Read or compare assay measurements saved in the project.",
    "recall": "Recall a confirmed choice, reason, or preference.",
    "clarify": "Ask for a required detail or direct the user to a confirmed action.",
}
assert set(TASKS) == set(get_args(RoutedTask))

PLOT_REFERENCE = re.compile(
    r"\b(?:same|current|previous|last|this|that)\s+(?:scatter|plot|chart|figure)\b",
    re.IGNORECASE,
)


class Criterion(AgentOutput):
    property: ADMETProperty
    direction: CriterionDirection


class RoutePlan(AgentOutput):
    task: RoutedTask
    scope: Scope
    molecule_ids: Annotated[
        tuple[Annotated[MoleculeId, RECORD_ID], ...], Field(max_length=MAX_SELECTION)
    ]
    recall_scope: RecallScope | None
    criteria: tuple[Criterion, ...]
    n: BatchSize | None
    question: str | None
    include_measurements: bool | None = None
    assay: AssayName | None = None
    endpoint: AssayEndpoint | None = None


@dataclass(frozen=True)
class RouteContext:
    state: SessionState
    lead_id: MoleculeId
    batch: tuple[MoleculeId, ...]
    recent: tuple[RecentTurn, ...]


def build_context(context: RouteContext) -> dict:
    return {
        "state": STATE.dump_python(context.state, mode="json", by_alias=True),
        "lead_id": context.lead_id,
        "current_batch": context.batch,
        "recent_dialogue": [
            {**asdict(turn), "origin": "saved_record"} for turn in context.recent
        ],
    }


ORCHESTRATOR = Agent(
    role="orchestrator",
    prompt=SYSTEM_PROMPT,
    schema=RoutePlan,
    build_context=build_context,
)


async def route(
    task: ModelRunner,
    message: str,
    *,
    state: SessionState,
    lead_id: MoleculeId,
    batch: tuple[MoleculeId, ...],
    recent: tuple[RecentTurn, ...],
    trace: ModelTrace,
) -> RoutePlan:
    plan, call_index = await run_agent(
        ORCHESTRATOR,
        task,
        RouteContext(state, lead_id, batch, recent),
        request=message,
        constraints={
            "tasks": TASKS,
            "scopes": get_args(Scope),
            "id_format": "HALO-N refers to the molecule ID N.",
            "recall_scopes": get_args(RecallScope),
            "admet_criteria": get_args(CriterionDirection),
            "admet_properties": get_args(ADMETProperty),
        },
        trace=trace,
    )
    if plan.task in ("assay_results", "retrieve", "strategy") and any(
        label is not None
        and not re.search(rf"(?<!\w){re.escape(label)}(?!\w)", message, re.IGNORECASE)
        for label in (plan.assay, plan.endpoint)
    ):
        trace.record_validation(
            call_index, "rejected", (ValidationIssue.UNVERIFIED_ROUTE,)
        )
        return plan.model_copy(
            update={
                "task": "clarify",
                "question": "Which saved assay or endpoint should I use?",
            }
        )
    trace.record_validation(call_index, "accepted")
    if (
        state.plot_turn_id is not None
        and plan.task == "plot"
        and PLOT_REFERENCE.search(message)
    ):
        return plan.model_copy(update={"task": "refine_plot"})
    return plan
