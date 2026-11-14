from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass

from halo.chem.eval.objectives import (
    OBJECTIVES,
    extract_source_values,
)
from halo.chem.sar.rgroup import Decomposition
from halo.llm.prompts.optimizer import SYSTEM_PROMPT
from halo.llm.runtime import Agent, AgentOutput, ModelRunner, ModelTrace, run_agent
from halo.models.generation import DesignRequest
from halo.models.molecule import MoleculeProfile
from halo.models.research import Citation
from halo.models.session import DesignHistory
from halo.refs import DECISION_EXCERPT_CHARS, decision_citation
from halo.types import CriterionDirection, OptimizationTask


class DesignProposal(AgentOutput):
    """A molecule or a replacement fragment at one mapped site."""

    smiles: str
    site: int | None
    rationale: str
    evidence_ids: tuple[str, ...]


class DesignResponse(AgentOutput):
    proposals: tuple[DesignProposal, ...]


class OptimizationResponse(AgentOutput):
    smiles: str


@dataclass(frozen=True)
class OptimizationContext:
    source: MoleculeProfile
    criteria: Mapping[OptimizationTask, CriterionDirection]
    targets: Mapping[OptimizationTask, float]


@dataclass(frozen=True, kw_only=True)
class DesignContext:
    source: MoleculeProfile
    design: DesignRequest
    history: DesignHistory
    decomposition: Decomposition | None
    previous_proposals: Sequence[str]


def build_optimization_context(context: OptimizationContext) -> dict:
    source = context.source
    values = extract_source_values(source)
    goals = []
    for key, direction in context.criteria.items():
        target = context.targets.get(key)
        goals.append(
            {
                "property": key,
                "label": OBJECTIVES[key].label,
                "direction": direction,
                "source_value": values.get(key),
                "bound": (
                    {
                        "operator": ">=" if direction == "increase" else "<=",
                        "value": target,
                    }
                    if target is not None
                    else None
                ),
            }
        )
    return {"source": {"smiles": source.canonical_smiles}, "goals": goals}


def design_citations(history: DesignHistory) -> dict[str, tuple[Citation, ...]]:
    return {
        "user_preferences": tuple(map(decision_citation, history.preferences)),
        "decision_history": tuple(
            decision_citation(item, max_text_chars=DECISION_EXCERPT_CHARS)
            for item in history.decisions
        ),
    }


def build_design_context(context: DesignContext) -> dict:
    design = context.design
    decomposition = context.decomposition
    return build_optimization_context(
        OptimizationContext(context.source, design.criteria, {})
    ) | {
        "mapped_core": decomposition.core_smiles if decomposition else None,
        "sites": [
            {"site": site, "fragment": fragment}
            for site, fragment in decomposition.groups.items()
        ]
        if decomposition
        else [],
        "previous_edits": [
            {"site": edit.site, "fragment": edit.to_fragment_smiles}
            for edit in context.history.previous_edits[-8:]
        ],
        "previous_proposals": context.previous_proposals,
        **{
            key: [{**asdict(item), "origin": "saved_record"} for item in citations]
            for key, citations in design_citations(context.history).items()
        },
    }


OPTIMIZER = Agent(
    role="optimizer",
    prompt=SYSTEM_PROMPT,
    schema=DesignResponse,
    build_context=build_design_context,
)
OPTIMIZATION = Agent(
    role="optimizer",
    prompt=SYSTEM_PROMPT,
    schema=OptimizationResponse,
    build_context=build_optimization_context,
)


async def propose_designs(
    task: ModelRunner,
    source: MoleculeProfile,
    design: DesignRequest,
    *,
    history: DesignHistory,
    decomposition: Decomposition | None,
    n: int,
    previous_proposals: Sequence[str],
    validation: Sequence[str],
    trace: ModelTrace,
) -> tuple[tuple[DesignProposal, ...], int]:
    """Propose mapped replacements when given a decomposition."""
    response, call_index = await run_agent(
        OPTIMIZER,
        task,
        DesignContext(
            source=source,
            design=design,
            history=history,
            decomposition=decomposition,
            previous_proposals=previous_proposals,
        ),
        request=design.instructions,
        constraints={
            "n": n,
            "validation": validation,
            "task_instructions": (
                "Use a listed site and put the replacement fragment with its "
                "mapped attachment in smiles. Preserve the core, other sites, "
                "and stereochemistry."
                if decomposition
                else "Use null for site and put the complete analog in smiles."
            ),
        },
        trace=trace,
    )
    return response.proposals, call_index


async def propose_analog(
    task: ModelRunner,
    context: OptimizationContext,
    trace: ModelTrace,
) -> tuple[str, int]:
    response, call_index = await run_agent(
        OPTIMIZATION,
        task,
        context,
        request="Propose one analog toward the requested property goals.",
        constraints={
            "task_instructions": (
                "Preserve the scaffold and make the fewest changes needed. "
                "Return the complete analog in smiles."
            )
        },
        trace=trace,
    )
    return response.smiles, call_index
