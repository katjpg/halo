from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from halo.llm.agents.researcher import (
    ResearchContext,
    ResponseOutput,
    build_context,
    research_context,
    verify_evidence,
)
from halo.llm.prompts.strategist import SYSTEM_PROMPT
from halo.llm.runtime import Agent, ModelRunner, ModelTrace, run_agent
from halo.llm.validate import OutputValidationError, ValidationIssue
from halo.models.analysis import ResultValue
from halo.models.research import Citation, Response
from halo.types import MoleculeId

MAX_CANDIDATES = 2


class StrategyResponse(ResponseOutput):
    candidate_ids: tuple[MoleculeId, ...]


@dataclass(frozen=True)
class StrategyContext:
    research: ResearchContext
    eligible_ids: tuple[MoleculeId, ...]
    comparable_result_ids: tuple[str, ...] = ()


def build_strategy_context(context: StrategyContext) -> dict:
    return build_context(context.research) | {
        "comparable_result_ids": context.comparable_result_ids
    }


def verify_strategy(output: StrategyResponse, context: StrategyContext) -> None:
    ids = output.candidate_ids
    if (
        len(ids) > MAX_CANDIDATES
        or len(set(ids)) != len(ids)
        or set(ids) - set(context.eligible_ids)
        or (context.eligible_ids and (not ids or not output.findings))
    ):
        raise OutputValidationError(ValidationIssue.INVALID_CANDIDATES)
    verify_evidence(output, context.research)


STRATEGIST = Agent(
    role="strategist",
    prompt=SYSTEM_PROMPT,
    schema=StrategyResponse,
    build_context=build_strategy_context,
    verify=verify_strategy,
    max_attempts=2,
)


async def propose_strategy(
    task: ModelRunner,
    question: str,
    *,
    molecules: Mapping[MoleculeId, str],
    evidence: Sequence[Citation],
    trace: ModelTrace,
    eligible_ids: tuple[MoleculeId, ...] = (),
    results: Sequence[ResultValue] = (),
    saved_evidence: Sequence[Citation] = (),
    comparable_result_ids: tuple[str, ...] = (),
) -> tuple[Response, tuple[MoleculeId, ...]]:
    """Return a strategy and up to two eligible candidate IDs."""
    context = research_context(
        question, molecules, evidence, results, saved_evidence=saved_evidence
    )
    output, _ = await run_agent(
        STRATEGIST,
        task,
        StrategyContext(context, eligible_ids, comparable_result_ids),
        request=question,
        constraints={"eligible_ids": eligible_ids, "max_candidates": MAX_CANDIDATES},
        trace=trace,
    )
    return output.to_response(), output.candidate_ids
