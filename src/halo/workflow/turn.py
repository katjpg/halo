from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime
from time import perf_counter
from typing import TYPE_CHECKING, Protocol, TypeVar

from pydantic import ValidationError

from halo.chem.eval.objectives import is_current
from halo.chem.eval.pareto import prediction_front
from halo.chem.results import Metric
from halo.core.errors import (
    ApplicationError,
    ConflictError,
    LimitError,
    NotFoundError,
    ParseError,
    ProviderError,
)
from halo.llm.agents.orchestrator import TASKS, RoutePlan, route
from halo.llm.agents.researcher import respond
from halo.llm.agents.strategist import propose_strategy as generate_strategy
from halo.llm.runtime import ModelCall, ModelRunner, ModelTrace
from halo.llm.validate import cited_names, validate_text
from halo.models.analysis import (
    AssayResult,
    MoleculeAnnotation,
    ResultValue,
)
from halo.models.generation import (
    DesignRequest,
    GenerationBatch,
    GenerationWrite,
    Occurrence,
    RGroupEdit,
)
from halo.models.molecule import MoleculeProfile
from halo.models.research import Citation
from halo.models.session import (
    ACTION,
    STATE,
    AcceptProposal,
    Action,
    Decision,
    Design,
    DesignHistory,
    ExperimentProposal,
    Project,
    Recall,
    RecentTurn,
    Research,
    Select,
    SessionState,
    SetPreference,
    StartedTurn,
    TurnRecord,
    TurnResult,
    TurnStatus,
)
from halo.refs import (
    CLUSTERS,
    COMPARISON,
    DECISION_EXCERPT_CHARS,
    DECISION_LABEL,
    MOLECULE_LABEL,
    PARETO,
    PROFILE,
    assay_ref,
    decision_citation,
    medchem_ref,
    molecule_label,
)
from halo.types import (
    DEFAULT_BATCH_SIZE,
    MAX_RECALL,
    MAX_RECORD_ID,
    MAX_SELECTION,
    ADMETProperty,
    DecisionId,
    MoleculeId,
    Progress,
    SessionId,
    TurnId,
    UserId,
)
from halo.workflow.analysis import (
    ADMET_FIELDS,
    analyze_clusters,
    build_scatter_plot,
    describe_changes,
    filter_plot,
    matched_assay_values,
    parse_plot_spec,
    parse_requested_fields,
    prediction_values,
)
from halo.workflow.generation import GenerationRunner, design_config
from halo.workflow.memory import (
    design_history,
    recall_records,
    select_preferences,
)
from halo.workflow.research import ResearchWorkflow

if TYPE_CHECKING:
    from halo.chem.medchem import StructuralAnnotation


logger = logging.getLogger(__name__)
TURN_TIMEOUT_S = 130
MAX_TOOL_CALLS = 4
MAX_CONTEXT_MOLECULES = 6
MAX_CONTEXT_RESULTS = 32
MAX_ASSAY_RESULTS = 4
T = TypeVar("T")


def parse_action(data: Mapping[str, object]) -> Action:
    try:
        return ACTION.validate_python(data)
    except ValidationError as exc:
        raise ParseError(exc.errors()[0]["msg"]) from exc


class TurnWriter(Protocol):
    async def start_turn(
        self,
        user_id: UserId,
        session_id: SessionId,
        request: dict,
        request_id: str | None,
    ) -> StartedTurn | TurnRecord: ...
    async def finish_turn(
        self,
        start: StartedTurn,
        *,
        status: TurnStatus,
        result: TurnResult,
        state: SessionState,
        trace: dict,
        calls: Sequence[ModelCall],
        generation: GenerationWrite | None = None,
    ) -> TurnRecord: ...


class TurnReader(Protocol):
    async def latest_batch(self, project: Project) -> GenerationBatch | None: ...
    async def molecule_ids(
        self, project: Project, ids: tuple[MoleculeId, ...]
    ) -> set[MoleculeId]: ...
    async def profiles(
        self, project: Project, ids: tuple[MoleculeId, ...] | None, *, limit: int
    ) -> dict[MoleculeId, MoleculeProfile]: ...
    async def profile_at(
        self, project: Project, source: Occurrence
    ) -> MoleculeProfile: ...
    async def core_smiles(self, project: Project, source: Occurrence) -> str | None: ...
    async def recent_edits(self, project: Project) -> tuple[RGroupEdit, ...]: ...
    async def matching_smiles(
        self, project: Project, smiles: Sequence[str]
    ) -> set[str]: ...
    async def recent_turns(self, start: StartedTurn) -> tuple[RecentTurn, ...]: ...
    async def turn(self, project: Project, turn_id: TurnId) -> TurnRecord: ...
    async def assay_results(
        self,
        project: Project,
        *,
        molecule_ids: tuple[MoleculeId, ...] | None = None,
        assay: str | None = None,
        endpoint: str | None = None,
        limit: int | None = None,
    ) -> tuple[AssayResult, ...]: ...
    async def decisions(
        self,
        project: Project,
        *,
        ids: tuple[DecisionId, ...] = (),
        as_of: datetime | None = None,
    ) -> tuple[Decision, ...]: ...


@dataclass
class TurnTrace:
    turn_id: TurnId
    model: ModelTrace
    tools: list[dict] = field(default_factory=list)
    decision_reads: list[DecisionId] = field(default_factory=list)
    applied_decision_ids: tuple[DecisionId, ...] = ()
    progress: Progress | None = None

    async def call(
        self, name: str, function: Callable[..., Awaitable[T]], *args, **kwargs
    ) -> T:
        if len(self.tools) >= MAX_TOOL_CALLS:
            raise ProviderError("The tool call limit was reached.")
        start = perf_counter()
        status = "failed"
        if self.progress:
            self.progress(f"{name}…")
        try:
            result = await function(*args, **kwargs)
            status = "completed"
            return result
        finally:
            self.tools.append(
                {
                    "id": f"TOOL/{self.turn_id}/{len(self.tools) + 1}",
                    "name": name,
                    "status": status,
                    "elapsed_ms": (perf_counter() - start) * 1000,
                }
            )

    def record(self, before: SessionState, after: SessionState) -> dict:
        return {
            "tools": self.tools,
            "decision_reads": tuple(dict.fromkeys(self.decision_reads)),
            "applied_decision_ids": self.applied_decision_ids,
            "state_before": STATE.dump_python(before, mode="json", by_alias=True),
            "state_after": STATE.dump_python(after, mode="json", by_alias=True),
        }


@dataclass(frozen=True)
class Transition:
    result: TurnResult
    state: SessionState
    generation: GenerationWrite | None = None


class TurnRunner:
    def __init__(
        self,
        reader: TurnReader,
        writer: TurnWriter,
        task: ModelRunner,
        generator: GenerationRunner,
        research: ResearchWorkflow,
        annotate: Callable[[Sequence[str]], Sequence[StructuralAnnotation]]
        | None = None,
    ) -> None:
        self.reader = reader
        self.writer = writer
        self.task = task
        self.generator = generator
        self.research = research
        self.annotate = annotate

    async def run(
        self,
        *,
        user_id: UserId,
        session_id: SessionId,
        message: str | None,
        action: Action | None = None,
        request_id: str | None = None,
        progress: Progress | None = None,
        on_start: Callable[[TurnId], None] | None = None,
    ) -> TurnRecord:
        """Replay the saved turn when its request ID and body match."""
        if (message is None) == (action is None):
            raise ParseError("Provide a message or a confirmed action.")
        request = {
            "message": message,
            "action": ACTION.dump_python(action, mode="json", by_alias=True)
            if action
            else None,
        }
        start = await self.writer.start_turn(user_id, session_id, request, request_id)
        if isinstance(start, TurnRecord):
            if on_start is not None:
                on_start(start.id)
            if progress:
                progress("Loaded the saved turn.")
            return start
        trace = TurnTrace(start.turn.id, self.task.new_trace(), progress=progress)
        try:
            if on_start is not None:
                on_start(start.turn.id)
            if progress:
                progress("Started the turn.")
            async with asyncio.timeout(TURN_TIMEOUT_S):
                transition = await self.dispatch(start, message, action, trace)
            turn = await self.writer.finish_turn(
                start,
                status="completed",
                result=transition.result,
                state=transition.state,
                trace=trace.record(start.state, transition.state),
                calls=trace.model.calls,
                generation=transition.generation,
            )
        except (Exception, asyncio.CancelledError) as exc:
            trace.applied_decision_ids = ()
            error = "The turn could not complete."
            error_code = (
                exc.code if isinstance(exc, ApplicationError) else "internal_error"
            )
            if isinstance(exc, (ParseError, ConflictError, NotFoundError, LimitError)):
                error = str(exc)
            elif isinstance(exc, TimeoutError):
                error = "The turn reached its time limit."
                error_code = "timeout"
            elif isinstance(exc, asyncio.CancelledError):
                error = "The turn was cancelled."
                error_code = "cancelled"
            try:
                result = await self.writer.finish_turn(
                    start,
                    status="cancelled"
                    if isinstance(exc, asyncio.CancelledError)
                    else "failed",
                    result=TurnResult(error=error, error_code=error_code),
                    state=start.state,
                    trace=trace.record(start.state, start.state),
                    calls=trace.model.calls,
                )
            except Exception:
                logger.exception("could not finalize turn %s", start.turn.id)
            else:
                if error_code not in ("internal_error", "cancelled"):
                    return result
            raise
        if progress:
            progress("Saved the turn and its results.")
        return turn

    async def dispatch_action(
        self,
        start: StartedTurn,
        action: Action,
        trace: TurnTrace,
    ) -> Transition:
        project = start.project
        state = start.state
        if isinstance(action, AcceptProposal):
            turn = await self.reader.turn(start.project, action.proposal_turn_id)
            if (
                not turn.output
                or turn.output.task != "strategy"
                or turn.output.proposal is None
                or turn.status != "completed"
            ):
                raise NotFoundError("The proposal does not exist in this project.")
            labels = " vs ".join(
                map(molecule_label, turn.output.proposal.comparison_ids)
            )
            return Transition(
                TurnResult("decide", message=f"Accepted {labels}."),
                state,
            )
        if isinstance(action, SetPreference):
            return Transition(
                TurnResult("remember", message="Saved the design preference."),
                state,
            )
        if isinstance(action, Recall):
            result = await self.recall(
                start.project, "Recall the selected decisions.", action, trace
            )
            return Transition(result, state)

        if isinstance(action, Select):
            parent = action.parent or state.parent
            if action.parent and parent.molecule_id not in action.molecule_ids:
                raise ParseError("Choose the active parent from the selection.")
            if set(action.molecule_ids) - await self.reader.molecule_ids(
                project, action.molecule_ids
            ):
                raise NotFoundError("A selected molecule is outside this project.")
            if action.parent is not None:
                await self.reader.profile_at(project, parent)
            state = replace(
                state,
                parent=parent,
                selected_ids=tuple(dict.fromkeys(action.molecule_ids)),
            )
            return Transition(
                TurnResult(
                    "select",
                    message=f"{molecule_label(parent.molecule_id)} is the active parent.",
                ),
                state,
            )
        if isinstance(action, Design):
            return await self.design(project, state, action, trace)

        if isinstance(action, Research):
            profiles = await self.load_profiles(project, (action.molecule_id,))
            result = await self.retrieve(
                action.molecule_id,
                profiles[action.molecule_id].canonical_smiles,
                action.question,
                trace,
            )
            return Transition(result, state)

        raise ParseError("Provide a confirmed action.")

    async def dispatch(
        self,
        start: StartedTurn,
        message: str | None,
        action: Action | None,
        trace: TurnTrace,
    ) -> Transition:
        if action is not None:
            return await self.dispatch_action(start, action, trace)
        project = start.project
        state = start.state
        if message is None:
            raise ParseError("Provide a message or a confirmed action.")
        named = tuple(
            dict.fromkeys(
                MoleculeId(int(mid)) for mid in MOLECULE_LABEL.findall(message)
            )
        )
        if len(named) > MAX_SELECTION:
            raise ParseError(f"Choose up to {MAX_SELECTION} molecules.")
        if any(not 1 <= mid <= MAX_RECORD_ID for mid in named) or (
            set(named) - await self.reader.molecule_ids(project, named)
        ):
            raise NotFoundError("A named molecule is outside this project.")
        batch, recent = await asyncio.gather(
            self.reader.latest_batch(project), self.reader.recent_turns(start)
        )
        plan = await route(
            self.task,
            message,
            state=state,
            lead_id=project.lead_id,
            batch=batch.molecule_ids if batch else (),
            recent=recent,
            trace=trace.model,
        )
        known_ids = {
            project.lead_id,
            state.parent.molecule_id,
            *state.selected_ids,
            *(batch.molecule_ids if batch else ()),
            *(mid for turn in recent for mid in turn.molecule_ids),
        }
        for turn in recent:
            known_ids.update(
                MoleculeId(int(mid)) for mid in MOLECULE_LABEL.findall(turn.request)
            )
        if plan.task not in TASKS:
            return Transition(
                TurnResult(
                    "clarify",
                    message="Use a confirmed action to change the project state.",
                ),
                state,
            )
        if plan.task == "clarify":
            question = plan.question or "Which molecule or criterion should I use?"
            validate_text(
                question,
                names=cited_names(message) | set(map(molecule_label, known_ids)),
            )
            return Transition(TurnResult("clarify", message=question), state)
        if plan.task == "recall":
            ids = tuple(
                dict.fromkeys(
                    DecisionId(int(ref)) for ref in DECISION_LABEL.findall(message)
                )
            )[:MAX_RECALL]
            recall = parse_action(
                {
                    "type": "recall",
                    "decision_ids": ids,
                    "scope": plan.recall_scope or "all",
                }
            )
            assert isinstance(recall, Recall)
            result = await self.recall(
                start.project,
                message,
                recall,
                trace,
                molecule_ids=named,
            )
            return Transition(result, state)
        if plan.task == "refine_plot":
            if state.plot_turn_id is None:
                raise ParseError("Create a scatter before refining it.")
            turn = await self.reader.turn(start.project, state.plot_turn_id)
            if (
                turn.session_id != start.turn.session_id
                or not turn.output
                or turn.output.plot is None
            ):
                raise NotFoundError("The active plot is unavailable in this session.")
            plot = turn.output.plot
            plot = filter_plot(
                plot, parse_plot_spec(message, plot.spec), state.selected_ids
            )
            return Transition(
                TurnResult(
                    "refine_plot",
                    message="Updated the plot selection.",
                    plot=plot,
                    values=turn.output.values,
                ),
                replace(state, plot_turn_id=start.turn.id),
            )

        if not named and set(plan.molecule_ids) - known_ids:
            return Transition(
                TurnResult(
                    "clarify", message="Which molecule should I use for this request?"
                ),
                state,
            )
        resolved = named or tuple(dict.fromkeys(plan.molecule_ids))
        if plan.task == "design":
            if resolved and resolved != (state.parent.molecule_id,):
                return Transition(
                    TurnResult(
                        "clarify",
                        message="Select the requested molecule as the active parent.",
                    ),
                    state,
                )
            criteria = {item.property: item.direction for item in plan.criteria}
            if len(criteria) != len(plan.criteria):
                raise ParseError("Choose one direction for each ADMET criterion.")
            if not criteria and batch:
                criteria = batch.criteria
            if not criteria:
                return Transition(
                    TurnResult(
                        "clarify",
                        message="Which ADMET properties should increase, decrease, or stay close to the source?",
                    ),
                    state,
                )
            design = parse_action(
                {
                    "type": "design",
                    "criteria": criteria,
                    "instructions": message,
                    "n": plan.n if plan.n is not None else DEFAULT_BATCH_SIZE,
                }
            )
            assert isinstance(design, Design)
            return await self.design(project, state, design, trace)

        if plan.task == "strategy":
            return await self.propose_strategy(
                project,
                state,
                message,
                resolved,
                trace,
                assay=plan.assay,
                endpoint=plan.endpoint,
            )

        ids = (
            resolved
            or {
                "lead": (project.lead_id,),
                "parent": (state.parent.molecule_id,),
                "selected": state.selected_ids,
                "batch": batch.molecule_ids if batch else (),
                "all": None,
            }[plan.scope]
        )
        if plan.task == "assay_results":
            result = await self.summarize_assay_results(
                project, ids, message, trace, assay=plan.assay, endpoint=plan.endpoint
            )
            return Transition(result, state)
        is_lookup = plan.task in ("retrieve", "activity")
        if is_lookup and ids is not None and len(ids) != 1:
            raise ParseError("Choose one molecule for the lookup.")
        profiles = (
            await self.reader.profiles(project, None, limit=2)
            if is_lookup and ids is None
            else await self.load_profiles(project, ids)
        )
        if is_lookup:
            if len(profiles) != 1:
                raise ParseError("Choose one molecule for the lookup.")
            ((mid, profile),) = profiles.items()
            if plan.task == "retrieve":
                values = (
                    await self.load_assay_values(
                        project, (mid,), trace, assay=plan.assay, endpoint=plan.endpoint
                    )
                    if plan.include_measurements
                    else ()
                )
                return Transition(
                    await self.retrieve(
                        mid, profile.canonical_smiles, message, trace, values=values
                    ),
                    state,
                )
            values = await trace.call(
                "Look up exact ChEMBL activities",
                self.research.activities,
                mid,
                profile.canonical_smiles,
                message,
            )
            return Transition(
                TurnResult(
                    "activity",
                    message=f"ChEMBL assay records for {molecule_label(mid)}."
                    if values
                    else f"No matching activity was found for {molecule_label(mid)} in the ChEMBL lookup.",
                    values=values,
                ),
                state,
            )
        if plan.task == "cluster":
            data = {mid: profile.canonical_smiles for mid, profile in profiles.items()}
            groups = await trace.call(
                "Cluster structures",
                asyncio.to_thread,
                analyze_clusters,
                data,
                core_smiles=await self.reader.core_smiles(project, state.parent),
            )
            return Transition(
                TurnResult(
                    "cluster",
                    message="Structural clusters and their medoids.",
                    clusters=groups,
                ),
                state,
            )
        baseline = batch.source if plan.scope == "batch" and batch else state.parent
        if plan.task == "converge":
            return await self.select_candidates(
                start, batch, message, profiles, trace, baseline=baseline
            )
        return await self.analyze(start, baseline, plan, message, profiles, trace)

    async def retrieve(
        self,
        molecule_id: MoleculeId,
        smiles: str,
        question: str,
        trace: TurnTrace,
        *,
        values: tuple[ResultValue, ...] = (),
    ) -> TurnResult:
        found, response = await trace.call(
            "Find and synthesize sources",
            self.research.respond,
            molecule_id=molecule_id,
            smiles=smiles,
            question=question,
            trace=trace.model,
            results=values,
        )
        return TurnResult(
            "retrieve",
            response=response,
            passages=found.passages,
            values=values,
            unavailable=found.unavailable,
            source_exclusions=found.exclusions,
        )

    async def load_assay_values(
        self,
        project: Project,
        ids: tuple[MoleculeId, ...] | None,
        trace: TurnTrace,
        *,
        assay: str | None = None,
        endpoint: str | None = None,
    ) -> tuple[ResultValue, ...]:
        if ids and len(ids) > MAX_ASSAY_RESULTS:
            raise ParseError(
                f"Choose up to {MAX_ASSAY_RESULTS} molecules for assay lookup."
            )

        compare = ids is not None and len(ids) > 1
        rows = await trace.call(
            "Read saved assay measurements",
            self.reader.assay_results,
            project,
            molecule_ids=ids,
            assay=assay,
            endpoint=endpoint,
            limit=None if compare else MAX_ASSAY_RESULTS,
        )
        values = tuple(
            ResultValue(
                id=assay_ref(row.id),
                molecule_id=row.molecule_id,
                endpoint=row.endpoint,
                value=row.value,
                unit=row.unit,
                method="Project assay",
                source_type="experimental",
                relation=row.relation,
                assay_id=row.assay,
                source_ref=row.source_ref,
                conditions=row.conditions,
            )
            for row in rows
        )
        if ids is None or len(ids) <= 1:
            return values

        matched = matched_assay_values(values, ids)
        selected = {value.id: value for value in matched}
        counts = dict.fromkeys(ids, int(bool(matched)))
        per_molecule = MAX_ASSAY_RESULTS // len(ids)
        for value in values:
            mid = value.molecule_id
            if value.id not in selected and counts[mid] < per_molecule:
                selected[value.id] = value
                counts[mid] += 1
        return tuple(selected.values())

    async def summarize_assay_results(
        self,
        project: Project,
        ids: tuple[MoleculeId, ...] | None,
        message: str,
        trace: TurnTrace,
        *,
        assay: str | None = None,
        endpoint: str | None = None,
    ) -> TurnResult:
        values = await self.load_assay_values(
            project, ids, trace, assay=assay, endpoint=endpoint
        )
        if not values:
            return TurnResult(
                "assay_results", message="No saved measurements match this selection."
            )
        try:
            response = await respond(
                self.task,
                message,
                molecules={
                    mid: None
                    for mid in (ids or tuple(row.molecule_id for row in values))
                },
                evidence=(),
                results=values,
                trace=trace.model,
            )
        except ProviderError:
            return TurnResult(
                "assay_results", values=values, unavailable=("Assay synthesis",)
            )
        return TurnResult("assay_results", response=response, values=values)

    async def load_profiles(
        self, project: Project, ids: tuple[MoleculeId, ...] | None
    ) -> dict[MoleculeId, MoleculeProfile]:
        if ids == ():
            raise ParseError("Choose the molecules for this request.")
        profiles = await self.reader.profiles(project, ids, limit=MAX_SELECTION + 1)
        if len(profiles) > MAX_SELECTION:
            raise ParseError(f"Choose up to {MAX_SELECTION} molecules.")
        if ids is None:
            lead = project.lead_id
            if lead not in profiles:
                raise NotFoundError("The project source profile is unavailable.")
            return {lead: profiles[lead]} | profiles
        if set(ids) - profiles.keys():
            raise NotFoundError("A selected molecule is outside this project.")
        return {mid: profiles[mid] for mid in ids}

    async def design(
        self, project: Project, state: SessionState, action: Design, trace: TurnTrace
    ) -> Transition:
        source, records, edits = await asyncio.gather(
            self.reader.profile_at(project, state.parent),
            self.reader.decisions(project),
            self.reader.recent_edits(project),
        )
        preferences = select_preferences(records)
        history = design_history(records, state.parent.molecule_id)
        trace.decision_reads.extend(item.id for item in (*preferences, *history))
        design = DesignRequest(
            instructions=action.instructions,
            criteria=action.criteria,
            n=action.n,
            core_smiles=action.core_smiles
            or await self.reader.core_smiles(project, state.parent),
        )
        before = len(trace.model.calls)
        prepared = await trace.call(
            "Generate and verify analogs",
            self.generator.prepare,
            source,
            design,
            trace.model,
            history=DesignHistory(
                preferences=preferences, decisions=history, previous_edits=edits
            ),
            find_existing=lambda smiles: self.reader.matching_smiles(project, smiles),
            progress=trace.progress,
        )
        count = len(prepared.candidates)
        applied = tuple(item.id for item in preferences) if count else ()
        trace.applied_decision_ids = applied
        config = design_config(
            design,
            self.generator.model_settings,
            source=prepared.source,
            decision_ids=tuple(item.id for item in history),
        )
        usage = trace.model.usage(after=before)
        generation = GenerationWrite(
            state.parent,
            self.generator.optimizer.settings.model,
            config,
            prepared,
            usage,
            applied,
        )
        return Transition(
            TurnResult(
                "design",
                message=f"Verified {count} of {action.n} requested analogs.",
                generated_count=count,
            ),
            state,
            generation=generation,
        )

    async def recall(
        self,
        project: Project,
        message: str,
        action: Recall,
        trace: TurnTrace,
        *,
        molecule_ids: tuple[MoleculeId, ...] = (),
    ) -> TurnResult:
        scope = (
            "selected_preferences"
            if action.scope == "applicable_preferences"
            else action.scope
        )
        records = recall_records(
            await self.reader.decisions(
                project, ids=action.decision_ids, as_of=action.as_of
            ),
            scope,
            action.decision_ids,
            query=message,
            molecule_ids=molecule_ids,
        )
        trace.decision_reads.extend(item.id for item in records)
        if not records:
            return TurnResult(
                "recall",
                message="No matching confirmed decision was found in this project.",
            )
        evidence = tuple(map(decision_citation, records))
        molecules = {
            mid: None
            for item in records
            if item.proposal
            for mid in item.proposal.comparison_ids
        }
        response = await respond(
            self.task,
            message,
            molecules=molecules,
            evidence=tuple(
                decision_citation(item, max_text_chars=DECISION_EXCERPT_CHARS)
                for item in records
            ),
            evidence_origin="saved_record",
            trace=trace.model,
        )
        return TurnResult("recall", response=response, evidence=evidence)

    async def propose_strategy(
        self,
        project: Project,
        state: SessionState,
        message: str,
        named: tuple[MoleculeId, ...],
        trace: TurnTrace,
        *,
        assay: str | None = None,
        endpoint: str | None = None,
    ) -> Transition:
        parent = state.parent.molecule_id
        comparison = tuple(
            mid for mid in (named or state.selected_ids) if mid != parent
        )
        if len(comparison) != 1:
            return Transition(
                TurnResult(
                    "clarify",
                    message="Choose a single comparator for the active parent.",
                ),
                state,
            )
        ids = (parent, *comparison)
        profiles = await self.load_profiles(project, ids)
        molecules = {mid: profile.canonical_smiles for mid, profile in profiles.items()}
        change = await trace.call(
            "Compare structural fragments",
            asyncio.to_thread,
            describe_changes,
            *molecules.values(),
            await self.reader.core_smiles(project, state.parent),
        )
        evidence = [
            Citation(
                COMPARISON,
                f"Active parent {molecule_label(parent)}, selected control {molecule_label(comparison[0])}. "
                + change,
            )
        ]
        values = await self.load_assay_values(
            project, ids, trace, assay=assay, endpoint=endpoint
        )
        comparable_ids = tuple(value.id for value in matched_assay_values(values, ids))
        history = design_history(await self.reader.decisions(project), parent)
        trace.decision_reads.extend(item.id for item in history)
        saved_evidence = tuple(
            decision_citation(item, max_text_chars=DECISION_EXCERPT_CHARS)
            for item in history
        )
        response, _ = await generate_strategy(
            self.task,
            message,
            molecules=molecules,
            evidence=evidence,
            saved_evidence=saved_evidence,
            results=values,
            comparable_result_ids=comparable_ids,
            trace=trace.model,
        )
        evidence.extend(saved_evidence)
        if not response.findings:
            return Transition(
                TurnResult(
                    "strategy",
                    response=response,
                    evidence=tuple(evidence),
                    values=values,
                ),
                state,
            )
        proposal = ExperimentProposal(
            state.parent,
            ids,
            response.findings[0].text,
            response.findings[0].evidence_ids,
            response.findings[0].result_ids,
        )
        return Transition(
            TurnResult(
                "strategy",
                response=response,
                evidence=tuple(evidence),
                values=values,
                proposal=proposal,
            ),
            state,
        )

    async def select_candidates(
        self,
        start: StartedTurn,
        batch: GenerationBatch | None,
        message: str,
        profiles: dict[MoleculeId, MoleculeProfile],
        trace: TurnTrace,
        *,
        baseline: Occurrence,
    ) -> Transition:
        state = start.state
        criteria = batch.criteria if batch else None
        if not criteria:
            return Transition(
                TurnResult(
                    "clarify",
                    message="Set the ADMET criteria before selecting candidates.",
                ),
                state,
            )
        baseline_id = baseline.molecule_id
        baseline_profile = profiles.get(baseline_id)
        if baseline_profile is None:
            baseline_profile = await self.reader.profile_at(start.project, baseline)
        profiles = profiles | {baseline_id: baseline_profile}
        profiles = await self.update_predictions(profiles, tuple(criteria), trace)
        ids = prediction_front(
            baseline_id,
            profiles[baseline_id].predictions,
            {mid: row.predictions for mid, row in profiles.items()},
            criteria,
        )
        if not ids:
            return Transition(
                TurnResult(
                    "converge",
                    message="There are no candidates with comparable predictions for these criteria.",
                ),
                state,
            )
        grouping = await trace.call(
            "Cluster Pareto candidates",
            asyncio.to_thread,
            analyze_clusters,
            {mid: profiles[mid].canonical_smiles for mid in ids},
        )
        ordered = tuple(
            dict.fromkeys(
                (*(group.representative_id for group in grouping.groups), *ids)
            )
        )
        limit = min(MAX_CONTEXT_MOLECULES, MAX_CONTEXT_RESULTS // len(criteria))
        considered = {
            mid: profiles[mid] for mid in (baseline_id, *ordered[: limit - 1])
        }
        values = prediction_values(considered, tuple(criteria))
        evidence = (
            Citation(
                PARETO,
                f"Baseline {molecule_label(baseline_id)}. Source-relative criteria: {criteria}. Eligible IDs: "
                + ", ".join(
                    molecule_label(mid) for mid in considered if mid != baseline_id
                ),
            ),
            Citation(
                CLUSTERS,
                "Structural groups among eligible candidates: "
                + "; ".join(
                    ", ".join(
                        molecule_label(mid)
                        for mid in group.molecule_ids
                        if mid in considered
                    )
                    for group in grouping.groups
                    if any(mid in considered for mid in group.molecule_ids)
                ),
            ),
        )
        response, candidate_ids = await generate_strategy(
            self.task,
            message,
            molecules={
                mid: profile.canonical_smiles for mid, profile in considered.items()
            },
            evidence=evidence,
            results=values,
            eligible_ids=tuple(mid for mid in considered if mid != baseline_id),
            trace=trace.model,
        )
        return Transition(
            TurnResult(
                "converge",
                response=response,
                evidence=evidence,
                clusters=grouping,
                candidate_ids=candidate_ids,
                values=values,
                baseline_id=baseline_id,
                criteria=criteria,
            ),
            state,
        )

    async def analyze(
        self,
        start: StartedTurn,
        baseline: Occurrence,
        plan: RoutePlan,
        message: str,
        profiles: dict[MoleculeId, MoleculeProfile],
        trace: TurnTrace,
    ) -> Transition:
        state = start.state
        baseline_id = baseline.molecule_id
        if plan.scope == "batch" or plan.task == "plot":
            baseline_profile = profiles.get(baseline_id)
            if baseline_profile is None:
                baseline_profile = await self.reader.profile_at(start.project, baseline)
            profiles = {baseline_id: baseline_profile} | profiles
        if plan.task == "plot":
            spec = parse_plot_spec(message)
            profiles = await self.update_predictions(profiles, (spec.x, spec.y), trace)
            plot = build_scatter_plot(spec, baseline_id, profiles, state.selected_ids)
            result_ids = {ref for point in plot.points for ref in point.result_ids}
            return Transition(
                TurnResult(
                    "plot",
                    message="ADMET scatter with source reference values.",
                    plot=plot,
                    values=tuple(
                        row
                        for row in prediction_values(profiles, (spec.x, spec.y))
                        if row.id in result_ids
                    ),
                ),
                replace(state, plot_turn_id=start.turn.id),
            )
        profiles = await self.update_predictions(profiles, ADMET_FIELDS, trace)
        values = prediction_values(profiles)
        response = None
        unavailable = ()
        annotations = ()
        if self.annotate is not None:
            annotated = await trace.call(
                "Check structural groups and alerts",
                asyncio.to_thread,
                self.annotate,
                [profile.canonical_smiles for profile in profiles.values()],
            )
            annotations = tuple(
                MoleculeAnnotation(
                    mid,
                    tuple(sorted({group.name for group in item.groups})),
                    {
                        catalog: tuple(
                            sorted(
                                {
                                    a.description
                                    for a in item.alerts
                                    if a.catalog == catalog
                                }
                            )
                        )
                        for catalog in sorted({a.catalog for a in item.alerts})
                    },
                )
                for mid, item in zip(profiles, annotated, strict=True)
            )
        evidence = ()
        if values:
            fields = parse_requested_fields(message) or ADMET_FIELDS
            limit = min(MAX_CONTEXT_MOLECULES, MAX_CONTEXT_RESULTS // len(fields))
            considered = dict(list(profiles.items())[:limit])
            evidence = (
                Citation(
                    PROFILE,
                    f"ADMET predictions for comparison. Baseline {molecule_label(baseline_id)} when present.",
                ),
                *(
                    Citation(
                        medchem_ref(row.molecule_id),
                        f"{molecule_label(row.molecule_id)} structural alerts: {row.alerts or 'None listed'}."[
                            :300
                        ],
                    )
                    for row in annotations
                    if row.molecule_id in considered
                ),
            )
            try:
                response = await respond(
                    self.task,
                    message,
                    role="analyst",
                    molecules={
                        mid: profile.canonical_smiles
                        for mid, profile in considered.items()
                    },
                    evidence=evidence,
                    results=tuple(
                        item
                        for item in values
                        if item.molecule_id in considered and item.endpoint in fields
                    ),
                    trace=trace.model,
                )
            except ProviderError:
                unavailable = ("ADMET interpretation",)
        return Transition(
            TurnResult(
                "profile",
                message="ADMET predictions.",
                response=response,
                values=values,
                evidence=evidence,
                annotations=annotations,
                unavailable=unavailable,
            ),
            state,
        )

    async def update_predictions(
        self,
        profiles: Mapping[MoleculeId, MoleculeProfile],
        fields: Sequence[ADMETProperty],
        trace: TurnTrace,
    ) -> dict[MoleculeId, MoleculeProfile]:
        missing = tuple(
            mid
            for mid, profile in profiles.items()
            if any(
                field not in profile.predictions
                or (
                    isinstance(value := profile.predictions[field], Metric)
                    and not is_current(
                        field,
                        value,
                        method=self.generator.prediction_methods.get(field),
                    )
                )
                for field in fields
            )
        )
        if not missing:
            return dict(profiles)
        updated = await trace.call(
            "Compute ADMET predictions",
            self.generator.profile,
            [profiles[mid].canonical_smiles for mid in missing],
        )
        return dict(profiles) | dict(zip(missing, updated, strict=True))
