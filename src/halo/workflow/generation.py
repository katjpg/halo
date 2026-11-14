from __future__ import annotations

import asyncio
import logging
from collections.abc import (
    AsyncIterator,
    Awaitable,
    Callable,
    Iterable,
    Mapping,
    Sequence,
)
from contextlib import asynccontextmanager
from dataclasses import dataclass
from threading import Lock
from typing import TYPE_CHECKING, Protocol

from halo.chem.eval.objectives import OBJECTIVES, is_current, objective_metric
from halo.chem.eval.profile import PropertyPredictor, profile_molecules
from halo.chem.results import Metric
from halo.chem.sar.comparison import protected_cores, verify_analog
from halo.chem.sar.rgroup import (
    Decomposition,
    decompose_molecule,
    infer_rgroup_edit,
    replace_rgroup,
)
from halo.core.errors import ConflictError, LimitError, ParseError, ProviderError
from halo.llm.agents.optimizer import (
    DesignProposal,
    OptimizationContext,
    design_citations,
    propose_analog,
    propose_designs,
)
from halo.llm.providers.base import ModelResponseError
from halo.llm.runtime import ModelCall, ModelRunner, ModelTrace
from halo.llm.validate import (
    OutputValidationError,
    ValidationIssue,
    cited_labels,
    cited_names,
    validate_text,
)
from halo.models.generation import (
    CandidateDesign,
    DesignConfig,
    DesignRequest,
    ModelSettings,
    PreparedCandidate,
    PreparedGeneration,
    RGroupEdit,
    RunConfig,
    RunOutcome,
    RunStart,
    SampleConfig,
)
from halo.models.molecule import MoleculeProfile
from halo.models.research import Citation
from halo.models.session import DesignHistory
from halo.models.usage import TokenUsage
from halo.types import (
    ADMETProperty,
    Criteria,
    DecisionId,
    GenerationRunId,
    MoleculeId,
    OptimizationTask,
    Progress,
    UserId,
)

if TYPE_CHECKING:
    from rdkit.Chem import Mol

logger = logging.getLogger(__name__)
MAX_DESIGN_ATTEMPTS = 3
OPTIMIZATION_TIMEOUT_S = 130


class GenerationRepository(Protocol):
    async def save_molecule(
        self, *, user_id: UserId, profile: MoleculeProfile
    ) -> MoleculeId: ...

    async def start_run(
        self,
        *,
        user_id: UserId,
        source_molecule_id: MoleculeId,
        model: str,
        config: RunConfig,
        request_id: str | None,
        request: dict | None,
    ) -> RunStart: ...

    async def finish_run(
        self,
        run_id: GenerationRunId,
        *,
        user_id: UserId,
        candidates: Sequence[PreparedCandidate],
        usage: TokenUsage,
        status: RunOutcome,
        criteria: Sequence[OptimizationTask] = (),
        source_predictions: Mapping[OptimizationTask, Metric] | None = None,
        calls: Sequence[ModelCall] = (),
    ) -> None: ...

    async def fail_run(
        self,
        run_id: GenerationRunId,
        *,
        user_id: UserId,
        reason: str,
        usage: TokenUsage,
        calls: Sequence[ModelCall] = (),
    ) -> None: ...


def design_config(
    design: DesignRequest,
    model_settings: ModelSettings,
    *,
    source: MoleculeProfile | None = None,
    decision_ids: tuple[DecisionId, ...] = (),
) -> DesignConfig:
    return DesignConfig(
        criteria=dict(design.criteria),
        n=design.n,
        core_smiles=design.core_smiles,
        instructions=design.instructions,
        model_settings=model_settings,
        decision_ids=decision_ids,
        source_predictions=source_predictions(source, design.criteria)
        if source
        else None,
    )


def source_predictions(
    source: MoleculeProfile, fields: Iterable[OptimizationTask]
) -> dict[OptimizationTask, Metric]:
    return {
        key: metric
        for key in fields
        if isinstance(metric := objective_metric(source, key), Metric)
    }


def verify_proposal(
    proposal: DesignProposal,
    decomposition: Decomposition | None,
    design: DesignRequest,
    source_smiles: str,
    records: Mapping[str, Citation],
    cores: Sequence[Mol] = (),
) -> CandidateDesign:
    if set(proposal.evidence_ids) - records.keys():
        raise OutputValidationError(ValidationIssue.UNKNOWN_DECISION)
    names = cited_names(design.instructions)
    for ref in proposal.evidence_ids:
        names.update(cited_names(records[ref].text) | cited_labels(records[ref].text))
        names.update(records[ref].identifiers)
    if proposal.site is not None and decomposition is not None:
        smiles, transformation = replace_rgroup(
            decomposition, site=proposal.site, fragment_smiles=proposal.smiles
        )
        names.add(f"R{proposal.site}")
    elif proposal.site is None and decomposition is None:
        smiles = verify_analog(proposal.smiles, cores or protected_cores(source_smiles))
        edit = infer_rgroup_edit(source_smiles, smiles)
        transformation = edit if isinstance(edit, RGroupEdit) else None
    else:
        raise ProviderError(
            "Use a mapped site for a fragment edit and null for a whole molecule."
        )
    validate_text(proposal.rationale, names=names)
    return CandidateDesign(
        smiles, proposal.rationale, proposal.evidence_ids, transformation
    )


@dataclass(frozen=True)
class GenerationStats:
    """Count standalone samples and the structures they returned."""

    requests: int
    extracted: int
    valid: int
    invalid: int
    source_duplicates: int
    repeated: int
    unique: int


class GenerationRunner:
    def __init__(
        self,
        optimizer: ModelRunner,
        predictor: PropertyPredictor,
        store: GenerationRepository,
        *,
        samples: int,
        prediction_methods: Mapping[ADMETProperty, str] | None = None,
        on_stats: Callable[[GenerationStats], None] | None = None,
    ) -> None:
        self.optimizer = optimizer
        self.predictor = predictor
        self.store = store
        self.samples = samples
        self.prediction_methods = dict(prediction_methods or {})
        self.model_settings = ModelSettings(
            max_tokens=optimizer.settings.max_tokens,
            temperature=optimizer.settings.temperature,
            reasoning_effort=optimizer.settings.reasoning_effort,
        )
        self.on_stats = on_stats
        self.inference = Lock()

    async def profile(self, smiles: Sequence[str]) -> list[MoleculeProfile]:
        def compute() -> list[MoleculeProfile]:
            with self.inference:
                return profile_molecules(smiles, [self.predictor])

        return await asyncio.to_thread(compute)

    async def save_profile(
        self, user_id: UserId, smiles: str
    ) -> tuple[MoleculeId, MoleculeProfile]:
        (profile,) = await self.profile([smiles])
        molecule_id = await self.store.save_molecule(user_id=user_id, profile=profile)
        return molecule_id, profile

    async def _prepare_source(
        self, source: MoleculeProfile, design: DesignRequest
    ) -> MoleculeProfile:
        def missing_predictions(profile: MoleculeProfile) -> list[ADMETProperty]:
            return [
                key
                for key in design.criteria
                if not is_current(
                    key,
                    profile.predictions.get(key),
                    method=self.prediction_methods.get(key),
                )
            ]

        if missing_predictions(source):
            (source,) = await self.profile([source.canonical_smiles])
            if missing := missing_predictions(source):
                raise ParseError(
                    f"Source predictions are unavailable for {', '.join(missing)}."
                )
        return source

    async def prepare(
        self,
        source: MoleculeProfile,
        design: DesignRequest,
        trace: ModelTrace,
        *,
        history: DesignHistory = DesignHistory(),
        find_existing: Callable[[Sequence[str]], Awaitable[set[str]]] | None = None,
        progress: Progress | None = None,
    ) -> PreparedGeneration:
        """Return up to the requested number of verified candidates in proposal order."""
        decomposition = (
            decompose_molecule(source.canonical_smiles, design.core_smiles)
            if design.core_smiles
            else None
        )
        if decomposition is not None and not decomposition.groups:
            raise ParseError("Choose a core with mapped attachment sites.")
        cores = (
            await asyncio.to_thread(protected_cores, source.canonical_smiles)
            if decomposition is None
            else ()
        )
        source = await self._prepare_source(source, design)
        if progress:
            progress("Source predictions ready; proposing analogs.")
        seen = {source.canonical_smiles}
        records = {
            item.id: item
            for citations in design_citations(history).values()
            for item in citations
        }
        designs = []
        validation: list[str] = []
        for _ in range(MAX_DESIGN_ATTEMPTS):
            if not trace.remaining or len(designs) == design.n:
                break
            remaining = design.n - len(designs)
            try:
                proposals, call_index = await propose_designs(
                    self.optimizer,
                    source,
                    design,
                    history=history,
                    decomposition=decomposition,
                    n=remaining,
                    previous_proposals=[item.smiles for item in designs],
                    validation=tuple(dict.fromkeys(validation))[-4:],
                    trace=trace,
                )
            except ModelResponseError:
                validation.append("Return the requested proposal schema.")
                continue
            issues: list[ValidationIssue] = []
            if not proposals:
                issues.append(ValidationIssue.MISSING_PROPOSALS)
            if len(proposals) > remaining:
                issues.append(ValidationIssue.PROPOSAL_LIMIT)
            verified: dict[str, CandidateDesign] = {}
            for proposal in proposals[: 2 * design.n]:
                if find_existing is None and len(verified) == remaining:
                    break
                try:
                    candidate = await asyncio.to_thread(
                        verify_proposal,
                        proposal,
                        decomposition,
                        design,
                        source.canonical_smiles,
                        records,
                        cores,
                    )
                except (ParseError, ProviderError) as exc:
                    issues.append(
                        exc.issue
                        if isinstance(exc, OutputValidationError)
                        else ValidationIssue.INVALID_STRUCTURE
                    )
                    continue
                if candidate.smiles in seen or candidate.smiles in verified:
                    issues.append(ValidationIssue.DUPLICATE_STRUCTURE)
                    continue
                verified[candidate.smiles] = candidate
            if find_existing is not None and verified:
                existing = await find_existing(tuple(verified))
                if existing:
                    issues.append(ValidationIssue.DUPLICATE_STRUCTURE)
                    seen.update(existing)
            accepted = [
                item for smiles, item in verified.items() if smiles not in seen
            ][:remaining]
            validation.extend(issue.value for issue in issues)
            if not accepted:
                status = "rejected"
            elif issues:
                status = "partial"
            else:
                status = "accepted"
            trace.record_validation(call_index, status, issues)
            designs.extend(accepted)
            seen.update(item.smiles for item in accepted)
        if progress:
            progress(f"Verified {len(designs)} of {design.n} requested analogs.")
        profiles = (
            await self.profile([item.smiles for item in designs]) if designs else []
        )
        if progress and profiles:
            progress("Evaluated candidate properties.")
        return PreparedGeneration(
            source,
            tuple(
                PreparedCandidate(profile=profile, design=design)
                for profile, design in zip(profiles, designs, strict=True)
            ),
        )

    async def optimize(
        self,
        *,
        user_id: UserId,
        source_molecule_id: MoleculeId,
        source_profile: MoleculeProfile,
        criteria: Criteria,
        n: int,
        request_id: str | None = None,
        progress: Progress | None = None,
        on_start: Callable[[GenerationRunId], None] | None = None,
    ) -> GenerationRunId:
        """Save up to n verified analogs with source and candidate predictions."""
        design = DesignRequest(
            instructions="Propose close analogs that pursue the requested ADMET criteria.",
            criteria=criteria,
            n=n,
        )
        async with self._record_run(
            user_id=user_id,
            source_molecule_id=source_molecule_id,
            config=design_config(design, self.model_settings),
            request_id=request_id,
            request={"criteria": dict(criteria), "n": n} if request_id else None,
        ) as (run_id, trace):
            if on_start is not None:
                on_start(run_id)
            if trace is None:
                if progress:
                    progress("Loaded the saved run.")
                return run_id
            if progress:
                progress("Started the generation run.")
            async with asyncio.timeout(OPTIMIZATION_TIMEOUT_S):
                prepared = await self.prepare(
                    source_profile, design, trace, progress=progress
                )
            await self.store.finish_run(
                run_id,
                user_id=user_id,
                candidates=prepared.candidates,
                criteria=tuple(design.criteria),
                usage=trace.usage(),
                calls=trace.calls,
                status=prepared.status,
                source_predictions=source_predictions(prepared.source, design.criteria),
            )
        if progress:
            progress("Saved the generation run.")
        return run_id

    async def optimize_tasks(
        self,
        *,
        user_id: UserId,
        source_molecule_id: MoleculeId,
        source_profile: MoleculeProfile,
        tasks: Sequence[OptimizationTask],
        targets: Mapping[OptimizationTask, float] | None = None,
        request_id: str | None = None,
        progress: Progress | None = None,
        on_start: Callable[[GenerationRunId], None] | None = None,
    ) -> GenerationRunId:
        """Sample and save distinct analogs for the selected objectives."""
        thresholds = dict(targets or {})
        criteria = {key: OBJECTIVES[key].direction for key in dict.fromkeys(tasks)}
        async with self._record_run(
            user_id=user_id,
            source_molecule_id=source_molecule_id,
            config=SampleConfig(
                samples=self.samples,
                targets={task: thresholds.get(task) for task in tasks},
                model_settings=self.model_settings,
            ),
            request_id=request_id,
            request={"tasks": list(tasks), "targets": thresholds}
            if request_id
            else None,
        ) as (run_id, trace):
            if on_start is not None:
                on_start(run_id)
            if trace is None:
                if progress:
                    progress("Loaded the saved run.")
                return run_id
            if progress:
                progress("Started the generation run.")
            async with asyncio.timeout(OPTIMIZATION_TIMEOUT_S):
                if any(
                    key in criteria
                    and not is_current(
                        key, source_profile.predictions.get(key), method=method
                    )
                    for key, method in self.prediction_methods.items()
                ):
                    (source_profile,) = await self.profile(
                        [source_profile.canonical_smiles]
                    )
                context = OptimizationContext(source_profile, criteria, thresholds)
                if progress:
                    progress("Source profile ready; sampling analogs.")
                candidates = await self._sample_analogs(context, trace)
                if progress:
                    progress(f"Verified {len(candidates)} distinct analogs.")
                profiles = await self.profile(candidates)
            prepared = PreparedGeneration(
                source_profile,
                tuple(PreparedCandidate(profile=profile) for profile in profiles),
            )
            await self.store.finish_run(
                run_id,
                user_id=user_id,
                candidates=prepared.candidates,
                criteria=tuple(criteria),
                usage=trace.usage(),
                calls=trace.calls,
                status=prepared.status,
                source_predictions=source_predictions(source_profile, criteria),
            )
        if progress:
            progress("Saved the generation run.")
        return run_id

    async def _sample_analogs(
        self, context: OptimizationContext, trace: ModelTrace
    ) -> list[str]:
        if self.samples > trace.remaining:
            raise LimitError("The requested samples exceed the model call budget.")
        source_smiles = context.source.canonical_smiles
        cores = await asyncio.to_thread(protected_cores, source_smiles)

        async def sample() -> tuple[int | None, str | None]:
            try:
                response, call_index = await propose_analog(
                    self.optimizer, context, trace
                )
            except Exception as exc:
                logger.warning("Optimizer sample failed (%s).", type(exc).__name__)
                return None, None
            return call_index, response

        async with asyncio.TaskGroup() as group:
            sample_tasks = [group.create_task(sample()) for _ in range(self.samples)]
        call_indices, responses = zip(
            *(task.result() for task in sample_tasks), strict=True
        )
        if all(response is None for response in responses):
            raise ProviderError("All optimizer samples failed.")
        candidates, stats, issues = await asyncio.to_thread(
            _select_analogs, responses, source_smiles, cores
        )
        for call_index, issue in zip(call_indices, issues, strict=True):
            if call_index is not None:
                if issue is None:
                    trace.record_validation(call_index, "accepted")
                else:
                    trace.record_validation(call_index, "rejected", (issue,))
        logger.debug("Optimizer samples: %s", stats)
        if self.on_stats is not None:
            self.on_stats(stats)
        return candidates

    @asynccontextmanager
    async def _record_run(
        self,
        *,
        user_id: UserId,
        source_molecule_id: MoleculeId,
        config: RunConfig,
        request_id: str | None,
        request: dict | None,
    ) -> AsyncIterator[tuple[GenerationRunId, ModelTrace | None]]:
        start = await self.store.start_run(
            user_id=user_id,
            source_molecule_id=source_molecule_id,
            model=self.optimizer.settings.model,
            config=config,
            request_id=request_id,
            request=request,
        )
        if start.replay:
            yield start.id, None
            return
        trace = self.optimizer.new_trace()
        try:
            yield start.id, trace
        except (Exception, asyncio.CancelledError) as exc:
            if isinstance(exc, ConflictError):
                raise
            reason = "Generation could not complete."
            if isinstance(exc, (ParseError, LimitError)):
                reason = str(exc)
            elif isinstance(exc, asyncio.CancelledError):
                reason = "Generation was cancelled."
            elif isinstance(exc, TimeoutError):
                reason = "Generation reached its time limit."
            try:
                await self.store.fail_run(
                    start.id,
                    user_id=user_id,
                    reason=reason,
                    usage=trace.usage(),
                    calls=trace.calls,
                )
            except Exception:
                logger.exception("could not mark generation run %s failed", start.id)
            if isinstance(exc, TimeoutError):
                raise ProviderError(reason) from exc
            raise


def _select_analogs(
    responses: Sequence[str | None], source_smiles: str, cores: Sequence[Mol]
) -> tuple[list[str], GenerationStats, list[ValidationIssue | None]]:
    invalid = source_duplicates = repeated = 0
    candidates: dict[str, None] = {}
    extracted = sum(response is not None for response in responses)
    issues: list[ValidationIssue | None] = []
    for response in responses:
        issues.append(None)
        if response is None:
            continue
        try:
            canonical = verify_analog(response, cores)
        except ParseError:
            invalid += 1
            issues[-1] = ValidationIssue.INVALID_STRUCTURE
            continue
        if canonical == source_smiles:
            source_duplicates += 1
            issues[-1] = ValidationIssue.DUPLICATE_STRUCTURE
        elif canonical in candidates:
            repeated += 1
            issues[-1] = ValidationIssue.DUPLICATE_STRUCTURE
        else:
            candidates[canonical] = None
    stats = GenerationStats(
        requests=len(responses),
        extracted=extracted,
        valid=extracted - invalid,
        invalid=invalid,
        source_duplicates=source_duplicates,
        repeated=repeated,
        unique=len(candidates),
    )
    return list(candidates), stats, issues
