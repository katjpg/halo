from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Annotated, Literal

from pydantic import Field, StringConstraints, model_validator

from halo.core.errors import ProviderError
from halo.llm.prompts.researcher import ANALYST_PROMPT, SEARCH_PROMPT, SYSTEM_PROMPT
from halo.llm.runtime import Agent, AgentOutput, ModelRunner, ModelTrace, run_agent
from halo.llm.validate import (
    OutputValidationError,
    ValidationIssue,
    cited_labels,
    cited_names,
    validate_text,
)
from halo.models.analysis import ResultValue, assay_signature
from halo.models.research import Citation, Finding, Response
from halo.refs import ASSAY_PREFIX, molecule_label
from halo.types import MoleculeId

EvidenceOrigin = Literal["application_record", "external_document", "saved_record"]
SearchTerms = Annotated[
    tuple[
        Annotated[
            str, StringConstraints(strip_whitespace=True, min_length=1, max_length=100)
        ],
        ...,
    ],
    Field(max_length=8),
]


class SearchPlan(AgentOutput):
    query: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1, max_length=1000)
    ]
    change_terms: SearchTerms
    property_terms: SearchTerms

    @model_validator(mode="after")
    def check_phrase_groups(self) -> "SearchPlan":
        if bool(self.change_terms) != bool(self.property_terms):
            raise ValueError("Provide both phrase groups or leave both empty.")
        return self


async def plan_search(
    task: ModelRunner,
    query: str,
    *,
    molecule_id: MoleculeId,
    smiles: str,
    trace: ModelTrace,
) -> SearchPlan:
    """Choose search phrases for the requested structural change and property."""
    plan, call_index = await task.run(
        SearchPlan,
        role="researcher",
        prompt=SEARCH_PROMPT,
        request=query,
        context={"molecule": {"id": molecule_id, "smiles": smiles}},
        constraints={},
        trace=trace,
    )
    trace.record_validation(call_index, "accepted")
    return plan


class ResponseOutput(AgentOutput):
    findings: tuple[Finding, ...]
    limitations: tuple[str, ...]

    def to_response(self) -> Response:
        return Response(self.findings, self.limitations)


@dataclass(frozen=True)
class ResearchContext:
    question: str
    molecules: Mapping[MoleculeId, str | None]
    records: Mapping[str, Citation]
    results: Mapping[str, ResultValue]
    origins: Mapping[str, EvidenceOrigin]


def research_context(
    question: str,
    molecules: Mapping[MoleculeId, str | None],
    evidence: Sequence[Citation],
    results: Sequence[ResultValue],
    *,
    evidence_origin: EvidenceOrigin = "application_record",
    saved_evidence: Sequence[Citation] = (),
) -> ResearchContext:
    records = {item.id: item for item in (*evidence, *saved_evidence)}
    by_result_id = {item.id: item for item in results}
    if len(records) != len(evidence) + len(saved_evidence) or len(by_result_id) != len(
        results
    ):
        raise ProviderError(
            "Evidence and result IDs must be unique within each collection."
        )
    return ResearchContext(
        question,
        molecules,
        records,
        by_result_id,
        {item.id: evidence_origin for item in evidence}
        | {item.id: "saved_record" for item in saved_evidence},
    )


def build_context(context: ResearchContext) -> dict:
    hidden_fields = {"source_ref"}
    if not context.records and len(context.results) == 1:
        (ref,) = context.results
        if ref.startswith(ASSAY_PREFIX):
            hidden_fields.add("value")
    return {
        "molecules": [
            {"id": mid, "label": molecule_label(mid), "smiles": smiles}
            for mid, smiles in context.molecules.items()
        ],
        "evidence": [
            {**asdict(item), "origin": context.origins[ref]}
            for ref, item in context.records.items()
        ],
        "results": [
            {
                **{
                    key: value
                    for key, value in asdict(item).items()
                    if value is not None and key not in hidden_fields
                },
                "origin": (
                    "saved_record"
                    if item.id.startswith(ASSAY_PREFIX)
                    else "external_record"
                    if item.source_type == "experimental"
                    else "application_record"
                ),
            }
            for item in context.results.values()
        ],
    }


def verify_response(
    response: Response,
    *,
    molecules: Mapping[MoleculeId, str | None],
    records: Mapping[str, Citation],
    results: Mapping[str, ResultValue],
    question: str,
) -> Response:
    """Validate cited claims against the provided molecules and evidence."""
    requested_names = cited_names(question)
    record_names = {
        ref: cited_names(record.text)
        | cited_labels(record.text)
        | set(record.identifiers)
        for ref, record in records.items()
    }
    result_names = {}
    for ref, result in results.items():
        conditions = result.conditions or {}
        labels = (
            result.endpoint,
            result.assay_description,
            result.target,
            conditions.get("assay_description"),
            conditions.get("target"),
        )
        result_names[ref] = cited_names(
            " ".join(label for label in labels if isinstance(label, str))
        )
    for finding in response.findings:
        molecule_ids = set(finding.molecule_ids)
        if (
            molecule_ids - molecules.keys()
            or set(finding.evidence_ids) - records.keys()
        ):
            raise OutputValidationError(ValidationIssue.OUTSIDE_CONTEXT)
        cited_results = []
        for ref in finding.result_ids:
            result = results.get(ref)
            if result is None or result.molecule_id not in molecule_ids:
                raise OutputValidationError(ValidationIssue.RESULT_MOLECULE)
            cited_results.append(result)
        if any(result.id.startswith(ASSAY_PREFIX) for result in cited_results):
            if {result.molecule_id for result in cited_results} != molecule_ids:
                raise OutputValidationError(ValidationIssue.ASSAY_COVERAGE)
            if len(cited_results) > 1:
                signatures = {assay_signature(result) for result in cited_results}
                if None in signatures or len(signatures) != 1:
                    raise OutputValidationError(ValidationIssue.INCOMPARABLE_ASSAYS)
        if not (finding.evidence_ids or finding.result_ids) and not any(
            molecules[mid] for mid in molecule_ids
        ):
            raise OutputValidationError(ValidationIssue.MISSING_REFERENCE)
        names = requested_names | set(map(molecule_label, molecule_ids))
        for ref in finding.result_ids:
            names.update(result_names[ref])
        for ref in finding.evidence_ids:
            names.update(record_names[ref])
        names.update(
            molecules[mid] for mid in molecule_ids if molecules[mid] is not None
        )
        validate_text(finding.text, names=names)
    limitation_names = requested_names | set(map(molecule_label, molecules))
    limitation_names.update(*record_names.values(), *result_names.values())
    for limitation in response.limitations:
        validate_text(limitation, names=limitation_names)
    if not response.findings and not response.limitations:
        raise OutputValidationError(ValidationIssue.MISSING_LIMITATION)
    return response


def verify_evidence(output: ResponseOutput, context: ResearchContext) -> None:
    verify_response(
        output.to_response(),
        molecules=context.molecules,
        records=context.records,
        results=context.results,
        question=context.question,
    )


RESEARCHER = Agent(
    role="researcher",
    prompt=SYSTEM_PROMPT,
    schema=ResponseOutput,
    build_context=build_context,
    verify=verify_evidence,
    max_attempts=2,
)
ANALYST = Agent(
    role="analyst",
    prompt=ANALYST_PROMPT,
    schema=ResponseOutput,
    build_context=build_context,
    verify=verify_evidence,
    max_attempts=2,
)


async def respond(
    task: ModelRunner,
    question: str,
    *,
    molecules: Mapping[MoleculeId, str | None],
    evidence: Sequence[Citation],
    results: Sequence[ResultValue] = (),
    trace: ModelTrace,
    role: Literal["researcher", "analyst"] = "researcher",
    evidence_origin: EvidenceOrigin = "application_record",
) -> Response:
    """Synthesize and verify claims from selected records."""
    context = research_context(
        question, molecules, evidence, results, evidence_origin=evidence_origin
    )
    output, _ = await run_agent(
        ANALYST if role == "analyst" else RESEARCHER,
        task,
        context,
        request=question,
        constraints={},
        trace=trace,
    )
    return output.to_response()
