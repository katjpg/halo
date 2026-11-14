from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite
from threading import Lock
from typing import TYPE_CHECKING, Self, TypeVar
from uuid import UUID, uuid4

import httpx
from pydantic import BaseModel, TypeAdapter, ValidationError

from halo.core.errors import ParseError
from halo.core.format import format_table
from halo.models.analysis import ResultValue
from halo.models.research import EvidenceScope, Response, SourceSearch
from halo.models.session import Project as ProjectRecord
from halo.models.session import Session, TurnResult
from halo.refs import molecule_label
from halo.schemas.generation import (
    CandidateResponse,
    GenerationRunResponse,
    OptimizationRequest,
)
from halo.schemas.molecule import MoleculeRequest, ProfileResponse
from halo.schemas.session import (
    ProjectRequest,
    TurnRequest,
    TurnResponse,
    TurnSummaryResponse,
)
from halo.types import (
    DEFAULT_BATCH_SIZE,
    BatchSize,
    Criteria,
    DecisionId,
    DesignCriteria,
    MoleculeId,
    Progress,
    ProjectId,
    SessionId,
)

if TYPE_CHECKING:
    from halo.clients.review import RunReview

T = TypeVar("T")
ONGOING_PREFERENCE = re.compile(
    r"\b(?:always|from now on|going forward|(?:each|every|all future) "
    r"(?:design |generation )?(?:rounds?|runs?|batches))\b",
    re.IGNORECASE,
)
NEXT_PREFERENCE = re.compile(
    r"\bnext (?:design |generation )?(?:round|run|batch)\b", re.IGNORECASE
)


class GenerationRun(GenerationRunResponse):
    """A saved run with a printable summary of its candidates."""

    def __str__(self) -> str:
        n_candidates = len(self.candidates)
        candidate_summary = (
            f"{n_candidates} of {self.n}" if self.n is not None else str(n_candidates)
        )
        status = self.status.replace("_", " ")
        lines = [
            f"{molecule_label(MoleculeId(self.source_molecule_id))} · {candidate_summary} analogs · {status}"
        ]
        if self.failure_reason:
            lines.append(self.failure_reason)
        if not self.candidates:
            return "\n".join(lines)

        fields = sorted(self.criteria or self.objectives)
        source_predictions = self.source_predictions or {}
        columns = ["Analog", "SMILES"]
        for field in fields:
            metric = source_predictions.get(field)
            columns.extend(
                (
                    f"Δ {field} ({metric.unit})" if metric else f"Δ {field}",
                    f"{field} goal",
                )
            )
        rows = []
        for candidate in self.candidates:
            changes = candidate.prediction_changes or {}
            row = [molecule_label(MoleculeId(candidate.molecule_id)), candidate.smiles]
            for field in fields:
                delta = changes.get(field)
                row.append(f"{delta:+.3g}" if delta is not None else "—")
                row.append(candidate.goal_outcomes.get(field, "—"))
            rows.append(row)

        lines.append(format_table(columns, rows))
        return "\n".join(lines)


@dataclass(frozen=True)
class ResearchResult(Response):
    """Cited findings, their sources, and the saved research turn."""

    sources: SourceSearch
    turn: TurnResponse

    def __str__(self) -> str:
        lines = response_lines(self, self.sources)
        if self.turn.result and self.turn.result.values:
            lines.append(format_results(self.turn.result.values))
        return "\n".join(lines)


def response_lines(
    response: Response, sources: SourceSearch | None = None
) -> list[str]:
    lines = [
        " ".join(
            (
                finding.text,
                *(f"[{ref}]" for ref in (*finding.evidence_ids, *finding.result_ids)),
            )
        )
        for finding in response.findings
    ]
    lines.extend(response.limitations)
    if sources is not None:
        groups: tuple[tuple[EvidenceScope, str], ...] = (
            ("exact", "Exact evidence"),
            ("analog", "Analog evidence"),
            ("topic", "Topic evidence"),
        )
        for scope, heading in groups:
            passages = sources.for_scope(scope)
            if not passages and (scope == "topic" or not sources.passages):
                continue
            lines.extend(("", heading))
            if not passages:
                lines.append("No sources retrieved.")
            for passage in passages:
                lines.append(f"[{passage.id}] {passage.title} ({passage.url})")
                if passage.license_url:
                    metadata = [
                        f"PMCID: {passage.pmcid}",
                        f"License: {passage.license_url}",
                        f"Location: {passage.locator}",
                    ]
                    if passage.doi:
                        metadata.append(f"DOI: {passage.doi}")
                    if passage.year is not None:
                        metadata.append(f"Year: {passage.year}")
                    lines.append(" · ".join(metadata))
                link = passage.link
                if link.source_ref and link.matched_smiles:
                    match = f"{link.source_ref} · {link.matched_smiles}"
                    if link.morgan_tanimoto is not None:
                        match += f" · Morgan Tanimoto {link.morgan_tanimoto:.3f}"
                    lines.append(match)
                lines.append(passage.text)
        if sources.unavailable:
            lines.append(f"Unavailable sources: {', '.join(sources.unavailable)}")
        if sources.exclusions:
            lines.append("Excluded sources:")
            lines.extend(
                f"{item.publication_id}: {item.reason}" for item in sources.exclusions
            )
    return lines


def format_results(values: Sequence[ResultValue]) -> str:
    return format_table(
        ("Result", "Molecule", "Property", "Value", "Source", "Method"),
        [
            (
                row.id,
                molecule_label(row.molecule_id),
                row.endpoint,
                f"{row.relation} {row.value:.4g} {row.unit}",
                row.source_type,
                row.method,
            )
            for row in values
        ],
    )


@dataclass(frozen=True)
class Reply:
    """A printable response with the saved turn and any generation run."""

    turn: TurnResponse
    run: GenerationRun | None = None

    @property
    def result(self) -> TurnResult | None:
        return self.turn.result

    def __str__(self) -> str:
        result = self.result
        if result is None:
            return f"Turn {self.turn.status}."
        lines = [result.message] if result.message else []
        if self.run is not None:
            lines = [str(self.run)]
        if result.response is not None:
            lines.extend(
                response_lines(
                    result.response,
                    SourceSearch(
                        result.passages, result.unavailable, result.source_exclusions
                    ),
                )
            )
        elif result.plot is not None:
            plot = result.plot
            points = [
                point for point in plot.points if point.molecule_id in plot.visible_ids
            ]
            lines.append(
                format_table(
                    (
                        "Molecule",
                        f"{plot.spec.x} ({plot.units[0]})",
                        f"{plot.spec.y} ({plot.units[1]})",
                    ),
                    [
                        (molecule_label(p.molecule_id), f"{p.x:.4g}", f"{p.y:.4g}")
                        for p in points
                    ],
                )
            )
        if result.values and result.plot is None:
            lines.append(format_results(result.values))
            if result.unavailable and result.response is None:
                lines.append(f"Unavailable: {', '.join(result.unavailable)}")
        if result.clusters is not None and result.response is None:
            lines.append(
                format_table(
                    ("Representative", "Members"),
                    [
                        (
                            molecule_label(group.representative_id),
                            ", ".join(map(molecule_label, group.molecule_ids)),
                        )
                        for group in result.clusters.groups
                    ],
                )
            )
        return "\n".join(lines) or f"Turn {self.turn.status}."


class History(BaseModel):
    """Saved turns in one session, in conversation order."""

    session_id: SessionId | None = None
    turns: tuple[TurnSummaryResponse, ...] = ()

    def __str__(self) -> str:
        exchanges = []
        for turn in self.turns:
            action = turn.request.get("action") or {}
            message = (
                turn.request.get("message")
                or action.get("text")
                or action.get("question")
                or action.get("instructions")
            )
            if not message:
                message = action.get("type", "Action").replace("_", " ").capitalize()
            result = turn.result
            lines = []
            if result is not None:
                if result.message or result.error:
                    lines.append(result.message or result.error)
                if result.response is not None:
                    lines.extend(response_lines(result.response))
            response = "\n".join(lines) or f"Turn {turn.status}."
            exchanges.append(f"[USER] {message}\n[HALO] {response}")
        return "\n\n".join(exchanges) or "No saved turns."


class HALOChemError(RuntimeError):
    """A failed request or saved turn, with its code and available IDs."""

    def __init__(
        self,
        message: str,
        *,
        code: str,
        status_code: int | None = None,
        turn_id: int | None = None,
        run_id: str | None = None,
        request_id: UUID | None = None,
        location: str | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.status_code = status_code
        self.turn_id = turn_id
        self.run_id = run_id
        self.request_id = request_id
        self.location = location


class HALOChem:
    """A synchronous client for a running HALO service.

    Use a user access token and close the client when finished. The timeout
    applies to each network phase.
    """

    def __init__(
        self, *, base_url: str, access_token: str, timeout_s: float = 180.0
    ) -> None:
        url = httpx.URL(base_url)
        if url.scheme not in ("http", "https") or not url.host:
            raise ParseError("Provide the HALO service URL, including http or https.")
        if url.userinfo or url.query or url.fragment:
            raise ParseError(
                "Provide a service URL without credentials, query, or fragment."
            )
        if not access_token.strip():
            raise ParseError("Provide a HALO user access token.")
        if not isfinite(timeout_s) or timeout_s <= 0:
            raise ParseError("timeout_s must be positive and finite.")
        self._http = httpx.Client(
            base_url=url,
            headers={"Authorization": f"Bearer {access_token.strip()}"},
            timeout=timeout_s,
        )
        self.projects = Projects(self)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def close(self) -> None:
        """Release the client's network resources."""
        self._http.close()

    def optimize(
        self,
        source_smiles: str,
        *,
        criteria: Criteria,
        n: int = DEFAULT_BATCH_SIZE,
        progress: bool | Progress = False,
        request_id: str | UUID | None = None,
    ) -> GenerationRun:
        """Generate analogs against a source without opening a project."""
        key = UUID(str(request_id)) if request_id is not None else uuid4()
        criteria = TypeAdapter(DesignCriteria).validate_python(criteria)
        n = TypeAdapter(BatchSize).validate_python(n)
        profile = self._save_molecule(source_smiles)
        return self._request(
            "POST",
            "/api/generation-runs",
            GenerationRun,
            body=OptimizationRequest(
                type="design",
                source_molecule_id=MoleculeId(profile.molecule_id),
                criteria=criteria,
                n=n,
                request_id=key,
            ),
            progress=progress,
        )

    def get_run(self, run_id: str | UUID) -> GenerationRun:
        """Read a saved generation run owned by the signed-in user."""
        return self._request(
            "GET", f"/api/generation-runs/{UUID(str(run_id))}", GenerationRun
        )

    def _save_molecule(self, smiles: str) -> ProfileResponse:
        return self._request(
            "POST",
            "/api/molecules",
            ProfileResponse,
            body=MoleculeRequest(smiles=smiles),
        )

    def _request(
        self,
        method: str,
        path: str,
        response_type: type[T],
        *,
        body: BaseModel | None = None,
        progress: bool | Progress = False,
    ) -> T:
        adapter = TypeAdapter(response_type)
        headers = {"Accept": "application/x-ndjson"} if progress else None
        payload = body.model_dump(mode="json", exclude_none=True) if body else None
        started: dict[str, object] = {}
        status_code = None
        try:
            with self._http.stream(
                method, path, json=payload, headers=headers
            ) as response:
                status_code = response.status_code
                if not response.is_success:
                    response.read()
                    try:
                        error = response.json()
                    except ValueError:
                        error = None
                    raise service_error(error, response.status_code)
                if (
                    headers
                    and response.headers.get("content-type", "").split(";")[0]
                    == headers["Accept"]
                ):
                    for line in response.iter_lines():
                        if not line:
                            continue
                        event = json.loads(line)
                        if not isinstance(event, dict):
                            break
                        if isinstance(record := event.get("started"), dict):
                            started = record
                        elif isinstance(step := event.get("step"), str):
                            if progress is True:
                                print(step, flush=True)
                            elif callable(progress):
                                progress(step)
                        elif "result" in event:
                            return adapter.validate_python(event["result"])
                        elif isinstance(error := event.get("error"), dict):
                            code = error.get("status_code")
                            if isinstance(code, int) and 400 <= code <= 599:
                                raise service_error(error, code)
                            break
                        else:
                            break
                    raise HALOChemError(
                        "The progress stream ended without a result.",
                        code="invalid_response",
                    )
                return adapter.validate_json(response.read())
        except (
            HALOChemError,
            httpx.RequestError,
            ValidationError,
            json.JSONDecodeError,
        ) as exc:
            if isinstance(exc, HALOChemError):
                error = exc
            elif isinstance(exc, httpx.TimeoutException):
                error = HALOChemError("HALO did not respond in time.", code="timeout")
            elif isinstance(exc, httpx.RequestError):
                error = HALOChemError(
                    "Could not reach the HALO service.", code="connection_error"
                )
            else:
                error = HALOChemError(
                    f"HALO returned an invalid response for {method} {path}.",
                    code="invalid_response",
                    status_code=status_code,
                )
            error.request_id = getattr(body, "request_id", None)
            turn_id, run_id, location = (
                started.get("turn_id"),
                started.get("run_id"),
                started.get("location"),
            )
            if type(turn_id) is int:
                error.turn_id = turn_id
            if isinstance(run_id, str):
                error.run_id = run_id
            if isinstance(location, str):
                error.location = location
            if error is exc:
                raise
            raise error from exc


def service_error(payload: object, status_code: int) -> HALOChemError:
    code, message = "http_error", f"HALO returned HTTP {status_code}."
    if isinstance(payload, dict):
        if isinstance(payload.get("code"), str):
            code = payload["code"]
        if isinstance(payload.get("message"), str):
            message = payload["message"]
    return HALOChemError(message, code=code, status_code=status_code)


class Projects:
    """Project creation and lookup for a HALO client."""

    def __init__(self, client: HALOChem) -> None:
        self._client = client

    def create(self, *, source_smiles: str) -> Project:
        """Save a source molecule and open its project."""
        profile = self._client._save_molecule(source_smiles)
        project = self._client._request(
            "POST",
            "/api/projects",
            ProjectRecord,
            body=ProjectRequest(source_molecule_id=MoleculeId(profile.molecule_id)),
        )
        return Project(self._client, project)

    def list(self) -> list[ProjectRecord]:
        """List the signed-in user's projects, newest first."""
        return self._client._request("GET", "/api/projects", list[ProjectRecord])

    def get(self, project_id: str | UUID, *, resume: bool = False) -> Project:
        """Open a saved project. Resume its latest session if requested."""
        record = self._client._request(
            "GET", f"/api/projects/{UUID(str(project_id))}", ProjectRecord
        )
        project = Project(self._client, record)
        if resume:
            project._session_id = project._latest_session()
        return project


class Project:
    """A saved project that starts a session on its first turn."""

    def __init__(self, client: HALOChem, project: ProjectRecord) -> None:
        self._client = client
        self._project = project
        self._session_id: SessionId | None = None
        self._session_lock = Lock()

    @property
    def id(self) -> ProjectId:
        return self._project.id

    @property
    def source_molecule_id(self) -> MoleculeId:
        return self._project.lead_id

    @property
    def session_id(self) -> SessionId | None:
        return self._session_id

    def generate_analogs(
        self,
        *,
        criteria: Criteria,
        n: int = DEFAULT_BATCH_SIZE,
        instructions: str | None = None,
        core_smiles: str | None = None,
        progress: bool | Progress = False,
        request_id: str | UUID | None = None,
    ) -> GenerationRun:
        """Generate analogs from the session's active parent.

        Args:
            criteria: Desired property changes from the parent.
            n: Number of analogs to request.
            instructions: Design guidance for this run.
            core_smiles: Mapped core to preserve.
            progress: Print live steps or pass them to a callback.
            request_id: Reuse this ID to recover the same request.

        Returns:
            The saved run. It can contain fewer verified candidates than requested.
        """
        action: dict[str, object] = {
            "type": "design",
            "criteria": criteria,
            "n": n,
            "core_smiles": core_smiles,
        }
        if instructions is not None:
            action["instructions"] = instructions
        turn = self._turn(
            TurnRequest.model_validate({"action": action}),
            progress=progress,
            request_id=request_id,
        )
        run = self._turn_run(turn)
        if run is None:
            raise HALOChemError(
                "The design turn returned no generation run.",
                code="invalid_response",
                turn_id=turn.id,
            )
        return run

    def remember(
        self, instruction: str, *, request_id: str | UUID | None = None
    ) -> DecisionId:
        """Save a design instruction for the next batch that produces analogs.

        Phrases such as "in every round" or "from now on" keep it active.
        """
        ongoing = ONGOING_PREFERENCE.search(instruction) is not None
        if ongoing and NEXT_PREFERENCE.search(instruction):
            raise ParseError(
                "Choose the next round or every round for this instruction."
            )
        request = TurnRequest.model_validate(
            {
                "action": {
                    "type": "set_preference",
                    "text": instruction,
                    "scope": "ongoing" if ongoing else "next_run",
                }
            }
        )
        turn = self._turn(request, request_id=request_id)
        if turn.result is None or turn.result.decision_id is None:
            raise HALOChemError(
                "The preference turn returned no decision.",
                code="invalid_response",
                turn_id=turn.id,
            )
        return turn.result.decision_id

    def set_parent(
        self,
        candidate: CandidateResponse,
        *,
        run: GenerationRunResponse,
        request_id: str | UUID | None = None,
    ) -> None:
        """Make a candidate from this run the session's parent and sole selection."""
        if run.project_id != self.id:
            raise ParseError("Choose a generation run from this project.")
        molecule_id = candidate.molecule_id
        if all(row.molecule_id != molecule_id for row in run.candidates):
            raise ParseError("Choose a candidate from the given run.")
        request = TurnRequest.model_validate(
            {
                "action": {
                    "type": "select",
                    "molecule_ids": [molecule_id],
                    "parent": {
                        "molecule_id": molecule_id,
                        "generation_run_id": run.id,
                    },
                }
            }
        )
        self._turn(request, request_id=request_id)

    def research(
        self,
        query: str,
        *,
        molecule_id: int | None = None,
        progress: bool | Progress = False,
        request_id: str | UUID | None = None,
    ) -> ResearchResult:
        """Find papers and patents for a project molecule.

        Args:
            query: The scientific question or search terms.
            molecule_id: Project molecule ID. Defaults to the lead.
            progress: Print live steps or pass them to a callback.
            request_id: Reuse this ID to recover the same request.

        Returns:
            Findings, source passages by relationship, and the saved turn.
        """
        request = TurnRequest.model_validate(
            {
                "action": {
                    "type": "research",
                    "molecule_id": self.source_molecule_id
                    if molecule_id is None
                    else molecule_id,
                    "question": query,
                }
            }
        )
        turn = self._turn(request, progress=progress, request_id=request_id)
        result = turn.result
        if result is None or result.task != "retrieve" or result.response is None:
            raise HALOChemError(
                "The research turn returned no response.",
                code="invalid_response",
                turn_id=turn.id,
            )
        return ResearchResult(
            findings=result.response.findings,
            limitations=result.response.limitations,
            sources=SourceSearch(
                result.passages, result.unavailable, result.source_exclusions
            ),
            turn=turn,
        )

    def ask(
        self,
        message: str,
        *,
        progress: bool | Progress = False,
        request_id: str | UUID | None = None,
    ) -> Reply:
        """Route a message and return a reply with its saved turn."""
        turn = self._turn(
            TurnRequest(message=message), progress=progress, request_id=request_id
        )
        return Reply(turn, self._turn_run(turn))

    def get_turn(
        self, turn_id: int, *, session_id: str | UUID | None = None
    ) -> TurnResponse:
        """Read a saved turn from the specified, active, or latest session."""
        selected = self._read_session_id(session_id)
        if selected is None:
            raise ParseError("Choose a session from this project.")
        return self._client._request(
            "GET", f"/api/sessions/{selected}/turns/{turn_id}", TurnResponse
        )

    def history(self, session_id: str | UUID | None = None) -> History:
        """Read a session's saved turns. Default to the active or latest session."""
        selected = self._read_session_id(session_id)
        if selected is None:
            return History()
        turns = self._client._request(
            "GET", f"/api/sessions/{selected}/turns", list[TurnSummaryResponse]
        )
        return History(session_id=selected, turns=tuple(turns))

    def _read_session_id(self, session_id: str | UUID | None) -> SessionId | None:
        if session_id is None:
            return self._session_id or self._latest_session()
        session = self._client._request(
            "GET", f"/api/sessions/{UUID(str(session_id))}", Session
        )
        if session.project_id != self.id:
            raise ParseError("Choose a session from this project.")
        return session.id

    def _latest_session(self) -> SessionId | None:
        sessions = self._client._request(
            "GET", f"/api/projects/{self.id}/sessions", list[Session]
        )
        return sessions[0].id if sessions else None

    def runs(self) -> list[GenerationRun]:
        """Retrieve this project's saved generation runs in creation order."""
        return self._client._request(
            "GET",
            f"/api/projects/{self.id}/generation-runs",
            list[GenerationRun],
        )

    def get_run(self, run_id: str | UUID) -> GenerationRun:
        """Retrieve a saved generation run in this project."""
        run = self._client.get_run(run_id)
        if run.project_id != self.id:
            raise ParseError("Choose a generation run from this project.")
        return run

    def inspect_run(
        self,
        run: GenerationRun | str | UUID,
        *,
        min_similarity: float = 0.70,
        min_core_coverage: float = 0.80,
    ) -> RunReview:
        """Review saved analogs, predicted changes, edits, and structural clusters."""
        from halo.clients.review import inspect_run

        saved = run if isinstance(run, GenerationRun) else self.get_run(run)
        if saved.project_id != self.id:
            raise ParseError("Choose a generation run from this project.")
        source = self._client._request(
            "GET", f"/api/molecules/{saved.source_molecule_id}", ProfileResponse
        )
        return inspect_run(
            saved,
            source.smiles,
            min_similarity=min_similarity,
            min_core_coverage=min_core_coverage,
        )

    def _turn_run(self, turn: TurnResponse) -> GenerationRun | None:
        run_id = turn.result.generation_run_id if turn.result else None
        if run_id is None:
            return None
        try:
            return self.get_run(run_id)
        except HALOChemError as exc:
            exc.request_id = turn.request_id
            exc.turn_id = turn.id
            exc.run_id = run_id
            exc.location = f"/api/generation-runs/{run_id}"
            raise

    def _turn(
        self,
        request: TurnRequest,
        *,
        progress: bool | Progress = False,
        request_id: str | UUID | None = None,
    ) -> TurnResponse:
        key = (
            UUID(str(request_id))
            if request_id is not None
            else request.request_id or uuid4()
        )
        request = request.model_copy(update={"request_id": key})
        with self._session_lock:
            if self._session_id is None:
                session = self._client._request(
                    "POST", f"/api/projects/{self.id}/sessions", Session
                )
                self._session_id = session.id
            session_id = self._session_id
        location = f"/api/sessions/{session_id}/turns"
        turn = self._client._request(
            "POST",
            location,
            TurnResponse,
            body=request,
            progress=progress,
        )
        if turn.status != "completed":
            error = turn.result.error if turn.result else None
            code = turn.result.error_code if turn.result else None
            raise HALOChemError(
                error
                or (
                    "The turn is still running."
                    if turn.status == "running"
                    else f"The turn {turn.status}."
                ),
                code=code or turn.status,
                turn_id=turn.id,
                request_id=key,
                location=f"{location}/{turn.id}",
            )
        return turn
