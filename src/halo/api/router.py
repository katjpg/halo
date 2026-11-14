import asyncio
import json
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Annotated, TypeVar
from uuid import UUID

from fastapi import APIRouter, Depends, Path, Query, Request, Response, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from halo.api.dependencies import (
    CallerDep,
    GenerationRunnerDep,
    GenerationStoreDep,
    ModelRatesDep,
    ResearchWorkflowDep,
    SessionWriterDep,
    TurnRunnerDep,
    get_caller,
)
from halo.api.errors import ErrorResponse, error_body
from halo.llm.cost import estimate_cost
from halo.models.analysis import AssayResult
from halo.models.session import Project, Session
from halo.schemas.generation import (
    GenerationRunRequest,
    GenerationRunResponse,
    OptimizationRequest,
    run_response,
)
from halo.schemas.molecule import MoleculeRequest, ProfileResponse, profile_response
from halo.schemas.research import AssayResultRequest, ResearchRequest, ResearchResponse
from halo.schemas.session import (
    ProjectRequest,
    TurnRequest,
    TurnResponse,
    TurnSummaryResponse,
)
from halo.types import (
    MAX_RECORD_ID,
    AssayEndpoint,
    AssayName,
    GenerationRunId,
    MoleculeId,
    Progress,
    ProjectId,
    SessionId,
    TurnId,
)

router = APIRouter()
RecordId = TypeVar("RecordId", TurnId, GenerationRunId)
api = APIRouter(
    prefix="/api",
    dependencies=[Depends(get_caller)],
    responses={
        401: {"model": ErrorResponse, "description": "Missing or invalid access token"},
        404: {"model": ErrorResponse, "description": "Record not found"},
        502: {"model": ErrorResponse, "description": "Upstream service failed"},
        422: {
            "model": ErrorResponse,
            "description": "Invalid request or local execution limit",
        },
        500: {"model": ErrorResponse, "description": "Unexpected server error"},
    },
)


def accepts_ndjson(accept: str) -> bool:
    for item in accept.split(","):
        media_type, *parameters = item.split(";")
        if media_type.strip().casefold() != "application/x-ndjson":
            continue
        quality = 1.0
        for parameter in parameters:
            key, _, value = parameter.partition("=")
            if key.strip().casefold() == "q":
                try:
                    quality = float(value)
                except ValueError:
                    quality = 0.0
        if 0 < quality <= 1:
            return True
    return False


def _ndjson(key: str, value: object) -> str:
    return json.dumps({key: value}, ensure_ascii=False, allow_nan=False) + "\n"


def progress_response(
    request: Request,
    execute: Callable[[Progress, Callable[[RecordId], None]], Awaitable[BaseModel]],
    *,
    id_key: str,
    location: Callable[[RecordId], str],
    status_code: int,
) -> StreamingResponse:
    """Send the saved record location, progress, and the result or error."""

    async def events() -> AsyncIterator[str]:
        lines: asyncio.Queue[str | None] = asyncio.Queue()

        def started(record_id: RecordId) -> None:
            lines.put_nowait(
                _ndjson("started", {id_key: record_id, "location": location(record_id)})
            )

        def progress(message: str) -> None:
            lines.put_nowait(_ndjson("step", message))

        async def run() -> None:
            try:
                result = await execute(progress, started)
                lines.put_nowait(
                    '{"result":' + result.model_dump_json(by_alias=True) + "}\n"
                )
            except Exception as exc:
                status, body = error_body(request, exc)
                lines.put_nowait(
                    _ndjson("error", body.model_dump() | {"status_code": status})
                )

        workers: set[asyncio.Task[None]] = request.app.state.progress_tasks

        def finished(worker: asyncio.Task[None]) -> None:
            workers.discard(worker)
            if worker.cancelled():
                lines.put_nowait(
                    _ndjson(
                        "error",
                        {
                            "code": "cancelled",
                            "message": "The operation was cancelled.",
                            "status_code": 503,
                        },
                    )
                )
            lines.put_nowait(None)

        worker = asyncio.create_task(run())
        workers.add(worker)
        worker.add_done_callback(finished)
        while (line := await lines.get()) is not None:
            yield line

    return StreamingResponse(
        events(),
        status_code=status_code,
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
    )


@router.get("/health", include_in_schema=False)
async def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get(
    "/ready",
    include_in_schema=False,
    responses={503: {"description": "Service not ready"}},
)
async def readiness(request: Request, response: Response) -> dict[str, str]:
    state = request.app.state
    if getattr(state, "ready", False) and await state.session_writer.is_ready():
        return {"status": "ready"}
    response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return {"status": "not_ready"}


@api.post("/molecules", status_code=status.HTTP_201_CREATED, tags=["molecules"])
async def create_molecule(
    body: MoleculeRequest,
    caller: CallerDep,
    runner: GenerationRunnerDep,
) -> ProfileResponse:
    molecule_id, _ = await runner.save_profile(caller.user_id, body.smiles)
    profile = await caller.reader.molecule_profile(molecule_id)
    return profile_response(molecule_id, profile)


@api.get("/molecules/{molecule_id}", tags=["molecules"])
async def get_molecule(
    molecule_id: Annotated[MoleculeId, Path(gt=0, le=MAX_RECORD_ID)], caller: CallerDep
) -> ProfileResponse:
    profile = await caller.reader.molecule_profile(molecule_id)
    return profile_response(molecule_id, profile)


@api.post(
    "/generation-runs",
    status_code=status.HTTP_201_CREATED,
    response_model=GenerationRunResponse,
    tags=["generation"],
    responses={
        201: {
            "headers": {"Location": {"schema": {"type": "string"}}},
            "content": {"application/x-ndjson": {"schema": {"type": "string"}}},
        },
        409: {
            "model": ErrorResponse,
            "description": "Request ID belongs to a different run",
        },
    },
)
async def create_generation_run(
    body: GenerationRunRequest,
    request: Request,
    response: Response,
    caller: CallerDep,
    runner: GenerationRunnerDep,
    rates: ModelRatesDep,
) -> GenerationRunResponse | StreamingResponse:
    source_id = body.source_molecule_id
    source_profile = await caller.reader.molecule_profile(source_id)
    request_id = str(body.request_id) if body.request_id else None

    def location(run_id: GenerationRunId) -> str:
        return request.url_for("get_generation_run", run_id=run_id).path

    async def execute(
        progress: Progress | None = None,
        on_start: Callable[[GenerationRunId], None] | None = None,
    ) -> GenerationRunResponse:
        if isinstance(body, OptimizationRequest):
            run_id = await runner.optimize(
                user_id=caller.user_id,
                source_molecule_id=source_id,
                source_profile=source_profile,
                criteria=body.criteria,
                n=body.n,
                request_id=request_id,
                progress=progress,
                on_start=on_start,
            )
        else:
            run_id = await runner.optimize_tasks(
                user_id=caller.user_id,
                source_molecule_id=source_id,
                source_profile=source_profile,
                tasks=body.tasks,
                targets=body.targets,
                request_id=request_id,
                progress=progress,
                on_start=on_start,
            )
        run = await caller.reader.generation_run(run_id)
        response.headers["Location"] = location(run_id)
        cost = estimate_cost(run.model, run.usage, rates)
        return run_response(run, cost)

    if accepts_ndjson(request.headers.get("accept", "")):
        return progress_response(
            request,
            execute,
            id_key="run_id",
            location=location,
            status_code=status.HTTP_201_CREATED,
        )
    return await execute()


@api.get("/generation-runs/{run_id}", tags=["generation"])
async def get_generation_run(
    run_id: UUID, caller: CallerDep, store: GenerationStoreDep, rates: ModelRatesDep
) -> GenerationRunResponse:
    run = await caller.reader.generation_run(GenerationRunId(str(run_id)))
    if (
        run.status == "running"
        and run.project_id is None
        and await store.expire_run(caller.user_id, run.id)
    ):
        run = await caller.reader.generation_run(run.id)
    cost = estimate_cost(run.model, run.usage, rates)
    return run_response(run, cost)


@api.post("/research", tags=["research"])
async def research(
    body: ResearchRequest,
    caller: CallerDep,
    workflow: ResearchWorkflowDep,
) -> ResearchResponse:
    profile = await caller.reader.molecule_profile(body.molecule_id)
    sources, response = await workflow.respond(
        molecule_id=body.molecule_id,
        smiles=profile.canonical_smiles,
        question=body.question,
    )
    return ResearchResponse(sources=sources, response=response)


@api.post("/projects", status_code=status.HTTP_201_CREATED, tags=["projects"])
async def create_project(
    body: ProjectRequest,
    caller: CallerDep,
    writer: SessionWriterDep,
) -> Project:
    await caller.reader.molecule_profile(body.source_molecule_id)
    return await writer.create_project(caller.user_id, body.source_molecule_id)


@api.get("/projects", tags=["projects"])
async def list_projects(caller: CallerDep) -> list[Project]:
    """List the caller's projects, newest first."""
    return await caller.reader.list_projects(caller.user_id)


@api.get("/projects/{project_id}", tags=["projects"])
async def get_project(project_id: UUID, caller: CallerDep) -> Project:
    return await caller.reader.project(caller.user_id, ProjectId(str(project_id)))


@api.post(
    "/projects/{project_id}/assay-results",
    status_code=status.HTTP_201_CREATED,
    tags=["projects"],
)
async def record_assay_result(
    project_id: UUID,
    body: AssayResultRequest,
    caller: CallerDep,
    writer: SessionWriterDep,
) -> AssayResult:
    """Replay the saved result when its source reference and contents match."""
    return await writer.record_assay_result(
        caller.user_id,
        ProjectId(str(project_id)),
        body.model_dump(mode="json", by_alias=True),
    )


@api.get("/projects/{project_id}/assay-results", tags=["projects"])
async def list_assay_results(
    project_id: UUID,
    caller: CallerDep,
    molecule_id: Annotated[MoleculeId | None, Query(gt=0, le=MAX_RECORD_ID)] = None,
    assay: Annotated[AssayName | None, Query()] = None,
    endpoint: Annotated[AssayEndpoint | None, Query()] = None,
) -> list[AssayResult]:
    """List saved measurements in reverse recording order."""
    project = await caller.reader.project(caller.user_id, ProjectId(str(project_id)))
    results = await caller.reader.assay_results(
        project,
        molecule_ids=(molecule_id,) if molecule_id is not None else None,
        assay=assay,
        endpoint=endpoint,
    )
    return list(results)


@api.get("/projects/{project_id}/generation-runs", tags=["projects"])
async def list_project_runs(
    project_id: UUID, caller: CallerDep, rates: ModelRatesDep
) -> list[GenerationRunResponse]:
    """List runs in creation order."""
    project = await caller.reader.project(caller.user_id, ProjectId(str(project_id)))
    runs = await caller.reader.project_runs(project)
    return [
        run_response(run, estimate_cost(run.model, run.usage, rates)) for run in runs
    ]


@api.post(
    "/projects/{project_id}/sessions",
    status_code=status.HTTP_201_CREATED,
    tags=["projects"],
)
async def create_session(
    project_id: UUID, caller: CallerDep, writer: SessionWriterDep
) -> Session:
    return await writer.create_session(caller.user_id, ProjectId(str(project_id)))


@api.get("/projects/{project_id}/sessions", tags=["projects"])
async def list_sessions(project_id: UUID, caller: CallerDep) -> list[Session]:
    """List chats for the project, newest first."""
    return await caller.reader.list_sessions(caller.user_id, ProjectId(str(project_id)))


@api.get("/sessions/{session_id}", tags=["sessions"])
async def get_session(session_id: UUID, caller: CallerDep) -> Session:
    return await caller.reader.session(SessionId(str(session_id)))


@api.post(
    "/sessions/{session_id}/turns",
    response_model=TurnResponse,
    tags=["sessions"],
    responses={
        200: {
            "headers": {"Location": {"schema": {"type": "string"}}},
            "content": {"application/x-ndjson": {"schema": {"type": "string"}}},
        },
        409: {"model": ErrorResponse, "description": "Conflicting turn request"},
    },
)
async def create_turn(
    session_id: UUID,
    body: TurnRequest,
    request: Request,
    response: Response,
    caller: CallerDep,
    runner: TurnRunnerDep,
) -> TurnResponse | StreamingResponse:
    def location(turn_id: TurnId) -> str:
        return request.url_for("get_turn", session_id=session_id, turn_id=turn_id).path

    async def execute(
        progress: Progress | None = None,
        on_start: Callable[[TurnId], None] | None = None,
    ) -> TurnResponse:
        turn = await runner.run(
            user_id=caller.user_id,
            session_id=SessionId(str(session_id)),
            message=body.message,
            action=body.action,
            request_id=str(body.request_id) if body.request_id else None,
            progress=progress,
            on_start=on_start,
        )
        response.headers["Location"] = location(turn.id)
        return TurnResponse.from_record(turn)

    if accepts_ndjson(request.headers.get("accept", "")):
        await caller.reader.session(SessionId(str(session_id)))
        return progress_response(
            request,
            execute,
            id_key="turn_id",
            location=location,
            status_code=status.HTTP_200_OK,
        )
    return await execute()


@api.get("/sessions/{session_id}/turns", tags=["sessions"])
async def list_turns(
    session_id: UUID, caller: CallerDep, writer: SessionWriterDep
) -> list[TurnSummaryResponse]:
    """List turns in conversation order."""
    session = SessionId(str(session_id))
    await writer.expire_turns(caller.user_id, session)
    turns = await caller.reader.list_turns(session)
    return [TurnSummaryResponse.from_record(turn) for turn in turns]


@api.get("/sessions/{session_id}/turns/{turn_id}", tags=["sessions"])
async def get_turn(
    session_id: UUID,
    turn_id: Annotated[TurnId, Path(gt=0, le=MAX_RECORD_ID)],
    caller: CallerDep,
    writer: SessionWriterDep,
) -> TurnResponse:
    session = SessionId(str(session_id))
    await writer.expire_turns(caller.user_id, session)
    turn = await caller.reader.session_turn(session, turn_id)
    return TurnResponse.from_record(turn)


router.include_router(api)
