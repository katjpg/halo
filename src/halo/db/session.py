import asyncio
import re
from collections.abc import Callable, Sequence
from datetime import datetime
from typing import Literal

from httpx import RequestError
from postgrest import AsyncPostgrestClient, AsyncSelectRequestBuilder
from pydantic import TypeAdapter

from halo.core.errors import DataError, NotFoundError
from halo.db.client import execute
from halo.db.store import (
    CANDIDATES,
    MODEL_CALLS,
    RGROUP_EDIT,
    RUN_CONFIG,
    RUN_SELECT,
    candidate_payload,
    embedded_row,
    profile_from_row,
    run_config_from_row,
    run_from_row,
)
from halo.llm.runtime import ModelCall
from halo.models.analysis import AssayResult
from halo.models.generation import (
    DesignConfig,
    GenerationBatch,
    GenerationWrite,
    Occurrence,
    RGroupEdit,
    Run,
)
from halo.models.molecule import MoleculeProfile
from halo.models.session import (
    STATE,
    TURN_RESULT,
    Decision,
    Project,
    RecentTurn,
    Session,
    SessionState,
    StartedTurn,
    TurnRecord,
    TurnResult,
    TurnStatus,
)
from halo.types import (
    DecisionId,
    GenerationRunId,
    MoleculeId,
    ProjectId,
    SessionId,
    TurnId,
    UserId,
)

PROJECT = TypeAdapter(Project)
ASSAY_RESULT = TypeAdapter(AssayResult)
SESSION = TypeAdapter(Session)
TURN = TypeAdapter(TurnRecord)
DECISION = TypeAdapter(Decision)
READ_BATCH_SIZE = 100
READINESS_TIMEOUT_S = 2
PROFILE_FIELDS = "id,canonical_smiles,molecule_profiles!inner(*)"
RECENT_TURNS = 3
RECENT_TEXT_CHARS = 400
RECENT_MOLECULES = 12
RECENT_EDITS_RUNS = 8
TURN_SUMMARY_FIELDS = (
    "message",
    "response",
    "error",
    "error_code",
)
TURN_SUMMARY_SELECT = (
    "id,session_id,turn_index,status,request,request_id,started_at,finished_at,task,run_id,decision_id,"
    + ",".join(f"{field}:output->{field}" for field in TURN_SUMMARY_FIELDS)
)


def turn_from_row(row: dict) -> TurnRecord:
    output = row["output"]
    if output is not None:
        output = output | {
            "task": row["task"],
            "generation_run_id": row["run_id"],
            "decision_id": row["decision_id"],
        }
        if proposal := output.get("proposal"):
            output["proposal"] = proposal | {"comparison_ids": row["comparison_ids"]}
        row = row | {"output": output}
    return TURN.validate_python(row)


async def _list_rows(
    select: Callable[[], AsyncSelectRequestBuilder],
    *,
    order: Literal["id", "created_at", "turn_index"] = "id",
    limit: int | None = None,
) -> list[dict]:
    boundary = None
    rows = []
    while limit is None or len(rows) < limit:
        query = select().order(order, desc=True)
        if order == "created_at":
            query = query.order("id", desc=True)
        if boundary is not None:
            if order == "created_at":
                timestamp = boundary["created_at"]
                query = query.or_(
                    f"created_at.lt.{timestamp},and(created_at.eq.{timestamp},id.lt.{boundary['id']})"
                )
            else:
                query = query.lt(order, boundary[order])
        count = (
            READ_BATCH_SIZE
            if limit is None
            else min(READ_BATCH_SIZE, limit - len(rows))
        )
        batch = (await execute(query.limit(count).execute())).data
        if not batch:
            break
        rows.extend(batch)
        boundary = batch[-1]
    return rows


class SessionReader:
    def __init__(self, client: AsyncPostgrestClient) -> None:
        self.client = client

    async def generation_run(self, run_id: GenerationRunId) -> Run:
        """Read a visible run with its candidates in ordinal order."""
        result = await execute(
            self.client.table("runs")
            .select(RUN_SELECT)
            .eq("id", run_id)
            .limit(1)
            .execute()
        )
        if not result.data:
            raise NotFoundError(f"Generation run {run_id} does not exist.")
        return run_from_row(result.data[0])

    async def molecule_profile(self, molecule_id: MoleculeId) -> MoleculeProfile:
        """Read a visible profile or raise NotFoundError."""
        result = await execute(
            self.client.table("molecules")
            .select("canonical_smiles, molecule_profiles(*)")
            .eq("id", molecule_id)
            .limit(1)
            .execute()
        )
        if not result.data:
            raise NotFoundError(f"Molecule {molecule_id} does not exist.")

        row = result.data[0]
        profile = row.get("molecule_profiles")
        if not profile:
            raise NotFoundError(f"Molecule {molecule_id} has no profile.")
        return profile_from_row(row["canonical_smiles"], embedded_row(profile))

    async def project(self, user_id: UserId, project_id: ProjectId) -> Project:
        result = await execute(
            self.client.table("projects")
            .select("*")
            .eq("id", project_id)
            .eq("user_id", user_id)
            .limit(1)
            .execute()
        )
        if not result.data:
            raise NotFoundError("The project does not exist.")
        return PROJECT.validate_python(result.data[0])

    async def list_projects(self, user_id: UserId) -> list[Project]:
        rows = await _list_rows(
            lambda: self.client.table("projects").select("*").eq("user_id", user_id),
            order="created_at",
        )
        return [PROJECT.validate_python(row) for row in rows]

    async def assay_results(
        self,
        project: Project,
        *,
        molecule_ids: tuple[MoleculeId, ...] | None = None,
        assay: str | None = None,
        endpoint: str | None = None,
        limit: int | None = None,
    ) -> tuple[AssayResult, ...]:
        if molecule_ids == ():
            return ()

        def select() -> AsyncSelectRequestBuilder:
            query = (
                self.client.table("assay_results")
                .select("*")
                .eq("project_id", project.id)
            )
            if molecule_ids is not None:
                query = query.in_("molecule_id", molecule_ids)
            for column, label in (("assay", assay), ("endpoint", endpoint)):
                if label is not None:
                    query = query.filter(column, "imatch", f"^{re.escape(label)}$")
            return query

        rows = await _list_rows(select, limit=limit)
        return tuple(ASSAY_RESULT.validate_python(row) for row in rows)

    async def list_sessions(
        self, user_id: UserId, project_id: ProjectId
    ) -> list[Session]:
        await self.project(user_id, project_id)
        rows = await _list_rows(
            lambda: (
                self.client.table("sessions").select("*").eq("project_id", project_id)
            ),
            order="created_at",
        )
        return [SESSION.validate_python(row) for row in rows]

    async def session(self, session_id: SessionId) -> Session:
        result = await execute(
            self.client.table("sessions")
            .select("*")
            .eq("id", session_id)
            .limit(1)
            .execute()
        )
        if not result.data:
            raise NotFoundError("The session does not exist.")
        return SESSION.validate_python(result.data[0])

    async def list_turns(self, session_id: SessionId) -> list[TurnRecord]:
        await self.session(session_id)
        rows = await _list_rows(
            lambda: (
                self.client.table("session_turns")
                .select(TURN_SUMMARY_SELECT)
                .eq("session_id", session_id)
            ),
            order="turn_index",
        )
        turns = []
        for row in reversed(rows):
            output = (
                {field: row[field] for field in TURN_SUMMARY_FIELDS}
                if row["status"] != "running"
                else None
            )
            turns.append(turn_from_row(row | {"output": output, "trace": None}))
        return turns

    async def session_turn(self, session_id: SessionId, turn_id: TurnId) -> TurnRecord:
        result = await execute(
            self.client.table("session_turns")
            .select("*")
            .eq("session_id", session_id)
            .eq("id", turn_id)
            .limit(1)
            .execute()
        )
        if not result.data:
            raise NotFoundError("The turn does not exist in this session.")
        return turn_from_row(result.data[0])

    async def project_runs(self, project: Project) -> tuple[Run, ...]:
        rows = await _list_rows(
            lambda: (
                self.client.table("runs")
                .select(RUN_SELECT)
                .eq("project_id", project.id)
                .eq("user_id", project.user_id)
            ),
            order="created_at",
        )
        return tuple(run_from_row(row) for row in reversed(rows))

    async def latest_batch(self, project: Project) -> GenerationBatch | None:
        result = await execute(
            self.client.table("runs")
            .select(
                "source_molecule_id,parent_run_id,config,"
                f"{CANDIDATES}!inner(ordinal,molecule_id)"
            )
            .eq("project_id", project.id)
            .eq("user_id", project.user_id)
            .order("created_at", desc=True)
            .order("id", desc=True)
            .order("ordinal", foreign_table="candidates")
            .limit(1)
            .execute()
        )
        if not result.data:
            return None
        row = result.data[0]
        config = run_config_from_row(row)
        return GenerationBatch(
            Occurrence(row["source_molecule_id"], row["parent_run_id"]),
            tuple(item["molecule_id"] for item in row["candidates"]),
            config.criteria if isinstance(config, DesignConfig) else {},
        )

    def project_molecules(self, project: Project) -> AsyncSelectRequestBuilder:
        return (
            self.client.table("project_molecules")
            .select("molecule_id")
            .eq("project_id", project.id)
        )

    async def molecule_ids(
        self, project: Project, ids: tuple[MoleculeId, ...]
    ) -> set[MoleculeId]:
        if not ids:
            return set()
        result = await execute(
            self.project_molecules(project).in_("molecule_id", ids).execute()
        )
        return {row["molecule_id"] for row in result.data}

    async def profiles(
        self, project: Project, ids: tuple[MoleculeId, ...] | None, *, limit: int
    ) -> dict[MoleculeId, MoleculeProfile]:
        """Load the seed and project candidates in molecule ID order."""
        query = self.project_molecules(project).order("molecule_id").limit(limit)
        if ids is not None:
            if not ids:
                return {}
            query = query.in_("molecule_id", ids)
        members = await execute(query.execute())
        if not members.data:
            return {}
        result = await execute(
            self.client.table("molecules")
            .select(PROFILE_FIELDS)
            .eq("user_id", project.user_id)
            .in_("id", [row["molecule_id"] for row in members.data])
            .order("id")
            .execute()
        )
        return {
            row["id"]: profile_from_row(
                row["canonical_smiles"], embedded_row(row["molecule_profiles"])
            )
            for row in result.data
        }

    async def profile_at(self, project: Project, source: Occurrence) -> MoleculeProfile:
        if source.run_id is None:
            if source.molecule_id == project.lead_id:
                return await self.molecule_profile(source.molecule_id)
        else:
            result = await execute(
                self.client.table("candidates")
                .select("runs!candidates_run_fkey!inner(project_id)")
                .eq("runs.project_id", project.id)
                .eq("run_id", source.run_id)
                .eq("molecule_id", source.molecule_id)
                .limit(1)
                .execute()
            )
            if result.data:
                return await self.molecule_profile(source.molecule_id)
        raise NotFoundError("The selected parent occurrence is outside this project.")

    async def matching_smiles(
        self, project: Project, smiles: Sequence[str]
    ) -> set[str]:
        if not smiles:
            return set()
        result = await execute(
            self.client.table("molecules")
            .select("id,canonical_smiles")
            .eq("user_id", project.user_id)
            .in_("canonical_smiles", smiles)
            .execute()
        )
        members = await self.molecule_ids(
            project, tuple(row["id"] for row in result.data)
        )
        return {row["canonical_smiles"] for row in result.data if row["id"] in members}

    async def core_smiles(self, project: Project, source: Occurrence) -> str | None:
        query = (
            self.client.table("runs")
            .select("core_smiles:config->core_smiles")
            .eq("project_id", project.id)
            .eq("user_id", project.user_id)
            .not_.is_("config->>core_smiles", "null")
            .order("created_at", desc=True)
            .order("id", desc=True)
            .limit(1)
        )
        if source.run_id is None:
            query = query.eq("source_molecule_id", source.molecule_id)
        else:
            query = query.eq("id", source.run_id)
        result = await execute(query.execute())
        return result.data[0]["core_smiles"] if result.data else None

    async def recent_edits(self, project: Project) -> tuple[RGroupEdit, ...]:
        result = await execute(
            self.client.table("runs")
            .select(f"{CANDIDATES}(transformation:design->transformation)")
            .eq("project_id", project.id)
            .eq("user_id", project.user_id)
            .order("created_at", desc=True)
            .order("id", desc=True)
            .order("ordinal", foreign_table="candidates")
            .limit(RECENT_EDITS_RUNS)
            .execute()
        )
        return tuple(
            RGROUP_EDIT.validate_python(entry["transformation"])
            for row in reversed(result.data)
            for entry in row["candidates"]
            if entry["transformation"]
        )

    async def recent_turns(self, start: StartedTurn) -> tuple[RecentTurn, ...]:
        result = await execute(
            self.client.table("session_turns")
            .select(
                "request,task,message:output->>message,"
                "finding:output->response->findings->0->>text,"
                "mentioned_ids:output->response->findings->0->molecule_ids,"
                "candidate_ids:output->candidate_ids,"
                "visible_ids:output->plot->visible_ids,"
                "comparison_ids"
            )
            .eq("session_id", start.turn.session_id)
            .eq("status", "completed")
            .order("turn_index", desc=True)
            .limit(RECENT_TURNS)
            .execute()
        )
        recent = []
        for row in reversed(result.data):
            request = row["request"]
            action = request.get("action") or {}
            action_ids = action.get("molecule_ids")
            if (molecule_id := action.get("molecule_id")) is not None:
                action_ids = (molecule_id,)
            ids = dict.fromkeys(
                mid
                for group in (
                    action_ids,
                    row["comparison_ids"],
                    row["candidate_ids"],
                    row["visible_ids"],
                    row["mentioned_ids"],
                )
                for mid in group or ()
            )
            text = (
                request.get("message")
                or action.get("question")
                or action.get("type", "")
            )
            recent.append(
                RecentTurn(
                    task=row["task"],
                    request=text[:RECENT_TEXT_CHARS],
                    response=(row["message"] or row["finding"] or "")[
                        :RECENT_TEXT_CHARS
                    ],
                    molecule_ids=tuple(ids)[:RECENT_MOLECULES],
                )
            )
        return tuple(recent)

    async def turn(self, project: Project, turn_id: TurnId) -> TurnRecord:
        result = await execute(
            self.client.table("session_turns")
            .select("*")
            .eq("id", turn_id)
            .eq("project_id", project.id)
            .limit(1)
            .execute()
        )
        if not result.data:
            raise NotFoundError("The turn does not exist in this project.")
        return turn_from_row(result.data[0])

    async def decisions(
        self,
        project: Project,
        *,
        ids: tuple[DecisionId, ...] = (),
        as_of: datetime | None = None,
    ) -> tuple[Decision, ...]:
        result = await execute(
            self.client.rpc(
                "project_decisions",
                {
                    "p_user_id": project.user_id,
                    "p_project_id": project.id,
                    "p_ids": ids,
                    "p_as_of": as_of.isoformat() if as_of else None,
                },
            ).execute()
        )
        return tuple(DECISION.validate_python(row) for row in result.data)


class SessionWriter:
    def __init__(self, client: AsyncPostgrestClient) -> None:
        self.client = client

    async def is_ready(self) -> bool:
        """Check whether the service connection can query the project schema."""
        try:
            async with asyncio.timeout(READINESS_TIMEOUT_S):
                await execute(
                    self.client.from_("projects").select("id").limit(0).execute()
                )
        except (DataError, RequestError, TimeoutError):
            return False
        return True

    async def create_project(
        self, user_id: UserId, source_molecule_id: MoleculeId
    ) -> Project:
        result = await execute(
            self.client.rpc(
                "create_project",
                {"p_user_id": user_id, "p_source_molecule_id": source_molecule_id},
            ).execute()
        )
        return PROJECT.validate_python(result.data)

    async def record_assay_result(
        self, user_id: UserId, project_id: ProjectId, result: dict
    ) -> AssayResult:
        saved = await execute(
            self.client.rpc(
                "record_assay_result",
                {"p_user_id": user_id, "p_project_id": project_id, "p_result": result},
            ).execute()
        )
        return ASSAY_RESULT.validate_python(saved.data)

    async def create_session(self, user_id: UserId, project_id: ProjectId) -> Session:
        result = await execute(
            self.client.rpc(
                "create_session", {"p_user_id": user_id, "p_project_id": project_id}
            ).execute()
        )
        return SESSION.validate_python(result.data)

    async def expire_turns(self, user_id: UserId, session_id: SessionId) -> None:
        await execute(
            self.client.rpc(
                "expire_turns", {"p_user_id": user_id, "p_session_id": session_id}
            ).execute()
        )

    async def start_turn(
        self,
        user_id: UserId,
        session_id: SessionId,
        request: dict,
        request_id: str | None,
    ) -> StartedTurn | TurnRecord:
        result = await execute(
            self.client.rpc(
                "start_project_turn",
                {
                    "p_user_id": user_id,
                    "p_session_id": session_id,
                    "p_request": request,
                    "p_request_id": request_id,
                },
            ).execute()
        )
        row = result.data
        turn = turn_from_row(row["turn"])
        if row["replay"]:
            return turn
        return StartedTurn(
            turn,
            PROJECT.validate_python(row["project"]),
            STATE.validate_python(row["state"]),
        )

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
    ) -> TurnRecord:
        generation_payload = None
        if generation is not None:
            candidates = generation.prepared.candidates
            generation_payload = {
                "source_molecule_id": generation.source.molecule_id,
                "parent_run_id": generation.source.run_id,
                "model": generation.model,
                "config": RUN_CONFIG.dump_python(
                    generation.config, mode="json", exclude_none=True
                ),
                "candidates": candidate_payload(
                    candidates, tuple(generation.config.criteria)
                ),
                "usage": generation.usage.model_dump(),
            }
        saved = await execute(
            self.client.rpc(
                "finish_project_turn",
                {
                    "p_user_id": start.project.user_id,
                    "p_turn_id": start.turn.id,
                    "p_status": status,
                    "p_task": result.task,
                    "p_output": TURN_RESULT.dump_python(
                        result, mode="json", by_alias=True
                    ),
                    "p_trace": trace,
                    "p_state": STATE.dump_python(state, mode="json", by_alias=True),
                    "p_calls": MODEL_CALLS.dump_python(calls, mode="json"),
                    "p_comparison_ids": result.proposal.comparison_ids
                    if result.proposal
                    else (),
                    "p_preference_ids": generation.preference_ids if generation else (),
                    "p_generation": generation_payload,
                },
            ).execute()
        )
        return turn_from_row(saved.data)
