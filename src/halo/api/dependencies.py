from collections.abc import Mapping
from dataclasses import dataclass
from typing import Annotated

from fastapi import Depends, Request

from halo.api.auth import access_token, verify_token
from halo.config.settings import ModelRates
from halo.db.client import user_client
from halo.db.session import SessionReader, SessionWriter
from halo.db.store import GenerationStore
from halo.types import UserId
from halo.workflow.generation import GenerationRunner
from halo.workflow.research import ResearchWorkflow
from halo.workflow.turn import TurnRunner


@dataclass(frozen=True)
class Caller:
    user_id: UserId
    reader: SessionReader


async def get_caller(
    request: Request, token: Annotated[str, Depends(access_token)]
) -> Caller:
    state = request.app.state
    user_id = await verify_token(state.auth, token, issuer=state.auth_issuer)
    client = user_client(state.settings.supabase, state.supabase_http, token)
    return Caller(user_id, SessionReader(client))


CallerDep = Annotated[Caller, Depends(get_caller)]


def get_turn_runner(request: Request, caller: CallerDep) -> TurnRunner:
    return request.app.state.make_turn_runner(caller.reader)


def get_session_writer(request: Request) -> SessionWriter:
    return request.app.state.session_writer


def get_generation_store(request: Request) -> GenerationStore:
    return request.app.state.generation_store


def get_generation_runner(request: Request) -> GenerationRunner:
    return request.app.state.generation_runner


def get_research_workflow(request: Request) -> ResearchWorkflow:
    return request.app.state.research_workflow


def get_model_rates(request: Request) -> Mapping[str, ModelRates]:
    return request.app.state.settings.model_rates


TurnRunnerDep = Annotated[TurnRunner, Depends(get_turn_runner)]
SessionWriterDep = Annotated[SessionWriter, Depends(get_session_writer)]
GenerationStoreDep = Annotated[GenerationStore, Depends(get_generation_store)]
GenerationRunnerDep = Annotated[GenerationRunner, Depends(get_generation_runner)]
ResearchWorkflowDep = Annotated[ResearchWorkflow, Depends(get_research_workflow)]
ModelRatesDep = Annotated[Mapping[str, ModelRates], Depends(get_model_rates)]
