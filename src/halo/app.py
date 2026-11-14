import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from functools import partial

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from httpx import AsyncClient, Timeout

from halo.api.errors import ErrorBoundary, register_error_handlers
from halo.api.router import router
from halo.chem.eval.profile import PropertyPredictor
from halo.chem.medchem import StructuralAnnotator
from halo.chem.predict.admet_ai import ADMETAIPredictor
from halo.clients.chem.chembl import ChEMBLClient
from halo.clients.chem.europe_pmc import EuropePMCClient
from halo.clients.chem.pubchem import PubChemClient
from halo.config.settings import ModelSettings, Settings, load_settings
from halo.core.log import configure_logging
from halo.db.client import auth_client, auth_url, service_client
from halo.db.session import SessionWriter
from halo.db.store import GenerationStore
from halo.llm.providers.base import ModelProvider
from halo.llm.providers.openai import OpenAIProvider
from halo.llm.runtime import ModelRunner
from halo.workflow.generation import GenerationRunner
from halo.workflow.research import ResearchWorkflow, SourceFinder
from halo.workflow.turn import TurnRunner

logger = logging.getLogger(__name__)
SUPABASE_TIMEOUT_S = 120


def create_app(
    settings: Settings | None = None,
    *,
    predictor: PropertyPredictor | None = None,
    provider: ModelProvider | None = None,
) -> FastAPI:
    resolved = settings or load_settings()
    configure_logging(resolved.server.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        async with AsyncExitStack() as stack:
            if resolved.config_path is not None:
                logger.info("Loaded configuration from %s", resolved.config_path)
            http = await stack.enter_async_context(
                AsyncClient(
                    timeout=Timeout(SUPABASE_TIMEOUT_S, connect=5.0), http2=True
                )
            )
            app.state.supabase_http = http
            app.state.auth = auth_client(resolved.supabase, http)
            app.state.auth_issuer = resolved.supabase.auth_issuer or auth_url(
                resolved.supabase
            )
            client = service_client(resolved.supabase, http)
            model_provider = provider
            if model_provider is None:
                model_provider = OpenAIProvider(resolved.openai)
                stack.push_async_callback(model_provider.aclose)
            app.state.settings = resolved
            integrations = resolved.integrations
            property_predictor = predictor
            if property_predictor is None:
                property_predictor = ADMETAIPredictor.load(
                    oracle_data_dir=integrations.oracle_data_dir
                )
                await asyncio.to_thread(property_predictor.warmup)
            roles = resolved.agents.roles
            optimizer = ModelRunner(
                provider=model_provider,
                settings=ModelSettings.model_validate(
                    roles.get("optimizer", resolved.optimizer).model_dump(
                        exclude={"samples"}
                    )
                ),
                runtime=resolved.runtime,
            )
            task = ModelRunner(
                provider=model_provider,
                settings=ModelSettings.model_validate(
                    resolved.agents.model_dump(exclude={"roles"})
                ),
                runtime=resolved.runtime,
                roles=roles,
            )
            chembl = await stack.enter_async_context(
                ChEMBLClient(
                    requests_per_second=integrations.chembl_requests_per_second
                )
            )
            pubchem = await stack.enter_async_context(
                PubChemClient(
                    requests_per_second=integrations.pubchem_requests_per_second
                )
            )
            papers = await stack.enter_async_context(
                EuropePMCClient(
                    requests_per_second=integrations.europe_pmc_requests_per_second
                )
            )
            sources = SourceFinder(chembl, pubchem, papers)
            research = ResearchWorkflow(sources, task)
            generation_store = GenerationStore(client)
            session_writer = SessionWriter(client)
            generation = GenerationRunner(
                optimizer,
                property_predictor,
                generation_store,
                samples=resolved.optimizer.samples,
                prediction_methods=(
                    property_predictor.methods
                    if isinstance(property_predictor, ADMETAIPredictor)
                    else None
                ),
            )
            app.state.generation_runner = generation
            app.state.generation_store = generation_store
            app.state.session_writer = session_writer
            app.state.research_workflow = research
            app.state.make_turn_runner = partial(
                TurnRunner,
                writer=session_writer,
                task=task,
                generator=generation,
                research=research,
                annotate=StructuralAnnotator().annotate_many,
            )
            logger.info("halo starting")
            app.state.ready = True
            try:
                yield
            finally:
                app.state.ready = False
                workers = tuple(app.state.progress_tasks)
                for worker in workers:
                    worker.cancel()
                await asyncio.gather(*workers, return_exceptions=True)
        logger.info("halo stopped")

    app = FastAPI(title="HALO", version="0.1.0", lifespan=lifespan)
    app.state.progress_tasks = set()
    app.add_middleware(ErrorBoundary)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=resolved.server.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type"],
        expose_headers=["Location"],
    )
    register_error_handlers(app)
    app.include_router(router)

    return app


app = create_app()
