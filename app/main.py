from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.config import Settings, get_settings
from app.connectors.bedrock import BedrockConnector
from app.logging_setup import configure_logging, get_logger
from app.storage.sqlite_store import SQLiteStore
from app.workflow.chat_workflow import build_chat_workflow
from app.workflow.runtime import WorkflowRuntime, set_runtime


logger = get_logger(__name__)


def _validate_local_only_runtime() -> None:
    forbidden = [
        "S3_UPLOAD_BUCKET",
        "DATABASE_URL",
        "DB_HOST",
        "REDIS_URL",
        "VALKEY_HOST",
        "SQS_DOCAI_PROCESSING_QUEUE",
        "SECRET_NAME",
    ]
    enabled = [k for k in forbidden if k in __import__("os").environ]
    if enabled:
        logger.warning(
            "Local-only mode: ignoring non-local service env vars: %s",
            ", ".join(enabled),
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    settings: Settings = get_settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.documents_dir.mkdir(parents=True, exist_ok=True)

    _validate_local_only_runtime()

    store = SQLiteStore(settings.db_path)
    store.initialize()
    bedrock = BedrockConnector(settings)
    chat_workflow = build_chat_workflow()

    app.state.settings = settings
    app.state.store = store
    app.state.bedrock = bedrock
    app.state.chat_workflow = chat_workflow

    set_runtime(
        WorkflowRuntime(
            settings=settings,
            bedrock=bedrock,
            store=store,
        )
    )

    logger.info("Local service initialized. db=%s data=%s", settings.db_path, settings.data_dir)
    yield


app = FastAPI(
    title="Table Line Item Local Service",
    version="1.0.0",
    description="Standalone FastAPI for document chat using Bedrock and local SQLite.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

