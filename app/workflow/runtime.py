from dataclasses import dataclass

from app.config import Settings
from app.connectors.bedrock import BedrockConnector
from app.storage.sqlite_store import SQLiteStore


@dataclass
class WorkflowRuntime:
    settings: Settings
    bedrock: BedrockConnector
    store: SQLiteStore


runtime: WorkflowRuntime | None = None


def set_runtime(value: WorkflowRuntime) -> None:
    global runtime
    runtime = value


def get_runtime() -> WorkflowRuntime:
    if runtime is None:
        raise RuntimeError("Workflow runtime not initialized.")
    return runtime

