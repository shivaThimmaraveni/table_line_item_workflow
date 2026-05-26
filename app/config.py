import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _as_bool(value: str | bool | None, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _parse_csv_list(value: str | None) -> tuple[str, ...]:
    if not value:
        return tuple()
    parts = [p.strip() for p in value.split(",")]
    return tuple(p for p in parts if p)


def _env_or_default(name: str, default: str) -> str:
    raw = os.getenv(name)
    if raw is None:
        return default
    stripped = raw.strip()
    return stripped if stripped else default


@dataclass(frozen=True)
class Settings:
    app_host: str
    app_port: int
    app_reload: bool
    aws_region: str
    bedrock_chat_model: str
    bedrock_chat_provider: str
    bedrock_summary_model: str
    bedrock_summary_provider: str
    bedrock_embed_model: str
    bedrock_embed_fallback_models: tuple[str, ...]
    data_dir: Path
    documents_dir: Path
    db_path: Path
    max_chunk_chars: int
    chunk_overlap_chars: int
    default_top_k: int
    max_top_k: int


def get_settings() -> Settings:
    data_dir = Path(os.getenv("DATA_DIR", "./data")).resolve()
    db_path = Path(os.getenv("DB_PATH", "./table_line_item.db")).resolve()
    return Settings(
        app_host=os.getenv("APP_HOST", "0.0.0.0"),
        app_port=int(os.getenv("APP_PORT", "8080")),
        app_reload=_as_bool(os.getenv("APP_RELOAD", "true"), True),
        aws_region=os.getenv("AWS_REGION", "us-east-1"),
        bedrock_chat_model=_env_or_default(
            "BEDROCK_CHAT_MODEL",
            "global.anthropic.claude-sonnet-4-5-20250929-v1:0",
        ),
        bedrock_chat_provider=_env_or_default(
            "BEDROCK_CHAT_PROVIDER",
            "anthropic",
        ),
        bedrock_summary_model=_env_or_default(
            "BEDROCK_SUMMARY_MODEL",
            "global.anthropic.claude-sonnet-4-5-20250929-v1:0",
        ),
        bedrock_summary_provider=_env_or_default(
            "BEDROCK_SUMMARY_PROVIDER",
            "anthropic",
        ),
        bedrock_embed_model=_env_or_default(
            "BEDROCK_EMBED_MODEL",
            "us.amazon.titan-embed-text-v2:0",
        ),
        bedrock_embed_fallback_models=_parse_csv_list(
            os.getenv("BEDROCK_EMBED_FALLBACK_MODELS")
        ),
        data_dir=data_dir,
        documents_dir=(data_dir / "documents"),
        db_path=db_path,
        max_chunk_chars=int(os.getenv("MAX_CHUNK_CHARS", "1400")),
        chunk_overlap_chars=int(os.getenv("CHUNK_OVERLAP_CHARS", "200")),
        default_top_k=int(os.getenv("DEFAULT_TOP_K", "5")),
        max_top_k=int(os.getenv("MAX_TOP_K", "20")),
    )

