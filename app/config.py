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


@dataclass(frozen=True)
class Settings:
    app_host: str
    app_port: int
    app_reload: bool
    aws_region: str
    bedrock_chat_model: str
    bedrock_summary_model: str
    bedrock_embed_model: str
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
        bedrock_chat_model=os.getenv(
            "BEDROCK_CHAT_MODEL",
            "global.anthropic.claude-sonnet-4-5-20250929-v1:0",
        ),
        bedrock_summary_model=os.getenv(
            "BEDROCK_SUMMARY_MODEL",
            "anthropic.claude-3-haiku-20240307-v1:0",
        ),
        bedrock_embed_model=os.getenv(
            "BEDROCK_EMBED_MODEL",
            "amazon.titan-embed-text-v2:0",
        ),
        data_dir=data_dir,
        documents_dir=(data_dir / "documents"),
        db_path=db_path,
        max_chunk_chars=int(os.getenv("MAX_CHUNK_CHARS", "1400")),
        chunk_overlap_chars=int(os.getenv("CHUNK_OVERLAP_CHARS", "200")),
        default_top_k=int(os.getenv("DEFAULT_TOP_K", "5")),
        max_top_k=int(os.getenv("MAX_TOP_K", "20")),
    )

