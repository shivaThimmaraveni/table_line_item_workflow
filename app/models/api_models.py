from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class DocumentRef(BaseModel):
    file_id: str
    file_name: str
    source: str


class ChatRequest(BaseModel):
    job_id: str
    run_id: str
    user_query: Optional[str] = ""
    auto_suggest: bool = False
    top_k: int = 5
    document_ids: List[str] = Field(default_factory=list)
    request_id: Optional[str] = None
    user_id: Optional[str] = None
    execution_id: Optional[int] = None
    utility_id: Optional[str] = None
    utility_name: Optional[str] = "table_line_item"


class UploadResponse(BaseModel):
    file_id: str
    file_name: str
    local_path: str
    size: int
    job_id: str
    run_id: str
    index_id: str
    chunk_count: int


class HealthResponse(BaseModel):
    status: str
    service: str
    bedrock_region: str
    db_path: str
    data_dir: str
    local_only: bool


class DonePayload(BaseModel):
    type: str = "done"
    answer: Optional[str] = None
    auto_suggestion_output: Optional[Dict[str, Any]] = None
    document_quick_summaries: List[Dict[str, Any]] = Field(default_factory=list)
    sources: List[Dict[str, Any]] = Field(default_factory=list)
    completed_steps: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    intent_relevant: Optional[bool] = None
    query_intent: Optional[str] = None
    query_for_retrieval: Optional[str] = None
    thoughts_summary: Optional[str] = None
    session_id: Optional[str] = None
    retrieval_scope: Optional[str] = None
    matched_files: List[str] = Field(default_factory=list)

