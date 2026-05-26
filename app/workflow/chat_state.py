from typing import Any, Dict, List, Optional

from langgraph.graph import MessagesState


class TableLineItemChatState(MessagesState):
    job_id: str
    run_id: str
    session_id: Optional[str]
    request_id: Optional[str]
    user_id: Optional[str]
    execution_id: Optional[int]
    utility_id: Optional[str]
    utility_name: Optional[str]

    index_id: str
    top_k: int

    auto_suggest: bool
    input_paths: List[Dict[str, Any]]
    document_catalog: List[Dict[str, Any]]
    document_quick_summaries: List[Dict[str, Any]]
    document_context_brief: Optional[str]
    document_summary: Optional[str]

    user_query: str
    current_step: str
    query_intent: Optional[str]
    query_for_retrieval: Optional[str]
    thoughts_summary: Optional[str]
    short_circuit: bool

    intent_relevant: bool
    intent_reason: Optional[str]
    needs_retrieval: bool
    target_file_names: List[str]

    matched_file_ids: List[str]
    matched_file_names: List[str]
    retrieval_scope: Optional[str]
    needs_sql: bool

    reuse_context_hit: bool
    reuse_context_reason: Optional[str]

    retrieved_chunks: List[Dict[str, Any]]
    auto_suggestion_output: Optional[Dict[str, Any]]
    answer: Optional[str]
    sources: List[Dict[str, Any]]

    completed_steps: List[str]
    errors: List[str]

