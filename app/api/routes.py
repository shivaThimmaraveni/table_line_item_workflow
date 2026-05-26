import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import Any, Dict, List

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage

from app.config import Settings
from app.logging_setup import get_logger
from app.models.api_models import ChatRequest, HealthResponse, UploadResponse
from app.storage.document_indexer import ensure_supported_file, index_document

logger = get_logger(__name__)

router = APIRouter()


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


@router.get("/health", response_model=HealthResponse)
async def health(request: Request):
    settings: Settings = request.app.state.settings
    return HealthResponse(
        status="online",
        service="table-line-item-local-service",
        bedrock_region=settings.aws_region,
        db_path=str(settings.db_path),
        data_dir=str(settings.data_dir),
        local_only=True,
    )


@router.post("/v1/documents/upload", response_model=UploadResponse)
async def upload_document(
    request: Request,
    file: UploadFile = File(...),
    job_id: str = "",
    run_id: str = "",
):
    if not job_id or not run_id:
        raise HTTPException(status_code=400, detail="job_id and run_id are required")

    settings: Settings = request.app.state.settings
    store = request.app.state.store
    bedrock = request.app.state.bedrock

    if not file.filename:
        raise HTTPException(status_code=400, detail="file filename is required")

    target_dir = settings.documents_dir / job_id / run_id
    target_dir.mkdir(parents=True, exist_ok=True)
    file_id = str(uuid.uuid4())
    target_path = target_dir / f"{file_id}__{Path(file.filename).name}"

    content = await file.read()
    target_path.write_bytes(content)

    try:
        ensure_supported_file(target_path)
    except Exception as e:
        target_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=str(e))

    content_hash = _hash_file(target_path)
    store.add_document(
        file_id=file_id,
        job_id=job_id,
        run_id=run_id,
        file_name=Path(file.filename).name,
        local_path=str(target_path),
        file_ext=target_path.suffix.lower(),
        file_size=os.path.getsize(target_path),
        content_hash=content_hash,
    )
    index_meta = await index_document(
        store=store,
        bedrock=bedrock,
        settings=settings,
        file_id=file_id,
        file_name=Path(file.filename).name,
        local_path=target_path,
        job_id=job_id,
        run_id=run_id,
    )
    return UploadResponse(
        file_id=file_id,
        file_name=Path(file.filename).name,
        local_path=str(target_path),
        size=os.path.getsize(target_path),
        job_id=job_id,
        run_id=run_id,
        index_id=index_meta["index_id"],
        chunk_count=index_meta["chunk_count"],
    )


def _serialize_messages(messages: List[Any]) -> str:
    rows = []
    for m in messages:
        role = "human" if isinstance(m, HumanMessage) else "ai"
        rows.append({"role": role, "content": m.content})
    return json.dumps(rows)


def _deserialize_messages(data: str) -> List[Any]:
    try:
        rows = json.loads(data)
    except Exception:
        return []
    out: List[Any] = []
    for row in rows:
        role = row.get("role")
        content = row.get("content", "")
        if role == "human":
            out.append(HumanMessage(content=content))
        elif role == "ai":
            out.append(AIMessage(content=content))
    return out


def _done_payload(values: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "type": "done",
        "answer": values.get("answer"),
        "auto_suggestion_output": values.get("auto_suggestion_output"),
        "document_quick_summaries": values.get("document_quick_summaries") or [],
        "sources": values.get("sources") or [],
        "completed_steps": values.get("completed_steps") or [],
        "errors": values.get("errors") or [],
        "intent_relevant": values.get("intent_relevant"),
        "query_intent": values.get("query_intent"),
        "query_for_retrieval": values.get("query_for_retrieval"),
        "thoughts_summary": values.get("thoughts_summary"),
        "session_id": values.get("session_id"),
        "retrieval_scope": values.get("retrieval_scope"),
        "matched_files": values.get("matched_file_names") or [],
    }


@router.post("/v1/analytics/table-line-item")
async def chat_table_line_item(req: ChatRequest, request: Request):
    workflow = request.app.state.chat_workflow
    store = request.app.state.store

    session_id = f"{req.job_id}-{req.run_id}"
    index_id = f"kb-{req.job_id}-{req.run_id}"
    request_id = req.request_id or str(uuid.uuid4())

    docs = store.get_documents_by_ids(req.job_id, req.run_id, req.document_ids)
    if not docs:
        raise HTTPException(
            status_code=400,
            detail="No documents found for this job/run. Upload first.",
        )
    document_catalog = [
        {
            "file_id": d["file_id"],
            "file_name": d["file_name"],
            "source": d["local_path"],
        }
        for d in docs
    ]

    checkpoint = store.get_checkpoint(session_id)
    prior_messages = _deserialize_messages(checkpoint["messages_json"]) if checkpoint else []
    messages = list(prior_messages)
    if req.user_query:
        messages.append(HumanMessage(content=req.user_query))

    initial_state: Dict[str, Any] = {
        "messages": messages,
        "job_id": req.job_id,
        "run_id": req.run_id,
        "session_id": session_id,
        "request_id": request_id,
        "user_id": req.user_id,
        "execution_id": req.execution_id,
        "utility_id": req.utility_id,
        "utility_name": req.utility_name,
        "index_id": index_id,
        "top_k": req.top_k,
        "auto_suggest": req.auto_suggest,
        "input_paths": document_catalog,
        "document_catalog": document_catalog,
        "document_quick_summaries": [],
        "document_context_brief": None,
        "document_summary": None,
        "user_query": req.user_query or "",
        "current_step": "",
        "query_intent": None,
        "query_for_retrieval": None,
        "thoughts_summary": None,
        "short_circuit": False,
        "intent_relevant": True,
        "intent_reason": None,
        "needs_retrieval": True,
        "target_file_names": [],
        "matched_file_ids": [],
        "matched_file_names": [],
        "retrieval_scope": "global",
        "needs_sql": False,
        "reuse_context_hit": False,
        "reuse_context_reason": None,
        "retrieved_chunks": [],
        "auto_suggestion_output": None,
        "answer": None,
        "sources": [],
        "completed_steps": [],
        "errors": [],
    }

    config = {"configurable": {"thread_id": session_id}}
    store.upsert_run(req.job_id, req.run_id, session_id, "in_progress")

    async def event_stream():
        try:
            merged_values: Dict[str, Any] = dict(initial_state)
            yield f"event: custom\ndata: {json.dumps({'type': 'heartbeat', 'step': 'initializing'})}\n\n"
            async for mode, chunk in workflow.astream(
                initial_state,
                config,
                stream_mode=["updates", "custom"],
            ):
                if mode == "custom":
                    event_type = chunk.get("type", "step")
                    store.log_event(
                        session_id=session_id,
                        request_id=request_id,
                        event_type=event_type,
                        event_data=chunk,
                    )
                    yield f"event: {event_type}\ndata: {json.dumps(chunk)}\n\n"
                elif mode == "updates":
                    for node_name, node_update in chunk.items():
                        if isinstance(node_update, dict):
                            merged_values.update(node_update)
                        event = {"node": node_name}
                        store.log_event(
                            session_id=session_id,
                            request_id=request_id,
                            event_type="node_complete",
                            event_data=event,
                        )
                        yield f"event: node_complete\ndata: {json.dumps(event)}\n\n"

            try:
                final_state = await workflow.aget_state(config)
                values = final_state.values if hasattr(final_state, "values") else {}
            except ValueError as e:
                if "No checkpointer set" not in str(e):
                    raise
                # Workflow is compiled without checkpointer; use streamed updates.
                values = merged_values
            done_payload = _done_payload(values)
            all_messages = values.get("messages") or messages
            store.save_checkpoint(
                thread_id=session_id,
                messages_json=_serialize_messages(all_messages),
                state_json=json.dumps(done_payload),
            )
            store.upsert_run(req.job_id, req.run_id, session_id, "completed")
            yield f"event: done\ndata: {json.dumps(done_payload)}\n\n"
        except Exception as e:
            logger.exception("chat stream failed")
            store.upsert_run(req.job_id, req.run_id, session_id, "failed")
            err = {
                "type": "error",
                "message": f"Processing failed: {e}",
                "session_id": session_id,
            }
            yield f"event: error\ndata: {json.dumps(err)}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )

