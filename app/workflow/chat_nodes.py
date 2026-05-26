import json
from datetime import datetime
from typing import Any, Dict, List, Optional

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.config import get_stream_writer

from app.workflow.chat_prompts import (
    ANSWER_SYSTEM_PROMPT,
    ANSWER_USER_PROMPT,
    AUTO_SUGGEST_SYSTEM_PROMPT,
    AUTO_SUGGEST_USER_PROMPT,
    INTENT_SYSTEM_PROMPT,
    INTENT_USER_PROMPT,
    REUSE_CONTEXT_SYSTEM_PROMPT,
    REUSE_CONTEXT_USER_PROMPT,
    SQL_ANSWER_SYSTEM_PROMPT,
    SQL_ANSWER_USER_PROMPT,
    SQL_GENERATION_SYSTEM_PROMPT,
    SQL_GENERATION_USER_PROMPT,
)
from app.workflow.chat_state import TableLineItemChatState
from app.workflow.runtime import get_runtime
from app.workflow.search_client import (
    build_index_id,
    fetch_neighbour_chunks,
    search_knowledge_base,
)
from app.workflow.table_sql_pipeline import (
    build_schema_description,
    execute_sql_on_sheets,
    is_spreadsheet_file,
    load_spreadsheet_tables,
)


def _json_or_default(value: Any, default: Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            obj = json.loads(value)
            if isinstance(obj, dict):
                return obj
        except Exception:
            return default
    return default


def _document_list_for_prompt(document_catalog: List[Dict[str, Any]]) -> str:
    if not document_catalog:
        return "(none)"
    lines: List[str] = []
    for i, item in enumerate(document_catalog, start=1):
        lines.append(f"{i}. {item.get('file_name')} (file_id: {item.get('file_id')})")
    return "\n".join(lines)


def _conversation_context(messages: List[Any]) -> str:
    if not messages:
        return ""
    lines: List[str] = []
    for msg in messages[-10:]:
        role = "User" if isinstance(msg, HumanMessage) else "Assistant"
        lines.append(f"{role}: {msg.content}")
    return "\n".join(lines)


def request_routing_node(state: TableLineItemChatState) -> Dict[str, Any]:
    rt = get_runtime()
    writer = get_stream_writer()
    writer({"type": "step", "step": "routing", "message": "Analyzing request..."})
    index_id = state.get("index_id") or build_index_id(state.get("job_id", ""), state.get("run_id", ""))
    out = {
        "index_id": index_id,
        "top_k": min(max(int(state.get("top_k") or rt.settings.default_top_k), 1), rt.settings.max_top_k),
        "completed_steps": list(state.get("completed_steps", [])) + ["routing"],
        "current_step": "routing",
        "short_circuit": False,
    }
    rt.store.log_event(
        session_id=state.get("session_id", ""),
        request_id=state.get("request_id"),
        event_type="routing",
        event_data={"index_id": index_id},
    )
    return out


async def auto_suggest_node(state: TableLineItemChatState) -> Dict[str, Any]:
    rt = get_runtime()
    writer = get_stream_writer()
    writer({"type": "step", "step": "suggesting", "message": "Generating suggested questions..."})

    docs = state.get("document_catalog") or []
    summaries: List[str] = []
    for d in docs[:6]:
        summaries.append(f"- {d.get('file_name')}")
    prompt = AUTO_SUGGEST_USER_PROMPT.format(document_context="\n".join(summaries) or "(none)")
    response = await rt.bedrock.invoke(
        {
            "model": rt.settings.bedrock_summary_model,
            "system_prompt": AUTO_SUGGEST_SYSTEM_PROMPT,
            "user_prompt": prompt,
            "temperature": 0.0,
            "max_tokens": 600,
        }
    )
    parsed = _json_or_default(response.get("response"), {"queries": []})
    queries = parsed.get("queries")
    if not isinstance(queries, list):
        queries = []
    out = {
        "auto_suggestion_output": {
            "auto_suggested_queries": [str(q) for q in queries[:4]],
        },
        "completed_steps": list(state.get("completed_steps", [])) + ["auto_suggest"],
        "current_step": "suggesting",
    }
    rt.store.log_event(
        session_id=state.get("session_id", ""),
        request_id=state.get("request_id"),
        event_type="auto_suggest",
        event_data=out["auto_suggestion_output"],
    )
    return out


async def intent_check_node(state: TableLineItemChatState) -> Dict[str, Any]:
    rt = get_runtime()
    writer = get_stream_writer()
    writer({"type": "step", "step": "checking_intent", "message": "Checking query relevance..."})
    user_query = state.get("user_query", "")
    document_catalog = state.get("document_catalog") or []
    response = await rt.bedrock.invoke(
        {
            "model": rt.settings.bedrock_summary_model,
            "system_prompt": INTENT_SYSTEM_PROMPT,
            "user_prompt": INTENT_USER_PROMPT.format(
                document_list=_document_list_for_prompt(document_catalog),
                query=user_query,
            ),
            "temperature": 0.0,
            "max_tokens": 500,
        }
    )
    parsed = _json_or_default(
        response.get("response"),
        {
            "intent_relevant": True,
            "needs_retrieval": True,
            "needs_sql": False,
            "query_intent": "content_qa",
            "query_for_retrieval": user_query,
            "direct_answer": "",
            "reason": "default fallback",
            "target_file_names": [],
        },
    )
    target_names = parsed.get("target_file_names")
    if not isinstance(target_names, list):
        target_names = []

    matched_file_ids: List[str] = []
    matched_file_names: List[str] = []
    lower_targets = {str(x).strip().lower() for x in target_names}
    for d in document_catalog:
        name = str(d.get("file_name", "")).lower()
        if name in lower_targets:
            matched_file_ids.append(str(d.get("file_id")))
            matched_file_names.append(str(d.get("file_name")))

    query_intent = str(parsed.get("query_intent", "content_qa"))
    direct_answer = str(parsed.get("direct_answer", ""))
    needs_retrieval = bool(parsed.get("needs_retrieval", True))
    short_circuit = bool(direct_answer) and not needs_retrieval and query_intent in {"list_documents", "count_documents"}

    if not bool(parsed.get("intent_relevant", True)):
        decline = "I can help with questions about your uploaded documents."
        return {
            "intent_relevant": False,
            "short_circuit": True,
            "answer": decline,
            "messages": [AIMessage(content=decline)],
            "completed_steps": list(state.get("completed_steps", [])) + ["intent_check"],
            "current_step": "checking_intent",
        }

    out = {
        "query_intent": query_intent,
        "intent_relevant": True,
        "needs_retrieval": needs_retrieval,
        "needs_sql": bool(parsed.get("needs_sql", False)),
        "query_for_retrieval": str(parsed.get("query_for_retrieval", user_query)),
        "intent_reason": str(parsed.get("reason", "")),
        "short_circuit": short_circuit,
        "target_file_names": [str(x) for x in target_names],
        "matched_file_ids": matched_file_ids,
        "matched_file_names": matched_file_names,
        "retrieval_scope": "narrowed" if matched_file_ids else "global",
        "completed_steps": list(state.get("completed_steps", [])) + ["intent_check"],
        "current_step": "checking_intent",
    }
    if short_circuit:
        out["answer"] = direct_answer
        out["messages"] = [AIMessage(content=direct_answer)]
    return out


async def reuse_context_node(state: TableLineItemChatState) -> Dict[str, Any]:
    rt = get_runtime()
    writer = get_stream_writer()
    writer({"type": "step", "step": "reuse_context", "message": "Checking prior conversation..."})
    conv = _conversation_context(state.get("messages") or [])
    if not conv:
        return {
            "reuse_context_hit": False,
            "reuse_context_reason": "No prior conversation context.",
            "completed_steps": list(state.get("completed_steps", [])) + ["reuse_context"],
            "current_step": "reuse_context",
        }
    response = await rt.bedrock.invoke(
        {
            "model": rt.settings.bedrock_summary_model,
            "system_prompt": REUSE_CONTEXT_SYSTEM_PROMPT,
            "user_prompt": REUSE_CONTEXT_USER_PROMPT.format(
                conv_context=conv,
                query=state.get("user_query", ""),
            ),
            "temperature": 0.0,
            "max_tokens": 400,
        }
    )
    parsed = _json_or_default(
        response.get("response"),
        {"answerable": False, "answer": "", "reason": "fallback"},
    )
    hit = bool(parsed.get("answerable")) and bool(str(parsed.get("answer", "")).strip())
    out = {
        "reuse_context_hit": hit,
        "reuse_context_reason": str(parsed.get("reason", "")),
        "completed_steps": list(state.get("completed_steps", [])) + ["reuse_context"],
        "current_step": "reuse_context",
    }
    if hit:
        out["answer"] = str(parsed.get("answer"))
        out["messages"] = [AIMessage(content=str(parsed.get("answer")))]
    return out


async def search_and_retrieve_node(state: TableLineItemChatState) -> Dict[str, Any]:
    rt = get_runtime()
    writer = get_stream_writer()
    writer({"type": "step", "step": "searching", "message": "Retrieving relevant document chunks..."})
    index_id = state.get("index_id", "")
    query = state.get("query_for_retrieval") or state.get("user_query", "")
    top_k = int(state.get("top_k") or rt.settings.default_top_k)
    retrieval_scope = state.get("retrieval_scope") or "global"
    matched_file_ids = state.get("matched_file_ids") or []
    results: List[Dict[str, Any]] = []

    if retrieval_scope == "narrowed" and matched_file_ids:
        merged: List[Dict[str, Any]] = []
        for file_id in matched_file_ids:
            merged.extend(
                await search_knowledge_base(
                    store=rt.store,
                    bedrock=rt.bedrock,
                    index_id=index_id,
                    query=query,
                    top_k=max(top_k, 3),
                    document_id=file_id,
                )
            )
        merged.sort(key=lambda x: float(x.get("score") or 0), reverse=True)
        results = merged[:top_k]
    else:
        results = await search_knowledge_base(
            store=rt.store,
            bedrock=rt.bedrock,
            index_id=index_id,
            query=query,
            top_k=top_k,
            document_id=None,
        )

    results = await fetch_neighbour_chunks(
        store=rt.store,
        index_id=index_id,
        retrieved=results,
        window=1,
    )
    sources: List[Dict[str, Any]] = []
    seen = set()
    for r in results:
        doc_id = r.get("document_id")
        if doc_id in seen:
            continue
        seen.add(doc_id)
        sources.append(
            {
                "file_name": r.get("file_name"),
                "document_id": doc_id,
                "score": r.get("score"),
            }
        )
    rt.store.log_retrieval_query(
        session_id=state.get("session_id", ""),
        index_id=index_id,
        query_text=query,
        top_k=top_k,
        retrieval_scope=retrieval_scope,
        matched_files=state.get("matched_file_names") or [],
    )
    return {
        "retrieved_chunks": results,
        "sources": sources,
        "completed_steps": list(state.get("completed_steps", [])) + ["search_and_retrieve"],
        "current_step": "searching",
    }


async def _generate_sql_answer(state: TableLineItemChatState) -> Optional[str]:
    rt = get_runtime()
    matched = state.get("matched_file_names") or []
    catalog = state.get("document_catalog") or []
    spreadsheets = [
        d for d in catalog
        if d.get("file_name") in matched and is_spreadsheet_file(str(d.get("file_name")))
    ]
    if not spreadsheets:
        spreadsheets = [d for d in catalog if is_spreadsheet_file(str(d.get("file_name")))]
    if not spreadsheets:
        return None
    doc = spreadsheets[0]
    file_path = str(doc.get("source"))
    file_name = str(doc.get("file_name"))
    tables, err = await load_spreadsheet_tables(file_path, file_name)
    if err or not tables:
        return f"Unable to load spreadsheet for SQL analysis: {err}"
    schema = build_schema_description(tables)
    sql_response = await rt.bedrock.invoke(
        {
            "model": rt.settings.bedrock_chat_model,
            "system_prompt": SQL_GENERATION_SYSTEM_PROMPT,
            "user_prompt": SQL_GENERATION_USER_PROMPT.format(
                schema=schema,
                query=state.get("user_query", ""),
            ),
            "temperature": 0.0,
            "max_tokens": 350,
        }
    )
    sql_json = _json_or_default(sql_response.get("response"), {"sql": ""})
    sql = str(sql_json.get("sql", "")).strip()
    if not sql:
        return None
    rows, sql_err = execute_sql_on_sheets(tables, sql)
    if sql_err:
        return f"SQL execution failed: {sql_err}"
    answer = await rt.bedrock.invoke(
        {
            "model": rt.settings.bedrock_chat_model,
            "system_prompt": SQL_ANSWER_SYSTEM_PROMPT,
            "user_prompt": SQL_ANSWER_USER_PROMPT.format(
                query=state.get("user_query", ""),
                sql=sql,
                rows=json.dumps(rows[:50], default=str),
            ),
            "temperature": 0.0,
            "max_tokens": 700,
        }
    )
    raw = answer.get("response")
    text = raw if isinstance(raw, str) else json.dumps(raw, default=str)
    return f"{text}\n\n<details><summary>SQL used</summary>\n\n```sql\n{sql}\n```\n</details>"


async def generate_answer_node(state: TableLineItemChatState) -> Dict[str, Any]:
    rt = get_runtime()
    writer = get_stream_writer()
    writer({"type": "step", "step": "generating", "message": "Generating answer..."})

    sql_answer = None
    if state.get("needs_sql"):
        sql_answer = await _generate_sql_answer(state)

    if sql_answer:
        writer({"type": "token", "content": sql_answer})
        return {
            "answer": sql_answer,
            "messages": [AIMessage(content=sql_answer)],
            "completed_steps": list(state.get("completed_steps", [])) + ["generate_answer"],
            "current_step": "generating",
        }

    context_blocks: List[str] = []
    for chunk in state.get("retrieved_chunks") or []:
        context_blocks.append(
            f"[Source: {chunk.get('file_name')}]\n{chunk.get('chunk_text')}"
        )
    context = "\n\n---\n\n".join(context_blocks) or "No relevant context found."
    conv = _conversation_context(state.get("messages") or [])
    user_prompt = ANSWER_USER_PROMPT.format(
        context=context + ("\n\nPrior conversation:\n" + conv if conv else ""),
        query=state.get("user_query", ""),
    )
    full_answer = ""
    async for token in rt.bedrock.stream_invoke(
        {
            "model": rt.settings.bedrock_chat_model,
            "system_prompt": ANSWER_SYSTEM_PROMPT + f"\nCurrent time: {datetime.utcnow().isoformat()}",
            "user_prompt": user_prompt,
            "temperature": 0.0,
            "max_tokens": 2048,
        }
    ):
        full_answer += token
        writer({"type": "token", "content": token})
    return {
        "answer": full_answer,
        "messages": [AIMessage(content=full_answer)],
        "completed_steps": list(state.get("completed_steps", [])) + ["generate_answer"],
        "current_step": "generating",
    }

