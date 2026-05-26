from typing import Literal

from langgraph.graph import END, StateGraph

from app.workflow.chat_nodes import (
    auto_suggest_node,
    generate_answer_node,
    intent_check_node,
    request_routing_node,
    reuse_context_node,
    search_and_retrieve_node,
)
from app.workflow.chat_state import TableLineItemChatState


def router(state: TableLineItemChatState) -> Literal["auto_suggest", "intent_check"]:
    if state.get("auto_suggest"):
        return "auto_suggest"
    return "intent_check"


def intent_router(state: TableLineItemChatState) -> Literal["reuse_context", "finish"]:
    if state.get("short_circuit"):
        return "finish"
    if state.get("intent_relevant", True):
        return "reuse_context"
    return "finish"


def reuse_context_router(state: TableLineItemChatState) -> Literal["search_and_retrieve", "finish"]:
    if state.get("needs_retrieval", False):
        return "search_and_retrieve"
    if state.get("reuse_context_hit", False):
        return "finish"
    return "search_and_retrieve"


def build_chat_workflow():
    graph = StateGraph(TableLineItemChatState)
    graph.add_node("request_routing", request_routing_node)
    graph.add_node("auto_suggest", auto_suggest_node)
    graph.add_node("intent_check", intent_check_node)
    graph.add_node("reuse_context", reuse_context_node)
    graph.add_node("search_and_retrieve", search_and_retrieve_node)
    graph.add_node("generate_answer", generate_answer_node)

    graph.set_entry_point("request_routing")
    graph.add_conditional_edges(
        "request_routing",
        router,
        {
            "auto_suggest": "auto_suggest",
            "intent_check": "intent_check",
        },
    )
    graph.add_edge("auto_suggest", END)
    graph.add_conditional_edges(
        "intent_check",
        intent_router,
        {
            "reuse_context": "reuse_context",
            "finish": END,
        },
    )
    graph.add_conditional_edges(
        "reuse_context",
        reuse_context_router,
        {
            "search_and_retrieve": "search_and_retrieve",
            "finish": END,
        },
    )
    graph.add_edge("search_and_retrieve", "generate_answer")
    graph.add_edge("generate_answer", END)
    return graph.compile()

