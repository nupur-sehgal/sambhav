from typing import TypedDict, Any
from langgraph.graph import StateGraph, START, END
from rag import answer_with_rag
from query_generator import run_readonly_query


class ChatState(TypedDict, total=False):
    question: str
    route: str
    answer: str
    sources: list[dict]
    pipeline: list[dict]
    results: list[dict]
    error: str


def route_question(state: ChatState) -> dict:
    """Choose semantic RAG or a structured database query."""
    question = state["question"]
    # Simple baseline router: aggregate/count/list/filter questions are structured.
    # In a later version, replace this with an LLM classifier.
    structured_terms = (
        "how many", "count", "total", "average", "sum", "maximum", "minimum",
        "list all", "show all", "find all", "sort by", "group by",
        "which has the highest", "how much", "number of"
    )
    lowered = question.lower()
    route = "database" if any(term in lowered for term in structured_terms) else "rag"
    return {"route": route}


def rag_node(state: ChatState) -> dict:
    result = answer_with_rag(state["question"])
    return {"answer": result["answer"], "sources": result["sources"]}


def database_node(state: ChatState) -> dict:
    result = run_readonly_query(state["question"])
    return {
        "answer": "Database query completed. Review the structured results below.",
        "pipeline": result["pipeline"],
        "results": result["results"],
    }


def error_node(state: ChatState) -> dict:
    return {"answer": "I couldn't complete that request safely.", "error": state.get("error", "Unknown error")}


def build_graph():
    graph = StateGraph(ChatState)
    graph.add_node("route", route_question)
    graph.add_node("rag", rag_node)
    graph.add_node("database", database_node)

    graph.add_edge(START, "route")
    graph.add_conditional_edges(
        "route",
        lambda state: state["route"],
        {"rag": "rag", "database": "database"},
    )
    graph.add_edge("rag", END)
    graph.add_edge("database", END)
    return graph.compile()


chat_graph = build_graph()


def ask(question: str) -> dict[str, Any]:
    try:
        return chat_graph.invoke({"question": question})
    except Exception as exc:
        # Keep details for local debugging; don't show secrets/connection strings.
        return {
            "question": question,
            "answer": "The request failed. Check your terminal for configuration or query errors.",
            "error": f"{type(exc).__name__}: {exc}",
        }