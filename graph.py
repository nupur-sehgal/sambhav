import json
import logging
import os
import re
from typing import Any, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from pymongo.errors import OperationFailure, ServerSelectionTimeoutError

from db import get_collection_schema_dict
from llm import get_llm, response_text
from query_generator import (
    MAX_ATTEMPTS,
    MAX_LIMIT,
    execute_pipeline,
    generate_mongodb_pipeline,
)
from rag import answer_with_rag

logger = logging.getLogger(__name__)

MAX_HISTORY_MESSAGES = 6
MAX_ROWS_FOR_ANSWER = 30
MAX_RESULT_CHARS = 6000

# The free Gemini tier allows only a few requests per minute, so database
# answers are templated by default. Set LLM_DB_ANSWER=true to have Gemini
# write them instead (costs one extra request per database question).
USE_LLM_ANSWER = os.getenv("LLM_DB_ANSWER", "false").strip().lower() == "true"


# ==================================================
# 1. State shared by all nodes
# ==================================================

class ChatState(TypedDict, total=False):
    question: str            # what the user typed
    history: list[dict]      # earlier turns: [{"role": ..., "content": ...}]
    standalone: str          # the question rewritten to make sense on its own
    route: str               # "rag" or "database"
    answer: str
    sources: list[dict]      # RAG route
    pipeline: list[dict] | None   # database route
    results: list[dict]           # database route
    attempts: int
    query_error: str | None
    error: str               # developer-facing detail


# ==================================================
# 2. Prompts
# ==================================================

REWRITE_SYSTEM = """Rewrite the user's latest message as a standalone question.

Use the conversation only to fill in what the message leaves out (for example
"and for 2023?" or "what about the first one?"). If the message already makes
sense on its own, return it unchanged. Reply with the question only."""

ROUTE_SYSTEM = """You route questions for a chatbot over a MongoDB collection.

Fields in the collection: __FIELDS__

Reply with exactly one word:
- database: the question needs counting, filtering, listing, sorting or calculating over records (for example "how many ...", "list all ... where ...", "average ...", "top 5 ...").
- documents: the question asks for an explanation, definition or summary that is answered by reading the text of documents."""

ANSWER_SYSTEM = """You answer questions using database query results.

Use only the results provided. Be concise and plain; do not mention MongoDB,
pipelines or JSON. If only some rows are shown out of a larger total, say so."""


# ==================================================
# 3. Nodes
# ==================================================

FOLLOWUP_CUES = re.compile(
    r"\b(it|its|they|them|their|those|these|that|this|he|she|his|her|same|above|"
    r"previous|earlier|also|too|instead|again|else)\b"
    r"|^\s*(and|but|then|so|what about|how about)\b",
    re.IGNORECASE,
)


def _looks_like_followup(question: str) -> bool:
    """Short messages and pronouns suggest the question needs the earlier context."""
    return len(question.split()) < 5 or bool(FOLLOWUP_CUES.search(question))


def rewrite_node(state: ChatState) -> dict:
    """Turn a follow-up into a standalone question (skipped when not needed)."""
    question = state["question"].strip()
    history = (state.get("history") or [])[-MAX_HISTORY_MESSAGES:]

    # No history, or a message that already stands on its own: save a model call.
    if not history or not _looks_like_followup(question):
        return {"standalone": question}

    transcript = "\n".join(
        f"{message.get('role', 'user')}: {str(message.get('content', ''))[:500]}"
        for message in history
    )

    response = get_llm().invoke([
        SystemMessage(content=REWRITE_SYSTEM),
        HumanMessage(
            content=f"Conversation:\n{transcript}\n\nLatest message: {question}"
        ),
    ])

    rewritten = response_text(response).strip().strip('"')

    # Fall back to the original if the model rambled or returned nothing.
    if not rewritten or len(rewritten) > 500:
        rewritten = question

    return {"standalone": rewritten}


# Wording that suggests counting, filtering, listing or calculating...
DATABASE_SIGNALS = re.compile(
    r"\b(how many|number of|count|total|sum of|average|maximum|minimum|highest|"
    r"lowest|most|least|fewest|top \d+|list|show|find|group(ed)? by|sort(ed)? by|"
    r"where|for each)\b"
)
# ...and wording that suggests an explanation or definition.
DOCUMENT_SIGNALS = re.compile(
    r"^\s*(what|who|why)\b|^\s*how (does|do|did|can|is|are)\b|"
    r"^\s*(explain|define|describe|summari[sz]e|compare|tell me about)\b|"
    r"\bdifference between\b"
)
RECORD_WORDS = re.compile(r"\b(documents?|records?|rows?|entries|entry|collection|database)\b")


def _mentions_data(question: str, fields: list[str]) -> bool:
    """Does the question talk about the records or one of the collection's fields?"""
    if RECORD_WORDS.search(question):
        return True

    words = set(re.findall(r"[a-z_]+", question))
    for field in fields:
        for part in field.lower().split("."):
            if part in words or part + "s" in words:
                return True
    return False


def route_by_rules(question: str, fields: list[str]) -> str | None:
    """
    Route obvious questions without calling the model.

    A counting/listing word only counts when the question also mentions the
    data ("how many documents", "list all titles"), so "how many layers does
    the OSI model have?" is not mistaken for a database question.
    Returns "database", "rag", or None when the question is ambiguous.
    """
    text = question.lower()
    mentions_data = _mentions_data(text, fields)

    if mentions_data and DATABASE_SIGNALS.search(text):
        return "database"
    if not mentions_data and DOCUMENT_SIGNALS.search(text):
        return "rag"
    return None


def route_node(state: ChatState) -> dict:
    """
    Decide between a database query and a document search.

    Clear-cut questions are routed by rules (no model call, which matters on
    the free tier's requests-per-minute limit); only ambiguous ones ask Gemini.
    """
    schema_fields = sorted(get_collection_schema_dict())

    rule_route = route_by_rules(state["standalone"], schema_fields)
    if rule_route:
        return {"route": rule_route}

    fields = ", ".join(schema_fields) or "(unknown)"

    response = get_llm().invoke([
        SystemMessage(content=ROUTE_SYSTEM.replace("__FIELDS__", fields)),
        HumanMessage(content=state["standalone"]),
    ])

    words = response_text(response).strip().lower().split()
    first_word = re.sub(r"[^a-z]", "", words[0]) if words else ""

    return {"route": "database" if first_word == "database" else "rag"}


def rag_node(state: ChatState) -> dict:
    result = answer_with_rag(state["standalone"])
    return {"answer": result["answer"], "sources": result["sources"]}


def generate_query_node(state: ChatState) -> dict:
    """Ask Gemini for a pipeline; a rejected pipeline sets query_error."""
    attempts = state.get("attempts", 0) + 1

    try:
        pipeline = generate_mongodb_pipeline(
            state["standalone"],
            previous_error=state.get("query_error"),
        )
    except ValueError as exc:
        return {"pipeline": None, "attempts": attempts, "query_error": str(exc)}

    return {"pipeline": pipeline, "attempts": attempts, "query_error": None}


def run_query_node(state: ChatState) -> dict:
    """Run the validated pipeline; a MongoDB failure sets query_error."""
    try:
        results = execute_pipeline(state["pipeline"])
    except OperationFailure as exc:
        return {"query_error": str(exc)}

    return {"results": results, "query_error": None}


def summarize_results(results: list[dict]) -> str:
    """A plain answer built without a model call."""
    if not results:
        return "No records matched that question."

    if len(results) == 1:
        items = [(key, value) for key, value in results[0].items() if key != "_id"]
        if items and len(items) <= 4:
            return "Result: " + ", ".join(f"{key} = {value}" for key, value in items)
        return "Found 1 result. See the table below."

    note = f" (showing the first {MAX_LIMIT})" if len(results) >= MAX_LIMIT else ""
    return f"Found {len(results)} results{note}. See the table below."


def answer_db_node(state: ChatState) -> dict:
    """Turn raw query results into an answer (templated unless LLM_DB_ANSWER=true)."""
    results = state.get("results") or []

    if not results or not USE_LLM_ANSWER:
        return {"answer": summarize_results(results)}

    shown = json.dumps(results[:MAX_ROWS_FOR_ANSWER], default=str)[:MAX_RESULT_CHARS]

    response = get_llm().invoke([
        SystemMessage(content=ANSWER_SYSTEM),
        HumanMessage(
            content=(
                f"Question: {state['standalone']}\n\n"
                f"Query results ({len(results)} row(s) in total, "
                f"first {min(len(results), MAX_ROWS_FOR_ANSWER)} shown):\n{shown}"
            )
        ),
    ])

    return {"answer": response_text(response).strip() or "The query returned results, but I could not summarise them."}


def fail_node(state: ChatState) -> dict:
    return {
        "answer": (
            "I couldn't turn that into a working database query. "
            "Try rephrasing it, or mention the specific fields you care about."
        ),
        "error": state.get("query_error") or "Query generation failed.",
        "results": [],
    }


# ==================================================
# 4. Routing between nodes
# ==================================================

def _retry_or_fail(state: ChatState) -> str:
    return "retry" if state.get("attempts", 0) < MAX_ATTEMPTS else "fail"


def after_generate(state: ChatState) -> str:
    return _retry_or_fail(state) if state.get("query_error") else "run"


def after_run(state: ChatState) -> str:
    return _retry_or_fail(state) if state.get("query_error") else "answer"


def build_graph():
    graph = StateGraph(ChatState)

    graph.add_node("rewrite", rewrite_node)
    graph.add_node("route", route_node)
    graph.add_node("rag", rag_node)
    graph.add_node("generate_query", generate_query_node)
    graph.add_node("run_query", run_query_node)
    graph.add_node("answer_db", answer_db_node)
    graph.add_node("fail", fail_node)

    graph.add_edge(START, "rewrite")
    graph.add_edge("rewrite", "route")
    graph.add_conditional_edges(
        "route",
        lambda state: state["route"],
        {"rag": "rag", "database": "generate_query"},
    )
    graph.add_conditional_edges(
        "generate_query",
        after_generate,
        {"run": "run_query", "retry": "generate_query", "fail": "fail"},
    )
    graph.add_conditional_edges(
        "run_query",
        after_run,
        {"answer": "answer_db", "retry": "generate_query", "fail": "fail"},
    )
    graph.add_edge("rag", END)
    graph.add_edge("answer_db", END)
    graph.add_edge("fail", END)

    return graph.compile()


chat_graph = build_graph()


# ==================================================
# 5. Entry point used by the app
# ==================================================

def friendly_message(exc: Exception) -> str:
    """A safe, readable explanation; full details go to the terminal log."""
    text = f"{type(exc).__name__} {exc}"

    if isinstance(exc, ServerSelectionTimeoutError):
        return (
            "I can't reach the database. Check your internet connection and "
            "that your IP address is allowed in Atlas (Network Access)."
        )
    if "GOOGLE_API_KEY" in text:
        return "The Gemini API key is missing. Add GOOGLE_API_KEY to your .env file."
    if "429" in text or "RESOURCE_EXHAUSTED" in text:
        if "PerDay" in text:
            return (
                "Today's free-tier quota for this model is used up. Try a different "
                "GEMINI_MODEL in .env, or wait until tomorrow."
            )
        wait = re.search(r"retry in ([\d.]+)s", text)
        when = f"about {round(float(wait.group(1)))} seconds" if wait else "a minute"
        return (
            "The free Gemini tier allows only a few requests per minute and that "
            f"limit was reached. Try again in {when}."
        )
    if "504" in text or "DEADLINE_EXCEEDED" in text or "timed out" in text.lower():
        return "The AI service took too long to respond. Try again, or try a different GEMINI_MODEL in .env."
    if "vectorSearch" in text:
        return (
            "Vector search failed. Check that the Atlas Vector Search index "
            "exists on the 'embedding' field."
        )

    return "Something went wrong while processing your question. Check the terminal for details."


def ask(question: str, history: list[dict] | None = None) -> dict[str, Any]:
    """Run the graph. Never raises: failures come back as a friendly answer."""
    try:
        return chat_graph.invoke({"question": question, "history": history or []})
    except Exception as exc:
        logger.exception("ask() failed")
        return {
            "question": question,
            "answer": friendly_message(exc),
            "error": f"{type(exc).__name__}: {exc}",
        }
