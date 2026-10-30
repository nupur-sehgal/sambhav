import os

import streamlit as st

# --------------------------------------------------
# 1. Page setup (must be the first Streamlit call)
# --------------------------------------------------

st.set_page_config(
    page_title="Sambhav MongoDB RAG Chatbot",
    page_icon="💬",
    layout="wide",
)

st.title("💬 Sambhav — MongoDB RAG Chatbot")
st.caption("Ask questions about your MongoDB data in plain English")

# Importing these reads .env and builds the graph. A setup mistake (for example
# a missing MONGODB_URI) is shown on the page instead of as a raw traceback.
try:
    from db import collection, get_collection_names, ping
    from graph import ask
except Exception as exc:
    st.error(f"Setup problem: {exc}")
    st.stop()

# Set SHOW_DEBUG=true in .env while developing to see error details on the page.
SHOW_DEBUG = os.getenv("SHOW_DEBUG", "false").strip().lower() == "true"

EXAMPLE_QUESTIONS = [
    "What is a DBMS?",
    "Explain primary keys.",
    "How many documents are in this collection?",
]

HISTORY_MESSAGES = 6  # earlier messages sent to the graph for follow-up questions


# --------------------------------------------------
# 2. Helpers
# --------------------------------------------------

@st.cache_data(ttl=60, show_spinner=False)
def get_db_status() -> dict:
    """Database status, cached for 60 seconds so reruns stay fast."""
    try:
        ping()
        return {
            "ok": True,
            "database": collection.database.name,
            "collection": collection.name,
            "collections": get_collection_names(),
            "documents": collection.estimated_document_count(),
        }
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def build_history(messages: list[dict]) -> list[dict]:
    """The last few turns, as plain role/content pairs."""
    return [
        {"role": message["role"], "content": message.get("content", "")}
        for message in messages[-HISTORY_MESSAGES:]
    ]


def render_sources(sources):
    """Show retrieved documents (RAG route)."""
    if not isinstance(sources, list) or not sources:
        return

    with st.expander(f"📚 Retrieved sources ({len(sources)})"):
        for index, source in enumerate(sources, start=1):
            if not isinstance(source, dict):
                st.write(f"[{index}] {source}")
                continue

            title = source.get("title") or source.get("name") or f"Source {index}"
            st.markdown(f"**[{index}] {title}**")

            body = source.get("content") or source.get("text")
            if body:
                st.write(body)
            else:
                st.json({
                    key: value
                    for key, value in source.items()
                    if key not in {"_id", "vector_score"}
                })

            score = source.get("vector_score")
            if isinstance(score, (int, float)):
                st.caption(f"Relevance score: {score:.3f}")

            if index < len(sources):
                st.divider()


def render_query_details(message: dict):
    """Show the pipeline and result table (database route only)."""
    if message.get("route") != "database":
        return

    pipeline = message.get("pipeline")
    if pipeline:
        with st.expander("🔍 Generated MongoDB pipeline"):
            st.json(pipeline)

    results = message.get("results")
    if results:
        with st.expander(f"🗃️ Database results ({len(results)} row(s))", expanded=True):
            try:
                st.dataframe(results, width="stretch")
            except Exception:
                st.json(results)


def render_assistant_message(message: dict):
    st.markdown(message.get("content") or "No answer returned.")

    route = message.get("route")
    if route:
        st.caption(f"Route: `{route}`")

    render_query_details(message)

    if route == "rag":
        render_sources(message.get("sources"))

    if SHOW_DEBUG and message.get("debug_error"):
        with st.expander("Developer error details"):
            st.code(message["debug_error"])


# --------------------------------------------------
# 3. Sidebar: database status and example questions
# --------------------------------------------------

with st.sidebar:
    st.header("Project status")

    status = get_db_status()

    if status["ok"]:
        st.success("MongoDB connected")
        st.write("Database:", status["database"])
        st.write("Collection:", status["collection"])
        st.write(
            "Collections:",
            ", ".join(status["collections"]) if status["collections"] else "None found",
        )
        st.metric("Documents in collection", status["documents"])
    else:
        st.error("Could not connect to MongoDB.")
        st.caption(
            "Check your .env settings, your internet connection, and that your "
            "IP address is allowed in Atlas (Network Access)."
        )
        if SHOW_DEBUG:
            st.code(status["error"])

    if st.button("Recheck connection"):
        get_db_status.clear()
        st.rerun()

    st.divider()
    st.subheader("Try asking")

    for example in EXAMPLE_QUESTIONS:
        if st.button(example, key=f"example-{example}", width="stretch"):
            st.session_state.queued_question = example

    st.divider()

    if st.button("Clear chat", width="stretch"):
        st.session_state.messages = []
        st.rerun()


# --------------------------------------------------
# 4. Chat history
# --------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            render_assistant_message(message)
        else:
            st.markdown(message.get("content", ""))


# --------------------------------------------------
# 5. Handle a new question
# --------------------------------------------------

question = st.chat_input("Ask a question about your MongoDB data...")

# An example button queues its text so it is handled like a typed question.
if not question:
    question = st.session_state.pop("queued_question", None)

if question:
    history = build_history(st.session_state.messages)  # before adding this turn

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = ask(question, history=history)  # never raises

        assistant_message = {
            "role": "assistant",
            "content": str(result.get("answer") or "No answer returned."),
            "route": result.get("route"),
            "pipeline": result.get("pipeline"),
            "results": result.get("results"),
            "sources": result.get("sources"),
            "debug_error": result.get("error"),
        }

        render_assistant_message(assistant_message)

    # Save the full response so it stays visible after reruns.
    st.session_state.messages.append(assistant_message)
