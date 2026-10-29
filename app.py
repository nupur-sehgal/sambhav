
import streamlit as st
from graph import ask
from db import get_collection_names, collection


# --------------------------------------------------
# 1. Page setup
# --------------------------------------------------

st.set_page_config(
    page_title="Sambhav MongoDB RAG Chatbot",
    page_icon="💬",
    layout="wide",
)

st.title("💬 Sambhav — MongoDB RAG Chatbot")
st.caption("Learn how LangChain, LangGraph, and MongoDB work together")


# --------------------------------------------------
# 2. Helper functions
# --------------------------------------------------

def check_mongodb_connection():
    """Check whether MongoDB responds to a ping."""
    collection.database.client.admin.command("ping")
    return True


def render_sources(sources):
    """Display retrieved documents, if any."""
    if not isinstance(sources, list) or not sources:
        return

    with st.expander("📚 Retrieved sources"):
        for index, source in enumerate(sources, start=1):
            if not isinstance(source, dict):
                st.write(f"Source {index}: {source}")
                continue

            title = source.get("title") or f"Source {index}"
            content = (
                source.get("content")
                or source.get("text")
                or "(No content field)"
            )

            st.markdown(f"**{title}**")
            st.write(content)

            score = source.get("score")
            if isinstance(score, (int, float)):
                st.caption(f"Vector score: {score:.3f}")

            if index < len(sources):
                st.divider()


def render_query_details(message):
    """Display query details when the graph returns them."""
    pipeline = message.get("pipeline")

    if pipeline is not None:
        with st.expander("🔍 Generated MongoDB aggregation pipeline"):
            st.json(pipeline)

    results = message.get("results")

    if results is not None:
        with st.expander("🗃️ Database results"):
            st.json(results)


def render_assistant_message(message):
    """Render a saved assistant response and its extra details."""
    st.markdown(message.get("content", "No answer returned."))

    route = message.get("route")
    if route:
        st.caption(f"LangGraph route: `{route}`")

    render_query_details(message)
    render_sources(message.get("sources"))

    error = message.get("error")
    if error:
        st.warning(error)

    # Detailed errors are useful while developing.
    debug_error = message.get("debug_error")
    if debug_error:
        with st.expander("Developer error details"):
            st.code(debug_error)


# --------------------------------------------------
# 3. Sidebar: database status and example questions
# --------------------------------------------------

with st.sidebar:
    st.header("Project status")

    try:
        check_mongodb_connection()
        st.success("MongoDB connection successful")

        st.write("Database:", collection.database.name)
        st.write("Collection:", collection.name)

        collection_names = get_collection_names()
        st.write(
            "Collections:",
            ", ".join(collection_names) if collection_names else "None found",
        )

        st.metric(
            "Documents in collection",
            collection.count_documents({}),
        )

    except Exception:
        st.error("Could not verify the MongoDB connection.")
        st.caption(
            "Check your .env settings and database connection. "
            "We will inspect db.py next."
        )

    st.info(
        "Vector Search requires an Atlas index configured for your "
        "embedding field if your retrieval code uses Atlas Vector Search."
    )

    st.divider()
    st.subheader("Try asking")

    example_questions = [
        "What is a DBMS?",
        "Explain primary keys.",
        "How many documents are in this collection?",
    ]

    for example in example_questions:
        st.markdown(f"- {example}")

    st.caption(
        "The answers depend on the database content and the "
        "query/retrieval logic implemented in the project."
    )


# --------------------------------------------------
# 4. Initialise chat history
# --------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []


# --------------------------------------------------
# 5. Display previous messages
# --------------------------------------------------

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            render_assistant_message(message)
        else:
            st.markdown(message.get("content", ""))


# --------------------------------------------------
# 6. Handle a new question
# --------------------------------------------------

question = st.chat_input("Ask a question about your MongoDB data...")

if question:
    # Show and save the user's question.
    st.session_state.messages.append(
        {"role": "user", "content": question}
    )

    with st.chat_message("user"):
        st.markdown(question)

    # Run the chatbot without letting an exception crash the page.
    with st.chat_message("assistant"):
        with st.spinner("Thinking and querying your data..."):
            try:
                result = ask(question)

                if not isinstance(result, dict):
                    raise TypeError(
                        "The ask() function must return a dictionary."
                    )

                answer = result.get("answer") or "No answer returned."

                assistant_message = {
                    "role": "assistant",
                    "content": str(answer),
                    "route": result.get("route"),
                    "pipeline": result.get("pipeline"),
                    "results": result.get("results", []),
                    "sources": result.get("sources", []),
                    "error": result.get("error"),
                }

            except Exception as exc:
                assistant_message = {
                    "role": "assistant",
                    "content": (
                        "Sorry, something went wrong while processing "
                        "your question. Please check the error details "
                        "below while debugging."
                    ),
                    "error": "The chatbot could not complete this request.",
                    "debug_error": f"{type(exc).__name__}: {exc}",
                }

        render_assistant_message(assistant_message)

        # Save the full response so it remains visible after reruns.
        st.session_state.messages.append(assistant_message)
