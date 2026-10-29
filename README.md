# Sambhav MongoDB RAG Chatbot

A learning-oriented chatbot combining:
- MongoDB Atlas for storage
- Local Sentence Transformers embeddings (`all-MiniLM-L6-v2`, 384 dimensions)
- MongoDB Atlas Vector Search for RAG retrieval
- LangChain + an OpenAI chat model for answers and query generation
- LangGraph for routing between semantic RAG and structured MongoDB queries
- Streamlit for the chat UI

> Important: MongoDB queries are not SQL. This project generates read-only MongoDB aggregation pipelines. If the client's database is actually PostgreSQL/MySQL, the database-query layer needs a SQL implementation instead.

## 1. Project structure

```text
sambhav_mongodb_rag_chatbot/
├── app.py
├── db.py
├── graph.py
├── query_generator.py
├── rag.py
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

## 2. Set up on Windows / VS Code

Open the project folder in VS Code, then in its PowerShell terminal run:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Using the venv's Python executable directly avoids PowerShell execution-policy problems with `Activate.ps1`.

Copy the environment template:

```powershell
Copy-Item .env.example .env
```

Edit `.env` and set:
- `MONGODB_URI`: your Atlas connection string
- `MONGODB_DB=college_db`
- `MONGODB_COLLECTION=documents`
- `MONGODB_VECTOR_INDEX=vector_index`
- `OPENAI_API_KEY`: a valid API key with API access/credits

Never commit `.env` or share the connection string/API key.

## 3. Verify MongoDB fields

This starter expects each document to contain:
- `title`: title/name of the document
- `content` or `text`: the text to answer from
- `embedding`: a 384-number vector made with `all-MiniLM-L6-v2`

Your earlier embedding script may have stored text under a different field. If so, update `rag.py`'s projection and context-building code to match the actual field. Do not regenerate embeddings with a different model unless you also rebuild the index with that model's dimensions.

## 4. Create the Atlas Vector Search index

In Atlas, open the `college_db.documents` collection and create a **Vector Search** index named `vector_index` with this definition:

```json
{
  "fields": [
    {
      "type": "vector",
      "path": "embedding",
      "numDimensions": 384,
      "similarity": "cosine"
    }
  ]
}
```

Wait until the index status is READY before testing RAG. The vector index field path and dimensions must exactly match your stored vectors.

## 5. Run the app

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Streamlit prints a local URL, usually `http://localhost:8501`.

## 6. Test questions

RAG questions:
- What is a DBMS?
- Explain a primary key.
- What do the notes say about database systems?

Structured-query questions:
- How many documents are in this collection?
- List all documents.
- Show documents sorted by title.

Structured query generation uses sample field names from the configured collection. If the client data has a complex schema, improve the schema introspection and test the generated pipeline carefully.

## 7. How the graph works

`graph.py` creates a LangGraph `StateGraph`:
1. `route` decides which path to use.
2. `rag` retrieves related documents using `$vectorSearch`, then asks the LLM to answer from those sources.
3. `database` asks the LLM for a MongoDB aggregation pipeline, validates it against an allowlist, and executes it read-only.
4. The final state is shown in Streamlit.

The current router is deliberately a simple keyword baseline, not a production classifier. Replace it with an LLM classifier or a more carefully tested intent classifier after understanding the basic flow.

## 8. Security notes

This is a learning starter, not a production-ready client deployment:
- Use a MongoDB database user with read-only privileges for this app.
- Do not expose secrets in source code or logs.
- Query validation is defense in depth; it does not replace least-privilege database credentials, query timeouts, or schema-specific testing.
- Only use approved collections and fields. Add per-user authorization before connecting private client data.
- Test generated pipelines on a staging dataset before production use.
- Consider logging query intent and timing, but avoid logging sensitive records.
- OpenAI API calls may send prompts and retrieved context to the configured model provider. Confirm the client's data-handling requirements before using real data.

## 9. Troubleshooting

**`MONGODB_URI is missing`**: ensure `.env` is in the same folder as `app.py` and has a valid URI.

**Vector search index not found**: verify the index is named `vector_index`, its path is `embedding`, its dimensions are 384, and its status is READY.

**No relevant documents**: inspect a sample document and confirm `embedding` is a numeric list of length 384, with `content` or `text` containing the source text.

**OpenAI authentication/billing error**: check `OPENAI_API_KEY`, account API access, and billing/credits.

**Wrong query fields**: inspect your real collection schema and update the prompt/schema strategy. The LLM must not guess field names.
