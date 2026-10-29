import json
import os

from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI")
DB_NAME = os.getenv("MONGODB_DB", "college_db")
COLLECTION_NAME = os.getenv("MONGODB_COLLECTION", "documents")

# Fields that must never reach the LLM or the UI (vectors are huge and unreadable).
EXCLUDED_FIELDS = {"embedding"}

if not MONGODB_URI:
    raise RuntimeError("MONGODB_URI is missing. Copy .env.example to .env and set it.")

# MongoClient connects lazily, so importing this module no longer crashes the
# app when the database is unreachable. Connection problems show up on the
# first query (or ping()), where the callers can handle them.
_client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=10000)
db = _client[DB_NAME]
collection = db[COLLECTION_NAME]


def ping() -> bool:
    """Raise if MongoDB is unreachable; return True otherwise."""
    _client.admin.command("ping")
    return True


def get_collection_names() -> list[str]:
    return db.list_collection_names()


# --------------------------------------------------
# Schema description for the query-generation model
# --------------------------------------------------

_schema_cache: dict | None = None


def _flatten(doc: dict, prefix: str = ""):
    """Yield (dotted_path, value) for every non-dict leaf in a document."""
    for key, value in doc.items():
        if key in EXCLUDED_FIELDS:
            continue
        path = f"{prefix}{key}"
        if isinstance(value, dict):
            yield from _flatten(value, f"{path}.")
        else:
            yield path, value


def _type_name(value) -> str:
    if isinstance(value, list) and value:
        return f"list[{type(value[0]).__name__}]"
    return type(value).__name__


def get_collection_schema_dict(sample_size: int = 50, refresh: bool = False) -> dict:
    """
    Return {field_path: {"types": [...], "examples": [...]}} for the collection.

    Uses a random sample, flattens nested documents to dotted paths, and caches
    the result so it is not rebuilt for every question. Pass refresh=True after
    the data changes. An empty collection returns {} and is not cached.
    """
    global _schema_cache

    if _schema_cache is not None and not refresh:
        return _schema_cache

    samples = list(
        collection.aggregate([
            {"$sample": {"size": sample_size}},
            {"$project": {field: 0 for field in EXCLUDED_FIELDS}},
        ])
    )

    types: dict[str, set[str]] = {}
    examples: dict[str, list[str]] = {}

    for doc in samples:
        for path, value in _flatten(doc):
            if path == "_id":
                continue
            types.setdefault(path, set()).add(_type_name(value))
            if isinstance(value, (str, int, float, bool)):
                seen = examples.setdefault(path, [])
                text = str(value)[:40]
                if len(seen) < 3 and text not in seen:
                    seen.append(text)

    schema: dict[str, dict] = {}
    for path, type_set in sorted(types.items()):
        entry: dict = {"types": sorted(type_set)}
        if path in examples:
            entry["examples"] = examples[path]
        schema[path] = entry

    if schema:
        _schema_cache = schema
    return schema


def get_collection_schema() -> str:
    """The schema as JSON text, ready to paste into a prompt."""
    schema = get_collection_schema_dict()
    if not schema:
        return "The collection is empty. No fields are available to inspect."
    return json.dumps(schema, indent=2, default=str)
