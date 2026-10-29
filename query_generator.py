import copy
import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from pymongo.errors import OperationFailure

from db import (
    EXCLUDED_FIELDS,
    collection,
    get_collection_schema,
    get_collection_schema_dict,
)
from llm import get_llm, response_text


# ==================================================
# 1. Configuration
# ==================================================

ALLOWED_STAGES = {
    "$match",
    "$group",
    "$sort",
    "$limit",
    "$project",
    "$count",
    "$unwind",
    "$skip",
    "$addFields",
    "$set",
}

BLOCKED_KEYS = {
    "$where",
    "$function",
    "$accumulator",
    "$out",
    "$merge",
    "$lookup",
    "$unionWith",
    "$graphLookup",
    "$currentOp",
    "$collStats",
    "$indexStats",
    "$planCacheStats",
}

# After one of these stages, field names are computed (e.g. "total"), so they
# can no longer be checked against the collection schema.
RESHAPING_STAGES = {"$group", "$project", "$addFields", "$set", "$count"}

MAX_LIMIT = 100
MAX_STAGES = 8
MAX_ATTEMPTS = 3


# ==================================================
# 2. Prompt
# ==================================================

SYSTEM_PROMPT = """You convert user questions into MongoDB aggregation pipelines.

Reply with ONLY a JSON object of the form {"pipeline": [ ...stages... ]}.
No markdown, no explanation.

COLLECTION SCHEMA (field: types and example values):
__SCHEMA__

ALLOWED STAGES: __STAGES__

RULES:
- Use only fields from the schema. Never invent fields.
- Do not filter unless the question asks for it.
- Match text values case-insensitively: {"field": {"$regex": "^value$", "$options": "i"}}
- Use value formats as shown in the schema examples (numbers as numbers, text as text).
- Count documents with {"$count": "total"}.
- To list documents, use $project to keep only the relevant fields.
- Sort with 1 (ascending) or -1 (descending).
- For "top N" or "highest/lowest N", use $sort then $limit.
- Never use a stage outside the allowed list.

EXAMPLES (field names are made up; use the schema above for real ones):
Q: How many records are there?
{"pipeline": [{"$count": "total"}]}
Q: List the names of active users alphabetically.
{"pipeline": [{"$match": {"status": "active"}}, {"$sort": {"name": 1}}, {"$project": {"name": 1, "_id": 0}}]}
Q: Average score per department.
{"pipeline": [{"$group": {"_id": "$department", "avg_score": {"$avg": "$score"}}}, {"$sort": {"avg_score": -1}}]}
Q: Top 3 products by price.
{"pipeline": [{"$sort": {"price": -1}}, {"$limit": 3}, {"$project": {"name": 1, "price": 1, "_id": 0}}]}
"""


# ==================================================
# 3. Schema fields
# ==================================================

def get_schema_fields() -> set[str]:
    """Field paths in the collection (dotted for nested fields)."""
    fields = {field for field in get_collection_schema_dict() if field != "_id"}

    if not fields:
        raise ValueError(
            "No document fields were found. "
            "Check that the collection contains documents."
        )

    return fields


# ==================================================
# 4. Pipeline safety validation
# ==================================================

def contains_blocked_key(value: Any) -> bool:
    """Reject prohibited operators anywhere in the pipeline."""
    if isinstance(value, dict):
        for key, child in value.items():
            if isinstance(key, str) and key in BLOCKED_KEYS:
                return True
            if contains_blocked_key(child):
                return True

    elif isinstance(value, list):
        return any(contains_blocked_key(item) for item in value)

    return False


def _as_int(value: Any, name: str) -> int:
    """Accept 5 or 5.0; reject booleans, strings and fractions."""
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer.")
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    if not isinstance(value, int):
        raise ValueError(f"{name} must be an integer.")
    return value


def validate_pipeline(pipeline: Any) -> list[dict]:
    """
    Validate structure and safety, and enforce a result limit.

    Returns a cleaned copy; the input is not modified.
    """
    if not isinstance(pipeline, list) or not pipeline:
        raise ValueError("The model returned an empty or invalid pipeline.")

    if len(pipeline) > MAX_STAGES:
        raise ValueError(f"Pipeline exceeds the maximum of {MAX_STAGES} stages.")

    pipeline = copy.deepcopy(pipeline)

    # Vectors must never be read, grouped on or returned.
    pipeline_text = json.dumps(pipeline, default=str)
    for field in EXCLUDED_FIELDS:
        if field in pipeline_text:
            raise ValueError(f"The pipeline must not reference the '{field}' field.")

    for stage in pipeline:
        if not isinstance(stage, dict) or len(stage) != 1:
            raise ValueError("Each pipeline stage must contain exactly one operator.")

        operator = next(iter(stage))

        if operator not in ALLOWED_STAGES:
            raise ValueError(f"Disallowed MongoDB stage: {operator}")

        if contains_blocked_key(stage):
            raise ValueError("Pipeline contains a prohibited operator.")

        if operator == "$count":
            name = stage["$count"]
            if (
                not isinstance(name, str)
                or not name
                or name.startswith("$")
                or "." in name
            ):
                raise ValueError(
                    '$count must use a plain field name such as {"$count": "total"}.'
                )

        elif operator == "$limit":
            value = _as_int(stage["$limit"], "$limit")
            if value < 1:
                raise ValueError("$limit must be at least 1.")
            stage["$limit"] = min(value, MAX_LIMIT)

        elif operator == "$skip":
            value = _as_int(stage["$skip"], "$skip")
            if value < 0:
                raise ValueError("$skip cannot be negative.")
            stage["$skip"] = value

    # The LAST stage must be a $limit, so the output is always capped.
    if "$limit" not in pipeline[-1]:
        pipeline.append({"$limit": MAX_LIMIT})

    return pipeline


def _stage_field_refs(operator: str, body: Any) -> set[str]:
    """Collect the collection fields a stage refers to."""
    refs: set[str] = set()

    def expr_refs(node: Any) -> None:
        # "$field" strings used as expressions (not "$$variables").
        if isinstance(node, dict):
            for child in node.values():
                expr_refs(child)
        elif isinstance(node, list):
            for child in node:
                expr_refs(child)
        elif isinstance(node, str) and node.startswith("$") and not node.startswith("$$"):
            refs.add(node[1:])

    def match_refs(node: Any) -> None:
        # Field names are keys; recurse only through $and/$or/$nor/$expr.
        if isinstance(node, dict):
            for key, child in node.items():
                if key.startswith("$"):
                    match_refs(child)
                else:
                    refs.add(key)
        elif isinstance(node, list):
            for child in node:
                match_refs(child)

    if operator == "$match":
        match_refs(body)
        expr_refs(body)
    elif operator == "$sort" and isinstance(body, dict):
        refs.update(body)
    elif operator in {"$group", "$project", "$addFields", "$set", "$unwind"}:
        expr_refs(body)

    return refs


def validate_fields(pipeline: list[dict], schema_fields: set[str]) -> None:
    """Reject references to fields that do not exist in the collection."""
    known = {field.split(".")[0] for field in schema_fields} | {"_id"}

    for stage in pipeline:
        operator, body = next(iter(stage.items()))

        unknown = sorted(
            ref
            for ref in _stage_field_refs(operator, body)
            if ref.split(".")[0] not in known
        )
        if unknown:
            raise ValueError(
                f"The pipeline uses unknown field(s) {unknown}. "
                f"Valid fields: {sorted(known - {'_id'})}"
            )

        if operator in RESHAPING_STAGES:
            break  # later stages may use computed field names


# ==================================================
# 5. Parse the model's response
# ==================================================

def parse_model_response(text: Any) -> list[dict]:
    """
    Accept common JSON response shapes:
      {"pipeline": [...]}
      [...]
    and recover the JSON if the model wrapped it in markdown or extra text.
    """
    if not isinstance(text, str):
        raise ValueError("The model returned a response that was not text.")

    text = text.strip()

    if not text:
        raise ValueError("The model returned an empty response.")

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = None
        decoder = json.JSONDecoder()

        for index, character in enumerate(text):
            if character not in "[{":
                continue
            try:
                candidate, _ = decoder.raw_decode(text[index:])
            except json.JSONDecodeError:
                continue
            if isinstance(candidate, (dict, list)):
                data = candidate
                break

        if data is None:
            raise ValueError(
                f"The model did not return valid JSON. Raw response: {text[:500]}"
            )

    if isinstance(data, list):
        return data

    if not isinstance(data, dict):
        raise ValueError("The model must return a JSON object or a pipeline array.")

    for key in ("pipeline", "aggregation_pipeline"):
        if key in data:
            pipeline = data[key]
            break
    else:
        raise ValueError(
            "The model returned JSON, but no pipeline was found. "
            f"Returned keys: {list(data.keys())}."
        )

    if isinstance(pipeline, dict):  # a single stage given without a list
        pipeline = [pipeline]

    if not isinstance(pipeline, list):
        raise ValueError("The generated pipeline must be a JSON array.")

    return pipeline


# ==================================================
# 6. Generate a pipeline from natural language
# ==================================================

def generate_mongodb_pipeline(
    question: str,
    previous_error: str | None = None,
) -> list[dict]:
    """
    Ask Gemini for a pipeline, then parse and validate it.

    If a previous attempt failed, pass its error message so the model can fix it.
    """
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Please enter a question.")

    schema_fields = get_schema_fields()

    system_prompt = (
        SYSTEM_PROMPT
        .replace("__SCHEMA__", get_collection_schema())
        .replace("__STAGES__", ", ".join(sorted(ALLOWED_STAGES)))
    )

    human_text = question.strip()
    if previous_error:
        human_text += (
            f"\n\nYour previous attempt failed: {previous_error}\n"
            "Return a corrected pipeline."
        )

    response = get_llm().invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_text),
    ])

    pipeline = validate_pipeline(parse_model_response(response_text(response)))
    validate_fields(pipeline, schema_fields)
    return pipeline


# ==================================================
# 7. Execute the read-only query
# ==================================================

def execute_pipeline(pipeline: list[dict]) -> list[dict]:
    """Run a validated pipeline and return JSON-safe results."""
    # validate_pipeline guarantees the last stage is $limit. Remove vectors
    # just before it so full documents never carry them back.
    safe_pipeline = pipeline[:-1] + [
        {"$project": {field: 0 for field in EXCLUDED_FIELDS}},
        pipeline[-1],
    ]

    results = list(
        collection.aggregate(safe_pipeline, maxTimeMS=5000, allowDiskUse=False)
    )

    return json.loads(json.dumps(results, default=str))


def run_readonly_query(question: str, max_attempts: int = MAX_ATTEMPTS) -> dict:
    """
    Generate, validate and execute a pipeline, retrying with the error message
    when the model's pipeline is rejected or MongoDB cannot run it.
    """
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Please enter a question.")

    error: str | None = None

    for attempt in range(1, max_attempts + 1):
        try:
            pipeline = generate_mongodb_pipeline(question, previous_error=error)
            results = execute_pipeline(pipeline)
        except (ValueError, OperationFailure) as exc:
            error = str(exc)
            continue

        return {
            "pipeline": pipeline,
            "results": results,
            "message": f"Query returned {len(results)} result(s).",
            "attempts": attempt,
        }

    raise ValueError(
        f"Could not produce a working query after {max_attempts} attempts. "
        f"Last error: {error}"
    )
