"""Try several Gemini models and report how long each takes (or how it fails)."""

import os
import time

from dotenv import load_dotenv

load_dotenv()

if not os.getenv("GOOGLE_API_KEY"):
    raise SystemExit("GOOGLE_API_KEY not found in .env")

t0 = time.time()
from langchain_google_genai import ChatGoogleGenerativeAI

print(f"import took {time.time() - t0:.1f}s\n")

# (model name, thinking_level or None for the model default)
CANDIDATES = [
    ("gemini-3.8-flash", "low"),
    ("gemini-3.8-flash", None),
    ("gemini-3.6-flash", "low"),
    ("gemini-3.5-flash-lite", "low"),
    ("gemini-3.1-flash-lite", "low"),
]

for model, thinking in CANDIDATES:
    kwargs = {"thinking_level": thinking} if thinking else {}
    llm = ChatGoogleGenerativeAI(model=model, timeout=30, max_retries=0, **kwargs)

    label = f"{model} (thinking={thinking or 'default'})"
    start = time.time()
    try:
        reply = llm.invoke("Reply with the single word: ok")
        text = str(getattr(reply, "text", "") or "").strip()[:40]
        print(f"OK    {time.time() - start:5.1f}s  {label}  -> {text!r}")
    except Exception as exc:
        message = str(exc).replace("\n", " ")[:120]
        print(f"FAIL  {time.time() - start:5.1f}s  {label}  -> {type(exc).__name__}: {message}")
