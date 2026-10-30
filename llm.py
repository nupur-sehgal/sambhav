"""One place to create the Gemini chat model, shared by every module."""

import os
from functools import lru_cache

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
# How much the model "thinks" before answering (Gemini 3 models). "low" keeps
# replies fast, which is plenty for writing a query. Set GEMINI_THINKING= (empty)
# in .env to use the model default instead.
GEMINI_THINKING = os.getenv("GEMINI_THINKING", "low").strip().lower() or None


@lru_cache(maxsize=1)
def get_llm() -> ChatGoogleGenerativeAI:
    if not os.getenv("GOOGLE_API_KEY"):
        raise RuntimeError(
            "GOOGLE_API_KEY is missing. Add it to your .env file "
            "(get a key from Google AI Studio)."
        )

    # Temperature is left at the model default on purpose: Google recommends
    # the default for Gemini 3 models. The JSON parser and the pipeline
    # validator handle any formatting slips.
    kwargs = {}
    if GEMINI_THINKING:
        kwargs["thinking_level"] = GEMINI_THINKING

    return ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        timeout=60,
        max_retries=1,
        **kwargs,
    )


def response_text(message) -> str:
    """
    Return the plain text of a model reply.

    Newer LangChain versions can return `content` as a list of blocks, so use
    the message's text accessor instead of `.content`.
    """
    text = getattr(message, "text", None)
    if isinstance(text, str):  # current versions: .text is a property
        return str(text)
    if callable(text):  # older versions expose text() as a method
        return str(text() or "")
    return ""