"""One place to create the Gemini chat model, shared by every module."""

import os
from functools import lru_cache

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")


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
    return ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        timeout=60,
        max_retries=2,
    )


def response_text(message) -> str:
    """
    Return the plain text of a model reply.

    Newer LangChain versions can return `content` as a list of blocks, so use
    the message's text accessor instead of `.content`.
    """
    text = getattr(message, "text", None)
    if callable(text):  # older versions expose text() as a method
        text = text()
    return str(text or "")
