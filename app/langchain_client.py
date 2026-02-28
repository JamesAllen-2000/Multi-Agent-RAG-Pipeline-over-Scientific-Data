"""
Optional LangChain chat model factory.

This stays intentionally thin:
- We still control prompts, JSON parsing, tool loop, and retries in our own code.
- LangChain is only used as a model wrapper interface.
"""
from app.config import get_settings
from app.llm_client import GROQ_BASE_URL, get_chat_model


def get_langchain_chat_model():
    """Return a ChatOpenAI model configured for OpenAI or Groq."""
    try:
        from langchain_openai import ChatOpenAI
    except ImportError as e:
        raise ImportError(
            "LangChain mode enabled but dependencies are missing. "
            "Install langchain-core and langchain-openai."
        ) from e

    settings = get_settings()
    model = get_chat_model()
    common_kwargs = {
        "model": model,
        "temperature": 0,
        "timeout": settings.openai_timeout,
        "max_retries": settings.openai_max_retries,
    }
    if (settings.llm_provider or "openai").strip().lower() == "groq":
        return ChatOpenAI(
            **common_kwargs,
            api_key=settings.groq_api_key,
            base_url=GROQ_BASE_URL,
        )
    return ChatOpenAI(
        **common_kwargs,
        api_key=settings.openai_api_key,
    )
