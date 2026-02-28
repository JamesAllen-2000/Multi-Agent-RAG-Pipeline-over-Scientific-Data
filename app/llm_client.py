"""
LLM client factory: returns an OpenAI-compatible client for chat (planner, reasoning).
Supports OpenAI and Groq (base_url + api_key). Embeddings stay on OpenAI (see config).
"""
from openai import OpenAI

from app.config import get_settings

GROQ_BASE_URL = "https://api.groq.com/openai/v1"


def get_chat_client() -> OpenAI:
    """
    Return client for chat completions (planner, reasoning agent).
    Uses GROQ_API_KEY + Groq base URL when llm_provider=groq, else OPENAI_API_KEY.
    """
    settings = get_settings()
    if getattr(settings, "llm_provider", "openai").strip().lower() == "groq":
        key = (settings.groq_api_key or "").strip()
        return OpenAI(
            base_url=GROQ_BASE_URL,
            api_key=key,
            timeout=getattr(settings, "openai_timeout", 60.0),
        )
    return OpenAI(
        api_key=settings.openai_api_key,
        timeout=settings.openai_timeout,
    )


def get_chat_model() -> str:
    """Return model name for chat (Groq or OpenAI depending on provider)."""
    settings = get_settings()
    if getattr(settings, "llm_provider", "openai").strip().lower() == "groq":
        return getattr(settings, "groq_model", "llama-3.3-70b-versatile") or "llama-3.3-70b-versatile"
    return settings.openai_model


def get_embedding_client():
    """Embeddings are HuggingFace or OpenAI."""
    settings = get_settings()
    provider = getattr(settings, "embedding_provider", "openai").strip().lower()
    if provider == "huggingface":
        from langchain_huggingface import HuggingFaceEmbeddings
        model_name = getattr(settings, "embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
        return HuggingFaceEmbeddings(model_name=model_name)

    return OpenAI(
        api_key=settings.openai_api_key,
        timeout=getattr(settings, "openai_timeout", 60.0),
    )


def has_llm_configured() -> bool:
    """True if the configured LLM provider has an API key set."""
    settings = get_settings()
    if getattr(settings, "llm_provider", "openai").strip().lower() == "groq":
        return bool((settings.groq_api_key or "").strip())
    return bool((settings.openai_api_key or "").strip())
