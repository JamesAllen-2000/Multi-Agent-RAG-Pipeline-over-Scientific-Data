from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Load from env (and .env file)."""

    # LLM: "openai" or "groq"
    llm_provider: str = "openai"
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    # Groq (OpenAI-compatible API at api.groq.com)
    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"

    # Optional thin LangChain wrapper for planner/reasoning
    # Set to True to use LangChain path.
    use_langchain: bool = True

    # Embeddings: "openai" or "huggingface"
    embedding_provider: str = "huggingface"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    openai_embedding_model: str = "text-embedding-3-small"

    # Paths
    data_dir: Path = Path("data")
    chroma_path: Path = Path("data/chroma")
    db_path: Path = Path("data/sources.db")

    # Retrieval
    top_k_docs: int = 5
    chunk_size: int = 512
    chunk_overlap: int = 64

    # Production: timeouts (seconds) and retries
    openai_timeout: float = 60.0
    openai_max_retries: int = 2
    query_timeout_seconds: float = 120.0
    max_concurrent_queries: int = 10

    # Logging
    log_level: str = "INFO"

    # arXiv API (retrieval: max results per live query; ingestion: delay between paged requests)
    arxiv_max_results: int = 15
    arxiv_request_delay_seconds: float = 3.0
    arxiv_timeout: float = 30.0

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


def get_settings() -> Settings:
    return Settings()
