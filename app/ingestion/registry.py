"""Registry of ingesters by source type. No magic — explicit mapping."""
from app.ingestion.base import BaseIngester
from app.ingestion.arxiv import ArxivIngester
from app.ingestion.documents import DocumentIngester
from app.ingestion.kaggle import KaggleIngester
from app.ingestion.structured import StructuredIngester

_INGESTERS: dict[str, type[BaseIngester]] = {
    "document": DocumentIngester,
    "structured": StructuredIngester,
    "arxiv": ArxivIngester,
    "kaggle": KaggleIngester,
}


def list_source_types() -> list[str]:
    return list(_INGESTERS.keys())


def get_ingester(source_type: str) -> BaseIngester:
    if source_type not in _INGESTERS:
        raise ValueError(f"Unknown source type: {source_type}. Known: {list_source_types()}")
    return _INGESTERS[source_type]()
