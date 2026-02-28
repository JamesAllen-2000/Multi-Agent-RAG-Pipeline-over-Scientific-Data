"""
arXiv API client: HTTP GET to export.arxiv.org, parse Atom XML.
No auth required. Respects max_results (cap 2000 per request) and optional 3s delay between calls.
See https://info.arxiv.org/help/api/user-manual.html
"""
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Any

from app.logging_config import get_logger

logger = get_logger(__name__)

ARXIV_NS = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
BASE_URL = "http://export.arxiv.org/api/query"


@dataclass
class ArxivEntry:
    """One paper from the API response."""
    arxiv_id: str
    title: str
    summary: str
    authors: list[str]
    published: str
    updated: str
    link_abs: str
    link_pdf: str
    primary_category: str


def _text(el: ET.Element | None, default: str = "") -> str:
    if el is None:
        return default
    return " ".join((t.strip() for t in (el.itertext() or []) if t and t.strip()))


def _find(el: ET.Element, path: str, ns: dict) -> ET.Element | None:
    return el.find(path, ns)


def parse_atom_feed(xml_bytes: bytes) -> list[ArxivEntry]:
    """Parse Atom feed body; return list of ArxivEntry. Error entries (single entry with <title>Error</title>) are skipped."""
    root = ET.fromstring(xml_bytes)
    entries: list[ArxivEntry] = []
    for entry_el in root.findall("atom:entry", ARXIV_NS):
        title_el = _find(entry_el, "atom:title", ARXIV_NS)
        if title_el is not None and _text(title_el).strip().lower() == "error":
            continue
        id_el = _find(entry_el, "atom:id", ARXIV_NS)
        id_url = _text(id_el)
        arxiv_id = id_url.replace("http://arxiv.org/abs/", "").replace("https://arxiv.org/abs/", "").rstrip("/")
        if not arxiv_id:
            continue
        title = _text(_find(entry_el, "atom:title", ARXIV_NS))
        summary = _text(_find(entry_el, "atom:summary", ARXIV_NS))
        authors = []
        for a in entry_el.findall("atom:author", ARXIV_NS):
            name_el = a.find("atom:name", ARXIV_NS)
            if name_el is not None:
                authors.append(_text(name_el))
        published = _text(_find(entry_el, "atom:published", ARXIV_NS))
        updated = _text(_find(entry_el, "atom:updated", ARXIV_NS))
        link_abs = ""
        link_pdf = ""
        for link in entry_el.findall("atom:link", ARXIV_NS):
            href = link.get("href") or ""
            rel = link.get("rel") or ""
            title_attr = link.get("title") or ""
            if rel == "alternate":
                link_abs = href
            if rel == "related" and title_attr == "pdf":
                link_pdf = href
        primary_el = entry_el.find("arxiv:primary_category", ARXIV_NS)
        primary_category = primary_el.get("term", "") if primary_el is not None else ""
        entries.append(
            ArxivEntry(
                arxiv_id=arxiv_id,
                title=title,
                summary=summary,
                authors=authors,
                published=published,
                updated=updated,
                link_abs=link_abs,
                link_pdf=link_pdf,
                primary_category=primary_category,
            )
        )
    return entries


def fetch_query(
    search_query: str,
    *,
    id_list: str | None = None,
    start: int = 0,
    max_results: int = 10,
    timeout: float = 30.0,
) -> list[ArxivEntry]:
    """
    GET export.arxiv.org/api/query with given params. max_results capped at 2000 per request.
    Returns list of ArxivEntry; empty on HTTP or parse error (errors logged).
    """
    if max_results <= 0 or max_results > 2000:
        max_results = min(10, 2000)
    params: dict[str, str | int] = {
        "search_query": search_query or "all:all",
        "start": start,
        "max_results": max_results,
    }
    if id_list and id_list.strip():
        params["id_list"] = id_list.strip()
    qs = urllib.parse.urlencode(params)
    url = f"{BASE_URL}?{qs}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "MultiAgentRAG/1.0 (scientific RAG pipeline)"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read()
        return parse_atom_feed(body)
    except Exception as e:
        logger.warning("arxiv_fetch_failed", extra={"error": type(e).__name__})
        return []


def fetch_with_paging(
    search_query: str,
    *,
    max_total: int = 50,
    per_request: int = 20,
    delay_seconds: float = 3.0,
    timeout: float = 30.0,
) -> list[ArxivEntry]:
    """
    Fetch up to max_total results by paging (start=0, start=per_request, ...).
    Applies delay_seconds between requests per arXiv API guidelines.
    """
    all_entries: list[ArxivEntry] = []
    start = 0
    while start < max_total:
        chunk = fetch_query(
            search_query,
            start=start,
            max_results=min(per_request, max_total - start, 2000),
            timeout=timeout,
        )
        if not chunk:
            break
        all_entries.extend(chunk)
        if len(chunk) < per_request:
            break
        start += len(chunk)
        if start < max_total and delay_seconds > 0:
            time.sleep(delay_seconds)
    return all_entries[:max_total]
