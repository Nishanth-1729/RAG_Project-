"""Semantic Scholar Academic Graph API client (works without a key, but is heavily rate limited)."""
from __future__ import annotations

import time

from ..config import settings
from ..utils.http import HttpError, http
from ..utils.logging import get_logger
from .models import PaperRecord

log = get_logger("s2")
BASE = "https://api.semanticscholar.org/graph/v1"
FIELDS = "paperId,title,abstract,authors,year,venue,citationCount,externalIds,openAccessPdf,url"


_fail_streak = 0
_blocked_until = 0.0


def _guarded_get(url: str, params: dict) -> dict:
    """Circuit breaker: the keyless S2 tier is almost always rate limited - after 2 consecutive failures stop
    calling it for 10 minutes (set SEMANTIC_SCHOLAR_API_KEY for reliable access)."""
    global _fail_streak, _blocked_until
    if time.time() < _blocked_until:
        raise HttpError("Semantic Scholar temporarily disabled (rate limited)")
    try:
        data = http.get(url, params=params, headers=_headers(), retries=1 if not settings.s2_api_key else 4)
        _fail_streak = 0
        return data
    except HttpError:
        _fail_streak += 1
        if _fail_streak >= 2 and not settings.s2_api_key:
            _blocked_until = time.time() + 600
        raise


def _headers() -> dict:
    return {"x-api-key": settings.s2_api_key} if settings.s2_api_key else {}


def _to_record(p: dict) -> PaperRecord | None:
    if not p or not p.get("title"):
        return None
    ext = p.get("externalIds") or {}
    return PaperRecord(
        title=p["title"].strip(), abstract=(p.get("abstract") or "").strip(),
        authors=[a.get("name", "") for a in (p.get("authors") or []) if a.get("name")],
        year=p.get("year"), venue=p.get("venue") or None, doi=ext.get("DOI"), arxiv_id=ext.get("ArXiv"),
        s2_id=p.get("paperId"), citation_count=p.get("citationCount"),
        pdf_url=(p.get("openAccessPdf") or {}).get("url") or (f"https://arxiv.org/pdf/{ext['ArXiv']}" if ext.get("ArXiv") else None),
        url=p.get("url"), sources=["semantic_scholar"],
    )


def search(query: str, limit: int = 20, year_from: int | None = None, year_to: int | None = None) -> list[PaperRecord]:
    params = {"query": query, "limit": min(limit, 100), "fields": FIELDS}
    if year_from or year_to:
        params["year"] = f"{year_from or ''}-{year_to or ''}"
    try:
        data = _guarded_get(f"{BASE}/paper/search", params)
    except HttpError as e:
        log.warning("Semantic Scholar search failed (%r) - continuing with other sources", e)
        return []
    return [r for r in (_to_record(p) for p in data.get("data", [])) if r]


def get_paper(paper_id: str) -> PaperRecord | None:
    try:
        return _to_record(_guarded_get(f"{BASE}/paper/{paper_id}", {"fields": FIELDS}))
    except HttpError:
        return None


def _edges(paper_id: str, kind: str, limit: int) -> list[PaperRecord]:
    key = "citedPaper" if kind == "references" else "citingPaper"
    try:
        data = _guarded_get(f"{BASE}/paper/{paper_id}/{kind}", {"fields": FIELDS, "limit": limit})
    except HttpError as e:
        log.warning("S2 %s failed for %s: %r", kind, paper_id, e)
        return []
    return [r for r in (_to_record(x.get(key) or {}) for x in data.get("data", [])) if r]


def get_references(paper_id: str, limit: int = 100) -> list[PaperRecord]:
    return _edges(paper_id, "references", limit)


def get_citations(paper_id: str, limit: int = 100) -> list[PaperRecord]:
    return _edges(paper_id, "citations", limit)
