"""OpenAlex works API client (optional; free tier is budget-limited without a key)."""
from __future__ import annotations

import re

from ..config import settings
from ..utils.http import HttpError, http
from ..utils.logging import get_logger
from .models import PaperRecord

log = get_logger("openalex")
BASE = "https://api.openalex.org/works"


def _abstract(inv: dict | None) -> str:
    if not inv:
        return ""
    pos = {}
    for word, idxs in inv.items():
        for i in idxs:
            pos[i] = word
    return " ".join(pos[i] for i in sorted(pos))


def _to_record(w: dict) -> PaperRecord | None:
    title = (w.get("title") or "").strip()
    if not title:
        return None
    doi = (w.get("doi") or "").replace("https://doi.org/", "") or None
    loc = w.get("best_oa_location") or w.get("primary_location") or {}
    arxiv = None
    for l in w.get("locations") or []:
        u = (l or {}).get("landing_page_url") or ""
        m = re.search(r"arxiv\.org/abs/([^\s?#]+?)(?:v\d+)?$", u)
        if m:
            arxiv = m.group(1)
    return PaperRecord(
        title=title, abstract=_abstract(w.get("abstract_inverted_index")),
        authors=[(a.get("author") or {}).get("display_name", "") for a in w.get("authorships", []) if a.get("author")],
        year=w.get("publication_year"), venue=((w.get("primary_location") or {}).get("source") or {}).get("display_name"),
        doi=doi, arxiv_id=arxiv, openalex_id=w.get("id"), citation_count=w.get("cited_by_count"),
        pdf_url=loc.get("pdf_url"), url=w.get("id"), sources=["openalex"])


def search(query: str, limit: int = 20, year_from: int | None = None, year_to: int | None = None) -> list[PaperRecord]:
    params = {"search": query, "per-page": min(limit, 50)}
    if year_from or year_to:
        params["filter"] = f"publication_year:{year_from or 1990}-{year_to or 2100}"
    if settings.openalex_api_key:
        params["api_key"] = settings.openalex_api_key
    if settings.contact_email:
        params["mailto"] = settings.contact_email
    try:
        data = http.get(BASE, params=params, retries=2)
    except HttpError as e:
        log.warning("OpenAlex search failed: %r", e)
        return []
    return [r for r in (_to_record(w) for w in data.get("results", [])) if r]
