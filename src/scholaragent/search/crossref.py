"""Crossref REST API client (metadata + DOIs; abstracts are often missing)."""
from __future__ import annotations

import re

from ..config import settings
from ..utils.http import HttpError, http
from ..utils.logging import get_logger
from .models import PaperRecord

log = get_logger("crossref")
BASE = "https://api.crossref.org/works"


def _to_record(it: dict) -> PaperRecord | None:
    title = (it.get("title") or [""])[0].strip()
    if not title:
        return None
    year = None
    for k in ("published-print", "published-online", "issued", "created"):
        parts = (it.get(k) or {}).get("date-parts") or [[None]]
        if parts and parts[0] and parts[0][0]:
            year = int(parts[0][0])
            break
    abstract = re.sub(r"<[^>]+>", " ", it.get("abstract") or "")
    abstract = re.sub(r"\s+", " ", abstract).strip()
    authors = [" ".join(filter(None, [a.get("given"), a.get("family")])) for a in it.get("author", [])]
    pdf = None
    for l in it.get("link", []) or []:
        if "pdf" in (l.get("content-type") or "") or str(l.get("URL", "")).endswith(".pdf"):
            pdf = l.get("URL")
    return PaperRecord(title=title, abstract=abstract, authors=[a for a in authors if a], year=year,
                       venue=(it.get("container-title") or [None])[0], doi=it.get("DOI"),
                       citation_count=it.get("is-referenced-by-count"), pdf_url=pdf, url=it.get("URL"),
                       sources=["crossref"])


def search(query: str, rows: int = 20, year_from: int | None = None, year_to: int | None = None) -> list[PaperRecord]:
    params = {"query.bibliographic": query, "rows": rows,
              "select": "DOI,title,abstract,author,issued,published-print,published-online,container-title,is-referenced-by-count,link,URL,created"}
    f = []
    if year_from:
        f.append(f"from-pub-date:{year_from}")
    if year_to:
        f.append(f"until-pub-date:{year_to}")
    f.append("type:journal-article,type:proceedings-article,type:posted-content")
    params["filter"] = ",".join(f[:2]) if len(f) > 1 else f[0]
    if settings.contact_email:
        params["mailto"] = settings.contact_email
    try:
        data = http.get(BASE, params=params, retries=3)
    except HttpError as e:
        log.warning("Crossref search failed: %r", e)
        return []
    return [r for r in (_to_record(i) for i in data.get("message", {}).get("items", [])) if r]


def lookup_title(title: str) -> PaperRecord | None:
    res = search(title, rows=3)
    return res[0] if res else None
