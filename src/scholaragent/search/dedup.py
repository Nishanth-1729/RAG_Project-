"""Deduplicate paper records coming from several sources (DOI -> arXiv id -> exact title -> fuzzy title)."""
from __future__ import annotations

import re

from rapidfuzz import fuzz

from ..utils.text import normalize_title
from .models import PaperRecord


def _arx(a: str | None) -> str | None:
    return re.sub(r"v\d+$", "", a.strip().lower()) if a else None


def _doi(d: str | None) -> str | None:
    if not d:
        return None
    d = d.lower().strip()
    d = re.sub(r"^https?://(dx\.)?doi\.org/", "", d)
    return d


def _surname(name: str) -> str:
    parts = name.strip().split()
    return parts[-1].lower() if parts else ""


def merge(a: PaperRecord, b: PaperRecord) -> PaperRecord:
    """Merge b into a (a is kept as the canonical record)."""
    a.abstract = a.abstract if len(a.abstract) >= len(b.abstract) else b.abstract
    a.authors = a.authors if len(a.authors) >= len(b.authors) else b.authors
    for f in ("year", "venue", "doi", "arxiv_id", "s2_id", "openalex_id", "url"):
        if not getattr(a, f) and getattr(b, f):
            setattr(a, f, getattr(b, f))
    if a.venue == "arXiv" and b.venue and b.venue != "arXiv":
        a.venue = b.venue
    if b.citation_count is not None:
        a.citation_count = max(a.citation_count or 0, b.citation_count)
    # prefer arXiv PDFs (stable, always parseable) over publisher links
    if b.pdf_url and (not a.pdf_url or ("arxiv.org" in b.pdf_url and "arxiv.org" not in a.pdf_url)):
        a.pdf_url = b.pdf_url
    a.sources = sorted(set(a.sources) | set(b.sources))
    a.references = list(dict.fromkeys(a.references + b.references))
    a.queries = list(dict.fromkeys(a.queries + b.queries))
    return a


def deduplicate(records: list[PaperRecord], fuzzy_threshold: float = 93.0) -> list[PaperRecord]:
    by_doi: dict[str, PaperRecord] = {}
    by_arx: dict[str, PaperRecord] = {}
    by_title: dict[str, PaperRecord] = {}
    out: list[PaperRecord] = []

    def index(r: PaperRecord):
        if _doi(r.doi):
            by_doi[_doi(r.doi)] = r
        if _arx(r.arxiv_id):
            by_arx[_arx(r.arxiv_id)] = r
        by_title[normalize_title(r.title)] = r

    for r in records:
        r.doi, r.arxiv_id = _doi(r.doi), _arx(r.arxiv_id)
        hit = (by_doi.get(r.doi) if r.doi else None) or (by_arx.get(r.arxiv_id) if r.arxiv_id else None) \
            or by_title.get(normalize_title(r.title))
        if hit is None and len(r.title) > 15:   # fuzzy title matching, guarded by year/author agreement
            nt = normalize_title(r.title)
            for cand in out:
                if abs(len(normalize_title(cand.title)) - len(nt)) > 25:
                    continue
                if fuzz.ratio(normalize_title(cand.title), nt) >= fuzzy_threshold:
                    if r.year and cand.year and abs(r.year - cand.year) > 1:
                        continue
                    if r.authors and cand.authors and _surname(r.authors[0]) != _surname(cand.authors[0]):
                        continue
                    hit = cand
                    break
        if hit is None:
            out.append(r)
            index(r)
        else:
            merge(hit, r)
            index(hit)
    return out
