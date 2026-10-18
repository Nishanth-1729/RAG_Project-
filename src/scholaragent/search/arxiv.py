"""arXiv Atom API client."""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET

from ..utils.http import HttpError, http
from ..utils.logging import get_logger
from .models import PaperRecord

log = get_logger("arxiv")
NS = {"a": "http://www.w3.org/2005/Atom", "ar": "http://arxiv.org/schemas/atom"}


def _clean(s: str | None) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


def parse_feed(xml: str) -> list[PaperRecord]:
    root = ET.fromstring(xml)
    out: list[PaperRecord] = []
    for e in root.findall("a:entry", NS):
        id_url = _clean(e.findtext("a:id", namespaces=NS))
        m = re.search(r"arxiv\.org/abs/(.+?)(v\d+)?$", id_url)
        aid = m.group(1) if m else None
        published = e.findtext("a:published", namespaces=NS) or ""
        doi_el = e.find("ar:doi", NS)
        pdf = None
        for l in e.findall("a:link", NS):
            if l.get("title") == "pdf":
                pdf = l.get("href")
        if not pdf and aid:
            pdf = f"https://arxiv.org/pdf/{aid}"
        out.append(PaperRecord(
            title=_clean(e.findtext("a:title", namespaces=NS)),
            abstract=_clean(e.findtext("a:summary", namespaces=NS)),
            authors=[_clean(a.findtext("a:name", namespaces=NS)) for a in e.findall("a:author", NS)],
            year=int(published[:4]) if published[:4].isdigit() else None,
            venue="arXiv", doi=_clean(doi_el.text) if doi_el is not None else None, arxiv_id=aid,
            pdf_url=pdf, url=id_url, sources=["arxiv"],
        ))
    return out


def search(query: str, max_results: int = 20, year_from: int | None = None, year_to: int | None = None,
           categories: list[str] | None = None, sort_by: str = "relevance") -> list[PaperRecord]:
    terms = [t for t in re.split(r"\s+", query.strip()) if t]
    q = " AND ".join(f"all:{t}" for t in terms) if len(terms) <= 4 else f'all:"{query.strip()}"'
    if len(terms) > 4:   # long queries: use the quoted phrase OR the bag of key words
        key = [t for t in terms if len(t) > 3][:6]
        q = f'(all:"{" ".join(terms[:6])}") OR (' + " AND ".join(f"all:{t}" for t in key[:4]) + ")"
    if categories:
        q = f"({q}) AND (" + " OR ".join(f"cat:{c}" for c in categories) + ")"
    if year_from or year_to:
        a, b = year_from or 1990, year_to or 2100
        q = f"({q}) AND submittedDate:[{a}01010000 TO {b}12312359]"
    try:
        xml = http.get("https://export.arxiv.org/api/query",
                       params={"search_query": q, "start": 0, "max_results": max_results,
                               "sortBy": sort_by, "sortOrder": "descending"}, as_text=True)
        return parse_feed(xml)
    except (HttpError, ET.ParseError) as e:
        log.warning("arXiv search failed for %r: %r", query, e)
        return []


def get_by_id(arxiv_id: str) -> PaperRecord | None:
    try:
        xml = http.get("https://export.arxiv.org/api/query", params={"id_list": arxiv_id, "max_results": 1}, as_text=True)
        res = parse_feed(xml)
        return res[0] if res else None
    except (HttpError, ET.ParseError):
        return None
