"""Open-access PDF retrieval for PaperRecords."""
from __future__ import annotations

import json
from pathlib import Path

from ..config import settings
from ..utils.http import HttpError, http
from ..utils.logging import get_logger
from .models import PaperRecord

log = get_logger("fetch")


def candidate_urls(r: PaperRecord) -> list[str]:
    urls = []
    if r.arxiv_id:
        urls.append(f"https://arxiv.org/pdf/{r.arxiv_id}")
    if r.pdf_url:
        urls.append(r.pdf_url)
    return list(dict.fromkeys(urls))


def fetch_pdf(r: PaperRecord, dest_dir: Path | None = None) -> Path | None:
    """Download the PDF (first working open-access URL). Returns the path or None. Never raises."""
    dest_dir = dest_dir or settings.raw_dir
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{r.paper_id}.pdf"
    side = dest.with_suffix(".meta.json")
    if dest.exists() and dest.stat().st_size > 10_000:
        return dest
    for url in candidate_urls(r):
        try:
            http.download(url, dest)
            side.write_text(json.dumps({
                "paper_id": r.paper_id, "title": r.title, "authors": r.authors, "year": r.year, "doi": r.doi,
                "arxiv_id": r.arxiv_id, "venue": r.venue, "abstract": r.abstract, "source": ",".join(r.sources) or "search",
            }, ensure_ascii=False))
            return dest
        except (HttpError, OSError) as e:
            log.info("PDF download failed (%s): %r", url, e)
    return None
