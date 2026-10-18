"""PDF -> ParsedDocument (cached on disk) -> chunks."""
from __future__ import annotations

import json
import re
from pathlib import Path

from ..config import settings
from ..utils.logging import get_logger, timed
from ..utils.text import slugify, stable_hash
from .chunker import Chunker, ChunkerConfig
from .metadata import fill_metadata
from .models import Chunk, ParsedDocument
from .pdf_parser import PDFParser
from .section_parser import SectionParser

log = get_logger("ingest")


def make_paper_id(path: Path, doi: str | None = None, arxiv_id: str | None = None) -> str:
    if arxiv_id:
        return "arxiv_" + slugify(arxiv_id.replace("/", "_"), 30)
    if doi:
        return "doi_" + stable_hash(doi.lower(), n=10)
    import hashlib
    return "p_" + hashlib.sha1(path.read_bytes()).hexdigest()[:10]       # identical files -> identical id


def parse_pdf(path: str | Path, overrides: dict | None = None, use_cache: bool = True,
              paper_id: str | None = None) -> ParsedDocument:
    path = Path(path)
    overrides = overrides or {}
    sidecar = path.with_suffix(".meta.json")
    if sidecar.exists():
        try:
            overrides = {**json.loads(sidecar.read_text()), **overrides}
        except Exception:  # noqa: BLE001
            log.warning("bad sidecar %s", sidecar)
    pid = paper_id or overrides.get("paper_id") or make_paper_id(path, overrides.get("doi"), overrides.get("arxiv_id"))
    cache = settings.parsed_dir / f"{pid}.json"
    if use_cache and cache.exists():
        return ParsedDocument.from_dict(json.loads(cache.read_text()))

    with timed("parse_pdf", file=path.name):
        blocks, info = PDFParser().parse(path)
        sections, extra = SectionParser().parse(blocks, info["body_font_size"])
        doc = ParsedDocument(paper_id=pid, path=str(path), n_pages=info["n_pages"], needs_ocr=info["needs_ocr"],
                             warnings=list(info["warnings"]), sections=sections,
                             source=overrides.get("source", "upload"))
        head = " ".join(b.text for b in extra["front"][:40]) + " " + " ".join(
            p.text for s in sections[:2] for p in s.paragraphs[:3])
        fill_metadata(doc, extra["front"], info["pdf_meta"], head, overrides)
        # abstract
        ab = next((s for s in sections if s.canonical == "abstract"), None)
        if ab and ab.paragraphs:
            doc.abstract = ab.text
        elif not doc.abstract and extra["front"]:
            guess = max((b.text for b in extra["front"] if b.page == 1), key=len, default="")
            if 400 < len(guess) < 3000:
                doc.abstract = guess
        if overrides.get("abstract"):
            doc.abstract = overrides["abstract"]
        refs = next((s for s in reversed(sections) if s.canonical == "references"), None)
        doc.references_text = refs.text if refs else ""
        if not sections:
            doc.warnings.append("no sections detected")
    cache.write_text(json.dumps(doc.to_dict(), ensure_ascii=False))
    return doc


def chunk_document(doc: ParsedDocument, cfg: ChunkerConfig | None = None, embed_fn=None) -> list[Chunk]:
    cfg = cfg or ChunkerConfig(strategy=settings.chunk_strategy, max_tokens=settings.chunk_tokens,
                               overlap=settings.chunk_overlap)
    chunks = Chunker(cfg, embed_fn=embed_fn).chunk(doc)
    # if the paper was unreadable but has an abstract (from the search layer), index the abstract so it stays searchable
    if not chunks and doc.abstract:
        from .chunker import _mk
        chunks = [_mk(doc, doc.abstract, None, 1, 1, 0, cfg, False, False)]
        chunks[0].section, chunks[0].section_canonical = "Abstract", "abstract"
        chunks[0].chunk_id = f"{doc.paper_id}_abstract_000"
    return chunks
