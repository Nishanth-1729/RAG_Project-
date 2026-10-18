"""The Library ties everything together: PDFs -> parse -> chunk -> embed -> Qdrant/DB/BM25, and builds Retrievers.

Papers are indexed ONCE (globally) and cached; collections / research runs are just sets of paper ids,
so the same paper never costs parsing or embedding twice.
"""
from __future__ import annotations

import re
import threading
from pathlib import Path
from typing import Callable, Sequence

from ..config import settings
from ..database.store import Store, get_store
from ..ingestion.chunker import ChunkerConfig
from ..ingestion.models import Chunk, ParsedDocument
from ..ingestion.pipeline import chunk_document, parse_pdf
from ..utils.logging import get_logger, timed
from ..utils.text import slugify
from .embedder import Embedder
from .hybrid import ChunkIndex, Retriever
from .reranker import Reranker
from .vectorstore import Filter, QdrantStore, VectorStore

log = get_logger("library")


class Library:
    def __init__(self, store: Store | None = None, embedder: Embedder | None = None, reranker: Reranker | None = None,
                 vstore: VectorStore | None = None, chunk_cfg: ChunkerConfig | None = None, use_rerank: bool = True):
        self.store = store or get_store()
        self.embedder = embedder or Embedder()
        self.reranker = reranker or (Reranker() if use_rerank else None)
        self.chunk_cfg = chunk_cfg or ChunkerConfig(strategy=settings.chunk_strategy, max_tokens=settings.chunk_tokens,
                                                    overlap=settings.chunk_overlap)
        coll = f"{settings.qdrant_collection}__{slugify(Path(self.embedder.model_name).name, 30)}__{self.chunk_cfg.strategy}{self.chunk_cfg.max_tokens}"
        self.vs: VectorStore = vstore or QdrantStore(collection=coll)
        self.index = ChunkIndex()
        self._loaded = False
        self._lock = threading.RLock()
        self.retriever = Retriever(self.index, self.vs, self.embedder, self.reranker, fusion=settings.fusion,
                                   rrf_weights=tuple(float(x) for x in settings.fusion_weights.split(",")))

    # ----------------------------------------------------------------- loading
    def load(self) -> "Library":
        with self._lock:
            if not self._loaded:
                chunks = self.store.get_chunks()
                self.index.set_chunks(chunks)
                self._loaded = True
                log.info("library loaded: %d chunks", len(chunks))
        return self

    # --------------------------------------------------------------- ingestion
    def has_paper(self, paper_id: str) -> bool:
        p = self.store.get_paper(paper_id)
        return bool(p and p["status"] == "indexed" and p["n_chunks"] > 0 and p.get("embed_model") == self.embedder.model_name)

    def ingest_pdf(self, path: str | Path, overrides: dict | None = None, collection_id: str | None = None,
                   force: bool = False) -> dict:
        """Parse + chunk + embed + store one PDF. Never raises: returns {"ok": bool, ...}."""
        path = Path(path)
        self.load()
        try:
            with timed("ingest_pdf", file=path.name) as info:
                doc = parse_pdf(path, overrides)
                if not force and not self.has_paper(doc.paper_id):
                    dup = self.store.find_duplicate(doc.title, doc.doi, doc.arxiv_id, exclude_id=doc.paper_id)
                    if dup and dup["status"] == "indexed":      # same work already indexed under another id: reuse it
                        if collection_id:
                            self.store.add_to_collection(collection_id, [dup["paper_id"]])
                        info["duplicate_of"] = dup["paper_id"]
                        return {"ok": True, "paper_id": dup["paper_id"], "title": dup["title"], "cached": True,
                                "duplicate_of": dup["paper_id"], "n_chunks": dup["n_chunks"]}
                if not force and self.has_paper(doc.paper_id):
                    if collection_id:
                        self.store.add_to_collection(collection_id, [doc.paper_id])
                    info["cached"] = True
                    return {"ok": True, "paper_id": doc.paper_id, "title": doc.title, "cached": True,
                            "n_chunks": self.store.get_paper(doc.paper_id)["n_chunks"]}
                chunks = chunk_document(doc, self.chunk_cfg, embed_fn=self.embedder if self.chunk_cfg.strategy == "semantic" else None)
                if doc.needs_ocr and not chunks:
                    self.store.upsert_paper(doc, 0, status="failed")
                    return {"ok": False, "paper_id": doc.paper_id, "title": doc.title, "error": "scanned PDF without text layer (OCR required)"}
                if not chunks:
                    self.store.upsert_paper(doc, 0, status="failed")
                    return {"ok": False, "paper_id": doc.paper_id, "title": doc.title, "error": "no text could be extracted"}
                vecs = self.embedder.encode_docs([c.embed_text(True) for c in chunks])
                self.vs.delete_paper(doc.paper_id)
                self.vs.upsert(chunks, vecs)
                self.store.upsert_paper(doc, len(chunks), embed_model=self.embedder.model_name,
                                        citation_count=(overrides or {}).get("citation_count"))
                self.store.replace_chunks(doc.paper_id, chunks)
                with self._lock:
                    self.index.remove_paper(doc.paper_id)
                    self.index.add(chunks)
                if collection_id:
                    self.store.add_to_collection(collection_id, [doc.paper_id])
                info["n_chunks"] = len(chunks)
                return {"ok": True, "paper_id": doc.paper_id, "title": doc.title, "n_chunks": len(chunks),
                        "cached": False, "warnings": doc.warnings}
        except Exception as e:  # noqa: BLE001  - a broken PDF must never kill a 30-paper run
            log.warning("ingest failed for %s: %r", path.name, e)
            return {"ok": False, "paper_id": None, "title": path.stem, "error": repr(e)[:300]}

    def ingest_many(self, paths: Sequence[Path], collection_id: str | None = None,
                    on_progress: Callable[[int, int, dict], None] | None = None) -> list[dict]:
        out = []
        for i, p in enumerate(paths, 1):
            r = self.ingest_pdf(p, collection_id=collection_id)
            out.append(r)
            if on_progress:
                on_progress(i, len(paths), r)
        return out

    def add_metadata_only(self, rec: dict) -> None:
        """Keep a paper whose full text is unavailable: abstract becomes a single searchable chunk."""
        from ..ingestion.chunker import _mk
        self.load()
        if not rec.get("abstract"):
            self.store.upsert_metadata_only(rec)
            return
        doc = ParsedDocument(paper_id=rec["paper_id"], path="", n_pages=0, title=rec["title"], authors=rec.get("authors", []),
                             abstract=rec["abstract"], year=rec.get("year"), doi=rec.get("doi"), arxiv_id=rec.get("arxiv_id"),
                             venue=rec.get("venue"), source=",".join(rec.get("sources", [])) or "search")
        cfg = self.chunk_cfg
        c = _mk(doc, rec["abstract"], None, 1, 1, 0, cfg, False, False)
        c.section, c.section_canonical, c.chunk_id = "Abstract (full text unavailable)", "abstract", f"{doc.paper_id}_abstract_000"
        vecs = self.embedder.encode_docs([c.embed_text(True)])
        self.vs.delete_paper(doc.paper_id)
        self.vs.upsert([c], vecs)
        self.store.upsert_paper(doc, 1, status="abstract_only", embed_model=self.embedder.model_name,
                                citation_count=rec.get("citation_count"))
        self.store.replace_chunks(doc.paper_id, [c])
        with self._lock:
            self.index.remove_paper(doc.paper_id)
            self.index.add([c])

    def remove_paper(self, paper_id: str) -> None:
        self.load()
        self.vs.delete_paper(paper_id)
        self.store.delete_paper(paper_id)
        with self._lock:
            self.index.remove_paper(paper_id)

    # ----------------------------------------------------------------- queries
    def search(self, query: str, paper_ids: Sequence[str] | None = None, k: int = 8, mode: str = "hybrid_rerank",
               **kw):
        self.load()
        flt = Filter(paper_ids=set(paper_ids)) if paper_ids is not None else kw.pop("flt", None)
        return self.retriever.search(query, k=k, mode=mode if self.reranker or mode != "hybrid_rerank" else "hybrid", flt=flt, **kw)

    def chunks_for(self, paper_id: str) -> list[Chunk]:
        self.load()
        return sorted((c for c in self.index.chunks.values() if c.paper_id == paper_id), key=lambda c: c.ordinal)

    def stats(self) -> dict:
        self.load()
        papers = self.store.list_papers()
        return {"papers": len(papers), "indexed": sum(p["status"] == "indexed" for p in papers),
                "chunks": len(self.index.chunks), "vectors": self.vs.count(), "embed_model": self.embedder.model_name}


_lib: Library | None = None


def get_library() -> Library:
    global _lib
    if _lib is None:
        _lib = Library().load()
    return _lib
