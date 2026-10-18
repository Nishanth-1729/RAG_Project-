"""Hybrid retriever: dense + BM25 -> RRF -> cross-encoder rerank, with evidence-preserving results."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

from ..ingestion.models import Chunk
from ..utils.logging import get_logger, trace
from .embedder import Embedder
from .fusion import max_per_group, rrf, rsf
from .reranker import Reranker
from .sparse import BM25
from .vectorstore import Filter, VectorStore

log = get_logger("retrieve")

MODES = ("dense", "bm25", "hybrid", "hybrid_rerank", "dense_rerank", "bm25_rerank")   # *_rerank single-source variants are for ablations


@dataclass
class Hit:
    chunk: Chunk
    score: float
    rank: int = 0
    scores: dict[str, float] = field(default_factory=dict)     # per-stage scores (dense/bm25/rrf/rerank)
    ranks: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        c = self.chunk
        return {"chunk_id": c.chunk_id, "paper_id": c.paper_id, "title": c.title, "authors": c.authors, "year": c.year,
                "section": c.section, "section_canonical": c.section_canonical, "page_start": c.page_start,
                "page_end": c.page_end, "text": c.text, "doi": c.doi, "score": round(self.score, 5),
                "scores": {k: round(v, 5) for k, v in self.scores.items()}, "rank": self.rank}


class ChunkIndex:
    """In-memory view of chunk text + BM25 index for a set of chunks."""

    def __init__(self, bm25_text: Callable[[Chunk], str] | None = None):
        self.chunks: dict[str, Chunk] = {}
        self.bm25 = BM25()
        self._dirty = True
        self._bm25_text = bm25_text or (lambda c: f"{c.section}\n{c.text}")

    def set_chunks(self, chunks: Sequence[Chunk]) -> None:
        self.chunks = {c.chunk_id: c for c in chunks}
        self._dirty = True

    def add(self, chunks: Sequence[Chunk]) -> None:
        for c in chunks:
            self.chunks[c.chunk_id] = c
        self._dirty = True

    def remove_paper(self, paper_id: str) -> None:
        self.chunks = {k: v for k, v in self.chunks.items() if v.paper_id != paper_id}
        self._dirty = True

    def ensure_built(self) -> None:
        if self._dirty:
            cs = list(self.chunks.values())
            self.bm25.fit([c.chunk_id for c in cs], [c.paper_id for c in cs], [self._bm25_text(c) for c in cs])
            self._dirty = False


class Retriever:
    def __init__(self, index: ChunkIndex, vstore: VectorStore, embedder: Embedder, reranker: Reranker | None = None,
                 contextual_rerank: bool = True, fusion: str = "rrf", rrf_weights: Sequence[float] = (1.0, 1.0)):
        self.index, self.vs, self.emb, self.reranker = index, vstore, embedder, reranker
        self.contextual_rerank = contextual_rerank
        self.fusion, self.rrf_weights = fusion, tuple(rrf_weights)

    # ------------------------------------------------------------- primitives
    def dense(self, query: str, k: int, flt: Filter | None) -> list[tuple[str, float]]:
        return self.vs.search(self.emb.encode_queries([query])[0], k, flt)

    def sparse(self, query: str, k: int, flt: Filter | None) -> list[tuple[str, float]]:
        self.index.ensure_built()
        bm = self.index.bm25
        mask = None
        if flt is not None and not flt.is_empty():
            cs = [self.index.chunks[i] for i in bm.ids]
            mask = np.array([_pass(c, flt) for c in cs])
        return bm.search(query, k, mask=mask)

    # ------------------------------------------------------------------ search
    def search(self, query: str, k: int = 8, mode: str = "hybrid_rerank", flt: Filter | None = None,
               n_candidates: int = 30, max_per_paper: int | None = None, rrf_k: int = 60,
               rrf_weights: Sequence[float] | None = None, fusion: str | None = None) -> list[Hit]:
        assert mode in MODES, mode
        t0 = time.perf_counter()
        self.index.ensure_built()
        hits: dict[str, Hit] = {}

        def hit_for(cid: str) -> Hit | None:
            c = self.index.chunks.get(cid)
            if c is None:
                return None
            return hits.setdefault(cid, Hit(chunk=c, score=0.0))

        d_list: list[tuple[str, float]] = []
        s_list: list[tuple[str, float]] = []
        if mode in ("dense", "hybrid", "hybrid_rerank", "dense_rerank"):
            d_list = self.dense(query, n_candidates, flt)
            for r, (cid, sc) in enumerate(d_list, 1):
                if (h := hit_for(cid)):
                    h.scores["dense"], h.ranks["dense"] = sc, r
        if mode in ("bm25", "hybrid", "hybrid_rerank", "bm25_rerank"):
            s_list = self.sparse(query, n_candidates, flt)
            for r, (cid, sc) in enumerate(s_list, 1):
                if (h := hit_for(cid)):
                    h.scores["bm25"], h.ranks["bm25"] = sc, r

        if mode in ("dense", "dense_rerank"):
            order = [(cid, sc) for cid, sc in d_list]
        elif mode in ("bm25", "bm25_rerank"):
            order = [(cid, sc) for cid, sc in s_list]
        else:
            fusion = fusion or self.fusion
            rrf_weights = rrf_weights or self.rrf_weights
            if fusion == "rsf":          # relative score fusion: min-max normalised scores, weighted sum
                order = rsf(d_list, s_list, rrf_weights)
            else:
                order = rrf([[c for c, _ in d_list], [c for c, _ in s_list]], k=rrf_k, weights=rrf_weights)
            for cid, sc in order:
                if cid in hits:
                    hits[cid].scores["fusion"] = sc
        ordered = [hits[cid] for cid, sc in order if cid in hits]
        for h, (_, sc) in zip(ordered, order):
            h.score = sc

        if mode.endswith("_rerank") and ordered:
            if self.reranker is None:
                raise ValueError("hybrid_rerank needs a reranker")
            pool = ordered[:n_candidates]
            texts = [self._rerank_text(h.chunk) for h in pool]
            sc = self.reranker.score(query, texts)
            for h, s in zip(pool, sc):
                h.scores["rerank"] = float(s)
                h.score = float(s)
            ordered = sorted(pool, key=lambda h: h.score, reverse=True)

        ordered = max_per_group(ordered, lambda h: h.chunk.paper_id, max_per_paper)[:k]
        for i, h in enumerate(ordered, 1):
            h.rank = i
        trace("retrieve", mode=mode, k=k, n=len(ordered), latency_s=round(time.perf_counter() - t0, 3))
        return ordered

    def _rerank_text(self, c: Chunk) -> str:
        return f"{c.title} | {c.section}\n{c.text}" if self.contextual_rerank else c.text


def _pass(c: Chunk, f: Filter) -> bool:
    if f.paper_ids is not None and c.paper_id not in f.paper_ids:
        return False
    if f.sections and c.section_canonical not in f.sections:
        return False
    if f.year_from and (c.year or 0) < f.year_from:
        return False
    if f.year_to and (c.year or 9999) > f.year_to:
        return False
    return True
