"""Experiment harness: build many index variants (chunking x embedder) quickly on the GPU and evaluate retrieval."""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from ..config import settings
from ..ingestion.chunker import Chunker, ChunkerConfig
from ..ingestion.models import Chunk, ParsedDocument
from ..ingestion.pipeline import parse_pdf
from ..retrieval.embedder import Embedder
from ..retrieval.hybrid import ChunkIndex, Retriever
from ..retrieval.reranker import Reranker
from ..retrieval.vectorstore import Filter, MemoryStore
from ..utils.logging import get_logger
from .retrieval_metrics import QResult, aggregate, by_type, evaluate_ranking, grade

log = get_logger("harness")


class EmbedCache:
    """Disk cache of document embeddings keyed by (model, text) - re-running experiments is cheap."""

    def __init__(self, model_name: str):
        self.model = model_name
        self.path = settings.cache_dir / "emb" / (hashlib.md5(model_name.encode()).hexdigest() + ".npz")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.store: dict[str, np.ndarray] = {}
        if self.path.exists():
            z = np.load(self.path, allow_pickle=False)
            self.store = dict(zip(z["keys"].tolist(), z["vecs"]))
        self._new = 0

    @staticmethod
    def _k(t: str) -> str:
        return hashlib.md5(t.encode()).hexdigest()

    def encode(self, emb: Embedder, texts: Sequence[str]) -> np.ndarray:
        keys = [self._k(t) for t in texts]
        missing = [i for i, k in enumerate(keys) if k not in self.store]
        if missing:
            vecs = emb.encode_docs([texts[i] for i in missing], show=len(missing) > 500)
            for i, v in zip(missing, vecs):
                self.store[keys[i]] = v
            self._new += len(missing)
            self.save()
        return np.stack([self.store[k] for k in keys])

    def save(self):
        np.savez(self.path, keys=np.array(list(self.store)), vecs=np.stack(list(self.store.values())))


def load_docs(manifest_entries: Sequence[dict]) -> list[ParsedDocument]:
    docs = []
    for e in manifest_entries:
        try:
            ov = {k: e.get(k) for k in ("title", "authors", "year", "doi", "arxiv_id", "venue", "abstract") if e.get(k)}
            ov["paper_id"] = e["paper_id"]
            docs.append(parse_pdf(e["pdf"], overrides=ov, paper_id=e["paper_id"]))
        except Exception as ex:  # noqa: BLE001
            log.warning("skip %s: %r", e.get("paper_id"), ex)
    return docs


@dataclass
class Variant:
    name: str
    chunk_cfg: ChunkerConfig
    embed_model: str = settings.embed_model
    contextual: bool = True            # prepend 'title | section' to chunk text before embedding
    rerank_model: str | None = None


class Experiment:
    """Holds parsed docs and builds retrievers for variants."""

    def __init__(self, docs: list[ParsedDocument], fusion: str = "rrf", rrf_weights: tuple = (1.0, 1.0)):
        self.docs = docs
        self.fusion, self.rrf_weights = fusion, rrf_weights
        self._embedders: dict[str, Embedder] = {}
        self._rerankers: dict[str, Reranker] = {}
        self._caches: dict[str, EmbedCache] = {}
        self._chunks: dict[tuple, list[Chunk]] = {}

    def embedder(self, name: str) -> Embedder:
        if name not in self._embedders:
            self._embedders[name] = Embedder(name)
        return self._embedders[name]

    def reranker(self, name: str) -> Reranker:
        if name not in self._rerankers:
            self._rerankers[name] = Reranker(name)
        return self._rerankers[name]

    def chunks(self, cfg: ChunkerConfig, embed_model: str) -> list[Chunk]:
        key = (cfg.strategy, cfg.max_tokens, cfg.overlap, embed_model if cfg.strategy == "semantic" else "")
        if key not in self._chunks:
            ch = Chunker(cfg, embed_fn=self.embedder(embed_model) if cfg.strategy == "semantic" else None)
            out = []
            for d in self.docs:
                out.extend(ch.chunk(d))
            self._chunks[key] = out
        return self._chunks[key]

    def retriever(self, v: Variant, use_rerank: bool = False) -> Retriever:
        emb = self.embedder(v.embed_model)
        chunks = self.chunks(v.chunk_cfg, v.embed_model)
        cache = self._caches.setdefault(v.embed_model, EmbedCache(v.embed_model))
        vecs = cache.encode(emb, [c.embed_text(v.contextual) for c in chunks])
        vs = MemoryStore(settings.device)
        vs.upsert(chunks, vecs)
        idx = ChunkIndex()
        idx.set_chunks(chunks)
        rr = self.reranker(v.rerank_model or settings.rerank_model) if use_rerank else None
        return Retriever(idx, vs, emb, rr, fusion=self.fusion, rrf_weights=self.rrf_weights)


def run_retrieval(retriever: Retriever, questions: Sequence[dict], mode: str, k: int = 10, n_candidates: int = 30,
                  **kw) -> tuple[list[QResult], dict, list[list]]:
    results, latencies, all_hits = [], [], []
    for q in list(questions)[:3]:                       # warm-up (CUDA kernels, caches) - not timed
        retriever.search(q["question"], k=k, mode=mode, n_candidates=n_candidates, **kw)
    by_paper: dict[str, list] = {}
    for c in retriever.index.chunks.values():
        by_paper.setdefault(c.paper_id, []).append(c)
    for q in questions:
        t0 = time.perf_counter()
        hits = retriever.search(q["question"], k=k, mode=mode, n_candidates=n_candidates, **kw)
        latencies.append(time.perf_counter() - t0)
        chunks = [h.chunk for h in hits]
        allg = sorted((g for p in {e["paper_id"] for e in q["evidences"]} for c in by_paper.get(p, []) if (g := grade(c, q["evidences"])) > 0), reverse=True)
        results.append(evaluate_ranking(q, chunks, max_k=k, n_relevant_in_corpus=sum(g == 3 for g in allg), ideal_grades=allg))
        all_hits.append(hits)
    agg = aggregate(results)
    agg["latency_ms_mean"] = round(1000 * float(np.mean(latencies)), 1)
    agg["latency_ms_p95"] = round(1000 * float(np.percentile(latencies, 95)), 1)
    return results, agg, all_hits


def save_json(obj, name: str) -> Path:
    p = settings.results_dir / f"{name}.json"
    p.write_text(json.dumps(obj, indent=1, ensure_ascii=False, default=str))
    return p
