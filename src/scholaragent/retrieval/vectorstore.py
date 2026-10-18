"""Vector stores: Qdrant (production) and an in-memory GPU store (fast, used for experiments)."""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np

from ..config import settings
from ..ingestion.models import Chunk
from ..utils.logging import get_logger

log = get_logger("vector")


@dataclass
class Filter:
    paper_ids: set[str] | None = None
    year_from: int | None = None
    year_to: int | None = None
    sections: set[str] | None = None        # canonical section names

    def is_empty(self) -> bool:
        return not (self.paper_ids is not None or self.year_from or self.year_to or self.sections)


class VectorStore(Protocol):
    def upsert(self, chunks: Sequence[Chunk], vectors: np.ndarray) -> None: ...
    def delete_paper(self, paper_id: str) -> None: ...
    def search(self, qvec: np.ndarray, k: int, flt: Filter | None = None) -> list[tuple[str, float]]: ...
    def count(self) -> int: ...
    def paper_ids(self) -> set[str]: ...


def point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


class QdrantStore:
    """Dense vectors + payload filters. Payload keeps the chunk metadata so a hit is never 'just text'."""

    def __init__(self, collection: str | None = None, dim: int | None = None, url: str | None = None):
        from qdrant_client import QdrantClient
        self.name = collection or settings.qdrant_collection
        self.client = QdrantClient(url=url or settings.qdrant_url, api_key=settings.qdrant_api_key, timeout=60,
                                   prefer_grpc=False, check_compatibility=False)
        self.dim = dim
        self._ensured = False

    def _ensure(self, dim: int) -> None:
        if self._ensured:
            return
        from qdrant_client import models as qm
        if not self.client.collection_exists(self.name):
            self.client.create_collection(
                self.name, vectors_config=qm.VectorParams(size=dim, distance=qm.Distance.COSINE),
                hnsw_config=qm.HnswConfigDiff(m=16, ef_construct=128))
            for f, t in (("paper_id", qm.PayloadSchemaType.KEYWORD), ("section_canonical", qm.PayloadSchemaType.KEYWORD),
                         ("year", qm.PayloadSchemaType.INTEGER)):
                self.client.create_payload_index(self.name, f, t)
        self._ensured = True

    def upsert(self, chunks: Sequence[Chunk], vectors: np.ndarray) -> None:
        from qdrant_client import models as qm
        if not len(chunks):
            return
        self._ensure(vectors.shape[1])
        pts = [qm.PointStruct(id=point_id(c.chunk_id), vector=v.tolist(), payload={
            "chunk_id": c.chunk_id, "paper_id": c.paper_id, "section": c.section, "section_canonical": c.section_canonical,
            "page_start": c.page_start, "page_end": c.page_end, "year": c.year, "title": c.title, "doi": c.doi,
            "ordinal": c.ordinal}) for c, v in zip(chunks, vectors)]
        for i in range(0, len(pts), 256):
            self.client.upsert(self.name, points=pts[i:i + 256], wait=True)

    def delete_paper(self, paper_id: str) -> None:
        from qdrant_client import models as qm
        if self.client.collection_exists(self.name):
            self.client.delete(self.name, points_selector=qm.FilterSelector(
                filter=qm.Filter(must=[qm.FieldCondition(key="paper_id", match=qm.MatchValue(value=paper_id))])), wait=True)

    @staticmethod
    def _to_filter(flt: Filter | None):
        from qdrant_client import models as qm
        if flt is None or flt.is_empty():
            return None
        must = []
        if flt.paper_ids is not None:
            must.append(qm.FieldCondition(key="paper_id", match=qm.MatchAny(any=sorted(flt.paper_ids))))
        if flt.sections:
            must.append(qm.FieldCondition(key="section_canonical", match=qm.MatchAny(any=sorted(flt.sections))))
        if flt.year_from or flt.year_to:
            must.append(qm.FieldCondition(key="year", range=qm.Range(gte=flt.year_from, lte=flt.year_to)))
        return qm.Filter(must=must)

    def search(self, qvec: np.ndarray, k: int, flt: Filter | None = None) -> list[tuple[str, float]]:
        if not self.client.collection_exists(self.name):
            return []
        if flt is not None and flt.paper_ids is not None and not flt.paper_ids:
            return []
        res = self.client.query_points(self.name, query=qvec.tolist(), limit=k, query_filter=self._to_filter(flt),
                                       with_payload=["chunk_id"], with_vectors=False).points
        return [(p.payload["chunk_id"], float(p.score)) for p in res]

    def count(self) -> int:
        return self.client.count(self.name, exact=True).count if self.client.collection_exists(self.name) else 0

    def paper_ids(self) -> set[str]:
        ids: set[str] = set()
        if not self.client.collection_exists(self.name):
            return ids
        offset = None
        while True:
            pts, offset = self.client.scroll(self.name, limit=1000, offset=offset, with_payload=["paper_id"], with_vectors=False)
            ids.update(p.payload["paper_id"] for p in pts)
            if offset is None:
                break
        return ids

    def drop(self) -> None:
        if self.client.collection_exists(self.name):
            self.client.delete_collection(self.name)
        self._ensured = False


class MemoryStore:
    """Exact cosine search on the GPU (torch) - used for experiments with many index variants."""

    def __init__(self, device: str = "cuda"):
        import torch
        self.torch = torch
        self.device = device if torch.cuda.is_available() and device.startswith("cuda") else "cpu"
        self.ids: list[str] = []
        self.meta: dict[str, Chunk] = {}
        self.mat = None

    def upsert(self, chunks: Sequence[Chunk], vectors: np.ndarray) -> None:
        for c in chunks:
            if c.chunk_id in self.meta:
                self.delete_paper(c.paper_id)
                break
        new = self.torch.from_numpy(vectors).to(self.device, dtype=self.torch.float16)
        self.mat = new if self.mat is None else self.torch.cat([self.mat, new])
        self.ids += [c.chunk_id for c in chunks]
        self.meta.update({c.chunk_id: c for c in chunks})

    def delete_paper(self, paper_id: str) -> None:
        keep = [i for i, cid in enumerate(self.ids) if self.meta[cid].paper_id != paper_id]
        if len(keep) == len(self.ids):
            return
        self.mat = self.mat[self.torch.tensor(keep, device=self.device)] if keep else None
        gone = [cid for cid in self.ids if self.meta[cid].paper_id == paper_id]
        self.ids = [self.ids[i] for i in keep]
        for g in gone:
            self.meta.pop(g, None)

    def search(self, qvec: np.ndarray, k: int, flt: Filter | None = None) -> list[tuple[str, float]]:
        from ..utils.gpu import on_gpu
        return on_gpu(self._search, qvec, k, flt)

    def _search(self, qvec: np.ndarray, k: int, flt: Filter | None = None) -> list[tuple[str, float]]:
        if self.mat is None:
            return []
        q = self.torch.from_numpy(qvec).to(self.device, dtype=self.torch.float16)
        sims = (self.mat @ q).float()
        if flt is not None and not flt.is_empty():
            ok = np.array([self._ok(self.meta[c], flt) for c in self.ids])
            sims = sims.masked_fill(~self.torch.from_numpy(ok).to(self.device), -1e9)
        k = min(k, len(self.ids))
        v, i = self.torch.topk(sims, k)
        return [(self.ids[j], float(s)) for s, j in zip(v.tolist(), i.tolist()) if s > -1e8]

    @staticmethod
    def _ok(c: Chunk, f: Filter) -> bool:
        if f.paper_ids is not None and c.paper_id not in f.paper_ids:
            return False
        if f.sections and c.section_canonical not in f.sections:
            return False
        if f.year_from and (c.year or 0) < f.year_from:
            return False
        if f.year_to and (c.year or 9999) > f.year_to:
            return False
        return True

    def count(self) -> int:
        return len(self.ids)

    def paper_ids(self) -> set[str]:
        return {self.meta[c].paper_id for c in self.ids}
