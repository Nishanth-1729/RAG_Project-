"""Metadata database (SQLite by default, PostgreSQL via DATABASE_URL)."""
from __future__ import annotations

import datetime as dt
import threading
import uuid
from contextlib import contextmanager
from typing import Any, Iterator, Sequence

from sqlalchemy import create_engine, delete, event, select
from sqlalchemy.orm import Session, sessionmaker

from ..config import settings
from ..ingestion.models import Chunk, ParsedDocument
from .models import (AnalysisRow, Base, ChunkRow, CollectionPaper, CollectionRow, EdgeRow, PaperRow, RunRow)


def _engine(url: str):
    kw: dict[str, Any] = {}
    if url.startswith("sqlite"):
        kw["connect_args"] = {"check_same_thread": False, "timeout": 30}
    eng = create_engine(url, future=True, pool_pre_ping=True, **kw)
    if url.startswith("sqlite"):
        @event.listens_for(eng, "connect")
        def _pragmas(dbapi_conn, _):  # noqa: ANN001
            cur = dbapi_conn.cursor()
            cur.execute("PRAGMA journal_mode=WAL")
            cur.execute("PRAGMA foreign_keys=ON")
            cur.execute("PRAGMA synchronous=NORMAL")
            cur.close()
    return eng


class Store:
    def __init__(self, url: str | None = None):
        self.engine = _engine(url or settings.database_url)
        Base.metadata.create_all(self.engine)
        self._S = sessionmaker(self.engine, expire_on_commit=False, future=True)
        self._lock = threading.RLock()

    @contextmanager
    def session(self) -> Iterator[Session]:
        with self._lock:
            s = self._S()
            try:
                yield s
                s.commit()
            except Exception:
                s.rollback()
                raise
            finally:
                s.close()

    # ------------------------------------------------------------------ papers
    def upsert_paper(self, doc: ParsedDocument, n_chunks: int, status: str = "indexed", embed_model: str | None = None,
                     citation_count: int | None = None) -> None:
        with self.session() as s:
            row = s.get(PaperRow, doc.paper_id) or PaperRow(paper_id=doc.paper_id)
            row.title, row.authors, row.year = doc.title, doc.authors, doc.year
            row.doi, row.arxiv_id, row.venue, row.abstract = doc.doi, doc.arxiv_id, doc.venue, doc.abstract
            row.source, row.path, row.n_pages, row.n_chunks = doc.source, doc.path, doc.n_pages, n_chunks
            row.status, row.warnings, row.embed_model = status, doc.warnings, embed_model
            if citation_count is not None:
                row.citation_count = citation_count
            s.add(row)

    def upsert_metadata_only(self, rec: dict) -> None:
        with self.session() as s:
            row = s.get(PaperRow, rec["paper_id"]) or PaperRow(paper_id=rec["paper_id"])
            row.title, row.authors, row.year = rec.get("title", ""), rec.get("authors", []), rec.get("year")
            row.doi, row.arxiv_id, row.venue = rec.get("doi"), rec.get("arxiv_id"), rec.get("venue")
            row.abstract, row.source = rec.get("abstract", ""), ",".join(rec.get("sources", [])) or "search"
            row.citation_count, row.status = rec.get("citation_count"), "metadata_only"
            s.add(row)

    def find_duplicate(self, title: str, doi: str | None, arxiv_id: str | None, exclude_id: str | None = None) -> dict | None:
        """Existing, indexed paper that is the same work (DOI / arXiv id / normalised title)."""
        from ..utils.text import normalize_title
        nt = normalize_title(title)
        with self.session() as s:
            for r in s.scalars(select(PaperRow).where(PaperRow.status.in_(["indexed", "abstract_only"]))):
                if r.paper_id == exclude_id:
                    continue
                if (doi and r.doi and doi.lower() == r.doi.lower()) or (arxiv_id and r.arxiv_id and arxiv_id == r.arxiv_id) \
                        or (len(nt) > 20 and normalize_title(r.title) == nt):
                    return self._paper_dict(r)
        return None

    def get_paper(self, paper_id: str) -> dict | None:
        with self.session() as s:
            r = s.get(PaperRow, paper_id)
            return self._paper_dict(r) if r else None

    def list_papers(self, paper_ids: Sequence[str] | None = None, status: str | None = None) -> list[dict]:
        with self.session() as s:
            q = select(PaperRow)
            if paper_ids is not None:
                q = q.where(PaperRow.paper_id.in_(list(paper_ids)))
            if status:
                q = q.where(PaperRow.status == status)
            return [self._paper_dict(r) for r in s.scalars(q.order_by(PaperRow.added_at.desc()))]

    def delete_paper(self, paper_id: str) -> None:
        with self.session() as s:
            s.execute(delete(ChunkRow).where(ChunkRow.paper_id == paper_id))
            s.execute(delete(CollectionPaper).where(CollectionPaper.paper_id == paper_id))
            s.execute(delete(AnalysisRow).where(AnalysisRow.paper_id == paper_id))
            s.execute(delete(PaperRow).where(PaperRow.paper_id == paper_id))

    @staticmethod
    def _paper_dict(r: PaperRow) -> dict:
        return {c.name: getattr(r, c.name) for c in r.__table__.columns if c.name != "added_at"} | {"added_at": r.added_at.isoformat()}

    # ------------------------------------------------------------------ chunks
    def replace_chunks(self, paper_id: str, chunks: Sequence[Chunk]) -> None:
        with self.session() as s:
            s.execute(delete(ChunkRow).where(ChunkRow.paper_id == paper_id))
            s.add_all([ChunkRow(chunk_id=c.chunk_id, paper_id=c.paper_id, ordinal=c.ordinal, section=c.section,
                                section_canonical=c.section_canonical, page_start=c.page_start, page_end=c.page_end,
                                text=c.text, n_tokens=c.n_tokens, has_equation=c.has_equation,
                                has_table=c.has_table, strategy=c.strategy) for c in chunks])

    def get_chunks(self, paper_ids: Sequence[str] | None = None) -> list[Chunk]:
        with self.session() as s:
            q = select(ChunkRow, PaperRow).join(PaperRow, PaperRow.paper_id == ChunkRow.paper_id)
            if paper_ids is not None:
                q = q.where(ChunkRow.paper_id.in_(list(paper_ids)))
            q = q.order_by(ChunkRow.paper_id, ChunkRow.ordinal)
            return [self._chunk(c, p) for c, p in s.execute(q).all()]

    def get_chunk_by_ids(self, ids: Sequence[str]) -> dict[str, Chunk]:
        if not ids:
            return {}
        with self.session() as s:
            q = select(ChunkRow, PaperRow).join(PaperRow, PaperRow.paper_id == ChunkRow.paper_id).where(ChunkRow.chunk_id.in_(list(ids)))
            return {c.chunk_id: self._chunk(c, p) for c, p in s.execute(q).all()}

    def neighbors(self, paper_id: str, ordinal: int, radius: int = 1) -> list[Chunk]:
        with self.session() as s:
            q = (select(ChunkRow, PaperRow).join(PaperRow, PaperRow.paper_id == ChunkRow.paper_id)
                 .where(ChunkRow.paper_id == paper_id, ChunkRow.ordinal.between(ordinal - radius, ordinal + radius))
                 .order_by(ChunkRow.ordinal))
            return [self._chunk(c, p) for c, p in s.execute(q).all()]

    @staticmethod
    def _chunk(c: ChunkRow, p: PaperRow) -> Chunk:
        return Chunk(chunk_id=c.chunk_id, paper_id=c.paper_id, text=c.text, section=c.section,
                     section_canonical=c.section_canonical, page_start=c.page_start, page_end=c.page_end,
                     ordinal=c.ordinal, strategy=c.strategy, n_tokens=c.n_tokens, has_equation=c.has_equation,
                     has_table=c.has_table, title=p.title, authors=list(p.authors or []), year=p.year, doi=p.doi,
                     source=p.source)

    # ------------------------------------------------------------- collections
    def create_collection(self, name: str, kind: str = "user", collection_id: str | None = None) -> str:
        cid = collection_id or f"{kind}_{uuid.uuid4().hex[:10]}"
        with self.session() as s:
            if not s.get(CollectionRow, cid):
                s.add(CollectionRow(collection_id=cid, name=name, kind=kind))
        return cid

    def add_to_collection(self, cid: str, paper_ids: Sequence[str], relevance: dict[str, float] | None = None) -> None:
        with self.session() as s:
            if s.get(CollectionRow, cid) is None:        # collections are created on first use
                s.add(CollectionRow(collection_id=cid, name=cid.replace("_", " ").title(), kind="user"))
                s.flush()
            for pid in paper_ids:
                row = s.get(CollectionPaper, (cid, pid))
                if row is None:
                    s.add(CollectionPaper(collection_id=cid, paper_id=pid, relevance=(relevance or {}).get(pid)))
                elif relevance and pid in relevance:
                    row.relevance = relevance[pid]

    def remove_from_collection(self, cid: str, paper_id: str) -> None:
        with self.session() as s:
            s.execute(delete(CollectionPaper).where(CollectionPaper.collection_id == cid, CollectionPaper.paper_id == paper_id))

    def collection_papers(self, cid: str) -> list[str]:
        with self.session() as s:
            return list(s.scalars(select(CollectionPaper.paper_id).where(CollectionPaper.collection_id == cid)))

    def list_collections(self) -> list[dict]:
        with self.session() as s:
            out = []
            for c in s.scalars(select(CollectionRow).order_by(CollectionRow.created_at.desc())):
                n = len(list(s.scalars(select(CollectionPaper.paper_id).where(CollectionPaper.collection_id == c.collection_id))))
                out.append({"collection_id": c.collection_id, "name": c.name, "kind": c.kind, "n_papers": n,
                            "created_at": c.created_at.isoformat()})
            return out

    # ---------------------------------------------------------------- analyses
    def save_analysis(self, paper_id: str, data: dict, model: str) -> None:
        with self.session() as s:
            row = s.get(AnalysisRow, paper_id) or AnalysisRow(paper_id=paper_id)
            row.data, row.model, row.created_at = data, model, dt.datetime.utcnow()
            s.add(row)

    def get_analysis(self, paper_id: str) -> dict | None:
        with self.session() as s:
            r = s.get(AnalysisRow, paper_id)
            return r.data if r else None

    # -------------------------------------------------------------------- runs
    def create_run(self, question: str, mode: str, params: dict) -> str:
        rid = uuid.uuid4().hex[:12]
        with self.session() as s:
            s.add(RunRow(run_id=rid, question=question, mode=mode, params=params, status="queued", events=[]))
        return rid

    def update_run(self, run_id: str, **fields: Any) -> None:
        with self.session() as s:
            r = s.get(RunRow, run_id)
            if r is None:
                return
            for k, v in fields.items():
                setattr(r, k, v)

    def add_run_event(self, run_id: str, event_: dict) -> None:
        with self.session() as s:
            r = s.get(RunRow, run_id)
            if r is not None:
                r.events = [*r.events, event_]

    def get_run(self, run_id: str) -> dict | None:
        with self.session() as s:
            r = s.get(RunRow, run_id)
            if not r:
                return None
            return {"run_id": r.run_id, "question": r.question, "mode": r.mode, "params": r.params, "status": r.status,
                    "events": r.events, "result": r.result, "error": r.error, "created_at": r.created_at.isoformat(),
                    "finished_at": r.finished_at.isoformat() if r.finished_at else None}

    def list_runs(self, limit: int = 50) -> list[dict]:
        with self.session() as s:
            return [{"run_id": r.run_id, "question": r.question, "mode": r.mode, "status": r.status,
                     "created_at": r.created_at.isoformat()}
                    for r in s.scalars(select(RunRow).order_by(RunRow.created_at.desc()).limit(limit))]

    # ------------------------------------------------------------------- edges
    def upsert_edges(self, edges: Sequence[dict]) -> None:
        with self.session() as s:
            for e in edges:
                ex = s.scalar(select(EdgeRow).where(EdgeRow.src == e["src"], EdgeRow.dst == e["dst"], EdgeRow.relation == e["relation"]))
                if ex is None:
                    s.add(EdgeRow(src=e["src"], dst=e["dst"], relation=e["relation"], src_type=e.get("src_type", "Paper"),
                                  dst_type=e.get("dst_type", "Paper"), meta=e.get("meta", {})))

    def get_edges(self, nodes: Sequence[str] | None = None) -> list[dict]:
        with self.session() as s:
            q = select(EdgeRow)
            if nodes is not None:
                q = q.where(EdgeRow.src.in_(list(nodes)) | EdgeRow.dst.in_(list(nodes)))
            return [{"src": e.src, "dst": e.dst, "relation": e.relation, "src_type": e.src_type,
                     "dst_type": e.dst_type, "meta": e.meta} for e in s.scalars(q)]


_store: Store | None = None


def get_store() -> Store:
    global _store
    if _store is None:
        _store = Store()
    return _store
