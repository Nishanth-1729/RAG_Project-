from __future__ import annotations

import datetime as dt

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Index, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def _now() -> dt.datetime:
    return dt.datetime.utcnow()


class Base(DeclarativeBase):
    pass


class PaperRow(Base):
    __tablename__ = "papers"
    paper_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    title: Mapped[str] = mapped_column(Text, default="")
    authors: Mapped[list] = mapped_column(JSON, default=list)
    year: Mapped[int | None] = mapped_column(Integer, nullable=True)
    doi: Mapped[str | None] = mapped_column(String(256), nullable=True)
    arxiv_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    venue: Mapped[str | None] = mapped_column(String(256), nullable=True)
    abstract: Mapped[str] = mapped_column(Text, default="")
    source: Mapped[str] = mapped_column(String(64), default="upload")
    path: Mapped[str | None] = mapped_column(Text, nullable=True)
    n_pages: Mapped[int] = mapped_column(Integer, default=0)
    n_chunks: Mapped[int] = mapped_column(Integer, default=0)
    citation_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    status: Mapped[str] = mapped_column(String(32), default="indexed")   # indexed | metadata_only | failed
    warnings: Mapped[list] = mapped_column(JSON, default=list)
    embed_model: Mapped[str | None] = mapped_column(String(128), nullable=True)
    added_at: Mapped[dt.datetime] = mapped_column(DateTime, default=_now)


class ChunkRow(Base):
    __tablename__ = "chunks"
    chunk_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    paper_id: Mapped[str] = mapped_column(String(64), ForeignKey("papers.paper_id", ondelete="CASCADE"), index=True)
    ordinal: Mapped[int] = mapped_column(Integer, default=0)
    section: Mapped[str] = mapped_column(Text, default="")
    section_canonical: Mapped[str] = mapped_column(String(32), default="other")
    page_start: Mapped[int] = mapped_column(Integer, default=1)
    page_end: Mapped[int] = mapped_column(Integer, default=1)
    text: Mapped[str] = mapped_column(Text)
    n_tokens: Mapped[int] = mapped_column(Integer, default=0)
    has_equation: Mapped[bool] = mapped_column(Boolean, default=False)
    has_table: Mapped[bool] = mapped_column(Boolean, default=False)
    strategy: Mapped[str] = mapped_column(String(16), default="section")


class CollectionRow(Base):
    __tablename__ = "collections"
    collection_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(256))
    kind: Mapped[str] = mapped_column(String(16), default="user")      # user | research
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=_now)


class CollectionPaper(Base):
    __tablename__ = "collection_papers"
    collection_id: Mapped[str] = mapped_column(String(64), ForeignKey("collections.collection_id", ondelete="CASCADE"), primary_key=True)
    paper_id: Mapped[str] = mapped_column(String(64), ForeignKey("papers.paper_id", ondelete="CASCADE"), primary_key=True)
    relevance: Mapped[float | None] = mapped_column(Float, nullable=True)


class AnalysisRow(Base):
    __tablename__ = "analyses"
    paper_id: Mapped[str] = mapped_column(String(64), ForeignKey("papers.paper_id", ondelete="CASCADE"), primary_key=True)
    data: Mapped[dict] = mapped_column(JSON, default=dict)
    model: Mapped[str] = mapped_column(String(128), default="")
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=_now)


class RunRow(Base):
    __tablename__ = "runs"
    run_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    question: Mapped[str] = mapped_column(Text, default="")
    mode: Mapped[str] = mapped_column(String(16), default="deep")      # deep | my_papers
    params: Mapped[dict] = mapped_column(JSON, default=dict)
    status: Mapped[str] = mapped_column(String(16), default="queued")  # queued|running|done|failed
    events: Mapped[list] = mapped_column(JSON, default=list)
    result: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=_now)
    finished_at: Mapped[dt.datetime | None] = mapped_column(DateTime, nullable=True)


class EdgeRow(Base):
    """Knowledge graph edges: (:Paper)-[:CITES]->(:Paper), (:Paper)-[:PROPOSES]->(:Method) ..."""
    __tablename__ = "edges"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    src: Mapped[str] = mapped_column(String(256), index=True)
    dst: Mapped[str] = mapped_column(String(256), index=True)
    relation: Mapped[str] = mapped_column(String(32))
    src_type: Mapped[str] = mapped_column(String(16), default="Paper")
    dst_type: Mapped[str] = mapped_column(String(16), default="Paper")
    meta: Mapped[dict] = mapped_column(JSON, default=dict)
    __table_args__ = (UniqueConstraint("src", "dst", "relation", name="uq_edge"),)
