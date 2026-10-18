"""Plain data models for the ingestion layer (JSON-serialisable)."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Block:
    page: int                 # 1-based
    text: str
    kind: str = "paragraph"   # paragraph | heading | equation | table | caption | reference
    font_size: float = 0.0
    bold: bool = False
    bbox: tuple[float, float, float, float] = (0, 0, 0, 0)


@dataclass
class Paragraph:
    text: str
    page_start: int
    page_end: int
    kind: str = "paragraph"   # paragraph | equation | table | caption


@dataclass
class Section:
    title: str                # raw heading text, e.g. "3.2 Hierarchical Approximation"
    number: str = ""          # "3.2"
    level: int = 1
    canonical: str = "other"  # abstract|introduction|related_work|background|method|experiments|results|discussion|limitations|conclusion|references|appendix|other
    page_start: int = 1
    page_end: int = 1
    paragraphs: list[Paragraph] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n\n".join(p.text for p in self.paragraphs)

    @property
    def label(self) -> str:
        return self.title


@dataclass
class ParsedDocument:
    paper_id: str
    path: str
    n_pages: int
    title: str = ""
    authors: list[str] = field(default_factory=list)
    abstract: str = ""
    year: int | None = None
    doi: str | None = None
    arxiv_id: str | None = None
    venue: str | None = None
    source: str = "upload"
    needs_ocr: bool = False
    sections: list[Section] = field(default_factory=list)
    references_text: str = ""
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ParsedDocument":
        d = dict(d)
        secs = []
        for s in d.pop("sections", []):
            paras = [Paragraph(**p) for p in s.pop("paragraphs", [])]
            secs.append(Section(paragraphs=paras, **s))
        return cls(sections=secs, **d)

    def full_text(self, include_references: bool = False) -> str:
        parts = []
        for s in self.sections:
            if s.canonical == "references" and not include_references:
                continue
            parts.append(f"## {s.title}\n{s.text}")
        return "\n\n".join(parts)


@dataclass
class Chunk:
    chunk_id: str
    paper_id: str
    text: str
    section: str = ""
    section_canonical: str = "other"
    page_start: int = 1
    page_end: int = 1
    ordinal: int = 0
    strategy: str = "section"
    n_tokens: int = 0
    has_equation: bool = False
    has_table: bool = False
    # denormalised paper metadata (so a chunk is never "just text")
    title: str = ""
    authors: list[str] = field(default_factory=list)
    year: int | None = None
    doi: str | None = None
    source: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def embed_text(self, with_context: bool = True) -> str:
        """Text that is actually embedded: contextual header (title + section) + body."""
        if not with_context:
            return self.text
        head = self.title.strip()
        if self.section:
            head = f"{head} | {self.section}" if head else self.section
        return f"{head}\n{self.text}" if head else self.text

    @property
    def cite(self) -> str:
        pg = f"p. {self.page_start}" if self.page_start == self.page_end else f"pp. {self.page_start}-{self.page_end}"
        return f"{self.section or 'n/a'}, {pg}"
