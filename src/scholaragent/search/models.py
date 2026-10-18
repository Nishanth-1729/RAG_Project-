from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

from ..utils.text import normalize_title, slugify, stable_hash


@dataclass
class PaperRecord:
    title: str
    abstract: str = ""
    authors: list[str] = field(default_factory=list)
    year: int | None = None
    venue: str | None = None
    doi: str | None = None
    arxiv_id: str | None = None
    s2_id: str | None = None
    openalex_id: str | None = None
    citation_count: int | None = None
    pdf_url: str | None = None
    url: str | None = None
    sources: list[str] = field(default_factory=list)
    references: list[str] = field(default_factory=list)   # s2 paper ids / titles of cited papers
    score: float = 0.0
    score_parts: dict[str, float] = field(default_factory=dict)
    queries: list[str] = field(default_factory=list)

    @property
    def paper_id(self) -> str:
        if self.arxiv_id:
            return "arxiv_" + slugify(re.sub(r"v\d+$", "", self.arxiv_id).replace("/", "_"), 30)
        if self.doi:
            return "doi_" + stable_hash(self.doi.lower(), n=10)
        return "t_" + stable_hash(normalize_title(self.title), n=10)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["paper_id"] = self.paper_id
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PaperRecord":
        d = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**d)
