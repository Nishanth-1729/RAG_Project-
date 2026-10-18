"""Evidence objects, citation formatting and citation validation. Answers never lose their evidence."""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

from rapidfuzz import fuzz

from ..retrieval.hybrid import Hit
from ..utils.text import normalize_for_match

_CITE_RE = re.compile(r"\[(\d+(?:\s*[,;–-]\s*\d+)*)\]")


@dataclass
class Evidence:
    id: int                      # the [n] used in the answer text
    chunk_id: str
    paper_id: str
    title: str
    authors: list[str]
    year: int | None
    section: str
    page_start: int
    page_end: int
    text: str
    doi: str | None = None
    score: float = 0.0

    @property
    def short_ref(self) -> str:
        return paper_label(self.authors, self.year)

    @property
    def location(self) -> str:
        pg = f"p. {self.page_start}" if self.page_start == self.page_end else f"pp. {self.page_start}-{self.page_end}"
        sec = f"§{self.section}" if re.match(r"^\d", self.section or "") else (self.section or "n/a")
        return f"{sec}, {pg}"

    def label(self) -> str:
        return f"{self.short_ref}, {self.location}"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["label"] = self.label()
        return d


def paper_label(authors: list[str], year: int | None) -> str:
    if not authors:
        return f"Unknown ({year or 'n.d.'})"
    sur = authors[0].split()[-1]
    if len(authors) == 1:
        return f"{sur} ({year or 'n.d.'})"
    if len(authors) == 2:
        return f"{sur} & {authors[1].split()[-1]} ({year or 'n.d.'})"
    return f"{sur} et al. ({year or 'n.d.'})"


def evidence_from_hits(hits: list[Hit], start_id: int = 1) -> list[Evidence]:
    out = []
    for i, h in enumerate(hits, start_id):
        c = h.chunk
        out.append(Evidence(id=i, chunk_id=c.chunk_id, paper_id=c.paper_id, title=c.title, authors=c.authors, year=c.year,
                            section=c.section, page_start=c.page_start, page_end=c.page_end, text=c.text, doi=c.doi,
                            score=h.score))
    return out


def extract_citation_ids(text: str) -> list[int]:
    ids: list[int] = []
    for m in _CITE_RE.finditer(text):
        for part in re.split(r"[,;]", m.group(1)):
            part = part.strip()
            if re.match(r"^\d+\s*[–-]\s*\d+$", part):
                a, b = [int(x) for x in re.split(r"[–-]", part)]
                ids.extend(range(a, b + 1))
            elif part.isdigit():
                ids.append(int(part))
    return ids


def strip_invalid_citations(text: str, valid_ids: set[int]) -> tuple[str, list[int]]:
    """Remove citations that point to evidence that does not exist (never trust the model's numbering)."""
    removed: list[int] = []

    def fix(m: re.Match) -> str:
        keep = []
        for p in re.split(r"\s*[,;]\s*", m.group(1)):
            if p.isdigit() and int(p) in valid_ids:
                keep.append(p)
            elif p.isdigit():
                removed.append(int(p))
            else:
                keep.append(p)
        return f"[{', '.join(keep)}]" if keep else ""
    return re.sub(r"\s{2,}", " ", _CITE_RE.sub(fix, text)), removed


def quote_support(quote: str, source: str, threshold: float = 88.0) -> float:
    """0-100 fuzzy score of how well `quote` is contained in `source` (punctuation/hyphenation tolerant)."""
    q, s = normalize_for_match(quote), normalize_for_match(source)
    if not q or not s:
        return 0.0
    if q in s:
        return 100.0
    return float(fuzz.partial_ratio(q, s))


def render_references(evidence: list[Evidence], used: set[int] | None = None) -> str:
    lines = []
    for e in evidence:
        if used is not None and e.id not in used:
            continue
        lines.append(f"[{e.id}] {e.short_ref} — *{e.title}*, {e.location}")
    return "\n".join(lines)
