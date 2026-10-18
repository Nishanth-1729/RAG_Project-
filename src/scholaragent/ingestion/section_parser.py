"""Section detection: list[Block] -> list[Section] with canonical scientific-paper labels.

Heading detection combines (a) numbering patterns ("3.2 Foo", "IV. FOO", "A.1 Foo"),
(b) well-known section names, and (c) typography (bold / larger than body text).
"""
from __future__ import annotations

import re
from collections import Counter

from .models import Block, Paragraph, Section

_NUM_HEADING = re.compile(r"^(?P<num>\d{1,2}(?:\.\d{1,2}){0,3}[.)]?|[IVX]{1,5}[.)]|[A-H](?:\.\d{1,2})+[.)]?|[A-H][.)])\s+(?P<title>[A-Z][^\n]{1,110})$")
_ROMAN_ONLY = re.compile(r"^[IVX]{1,5}$")

# canonical name -> regex over the lowercase, number-stripped title
_CANON_RULES: list[tuple[str, re.Pattern]] = [
    ("abstract", re.compile(r"^abstract\b")),
    ("references", re.compile(r"^(references?|bibliography|literature cited|works cited)\b")),
    ("appendix", re.compile(r"^(appendix|appendices|supplement|supplementary)")),
    ("acknowledgments", re.compile(r"^(acknowledg|funding|author contributions|conflicts? of interest|declarations?|ethic|data availability|code availability|reproducibility)")),
    ("introduction", re.compile(r"^(introduction|overview and motivation|motivation)\b")),
    ("related_work", re.compile(r"(related work|prior work|previous work|literature review|state of the art|existing (methods|approaches))")),
    ("limitations", re.compile(r"(limitation|threats to validity|shortcoming|failure (cases|modes)|negative results)")),
    ("conclusion", re.compile(r"^(conclusions?|concluding|summary|conclusions? (and|&) (future|outlook|discussion)|discussions? (and|&) conclusions?|future (work|directions?)|outlook)\b")),
    ("discussion", re.compile(r"^(discussions?|analysis and discussion)\b")),
    ("experiments", re.compile(r"(experiment|evaluation|empirical|benchmark|numerical (results|experiments|examples|tests)|implementation details|experimental (setup|setting)|setup\b|datasets?\b|test problems|case stud)")),
    ("results", re.compile(r"(results?\b|ablation|performance|findings|comparison with|comparisons?\b|analysis\b)")),
    ("background", re.compile(r"(background|preliminar|notation|problem (statement|setting|formulation|definition)|definitions?\b|theoretical foundations?)")),
    ("method", re.compile(r"(method|approach|model\b|models\b|algorithm|framework|architecture|proposed|formulation|technique|network|solver|scheme|construction|design|training|theor(y|etical)|analysis of)")),
]
_KNOWN_TOP = re.compile(r"^(abstract|introduction|related work|background|preliminaries|methods?|methodology|approach|experiments?|evaluation|results|discussion|limitations|conclusions?|references|bibliography|appendix|acknowledg\w+)\b", re.I)


def canonicalize(title: str) -> str:
    t = re.sub(r"^(?:\d{1,2}(?:\.\d{1,2}){0,3}[.)]?|[IVX]{1,5}[.)]|[A-H](?:\.\d{1,2})+[.)]?|[A-H][.)])\s+", "", title.strip()).lower()
    t = re.sub(r"[^a-z0-9 &/-]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    for name, rx in _CANON_RULES:
        if rx.search(t):
            return name
    return "other"


def _split_number(heading: str) -> tuple[str, str]:
    m = _NUM_HEADING.match(heading.strip())
    if m:
        return m.group("num").rstrip(".)"), m.group("title").strip()
    return "", heading.strip()


def _level(number: str) -> int:
    if not number:
        return 1
    if _ROMAN_ONLY.match(number):
        return 1
    return number.count(".") + 1


def is_heading(b: Block, body: float, prev_kind: str | None) -> bool:
    t = b.text.strip()
    if b.kind in ("equation", "caption", "table", "table_dup"):
        return False
    if len(t) > 120 or len(t) < 3 or "\n" in t:
        return False
    if t.endswith((".", ",", ";")) and not _NUM_HEADING.match(t) and not _KNOWN_TOP.match(t):
        return False
    words = t.split()
    if len(words) > 14:
        return False
    big = b.font_size >= body * 1.07
    typo = b.bold or big
    m = _NUM_HEADING.match(t)
    if m:
        num = m.group("num").rstrip(".)")
        title = m.group("title")
        # reject things like "2 Experiments were" prose or numbers that look like list/eq items
        if len(words) > 10 and not typo:
            return False
        if _ROMAN_ONLY.match(num) and not (typo or t.isupper() or _KNOWN_TOP.match(title)):
            return False
        if re.match(r"^\d{1,2}$", num) and not (typo or _KNOWN_TOP.match(title)):
            # plain numbered text like "3 samples were..." must look like a heading typographically
            return False
        return True
    if _KNOWN_TOP.match(t) and len(words) <= 6 and (typo or t.isupper() or t.istitle()):
        return True
    if typo and big and len(words) <= 8 and t[0].isupper() and not t.endswith((".", ",")):
        return True
    if t.isupper() and 1 <= len(words) <= 6 and len(t) > 4 and b.bold:
        return True
    return False


_END_SENT = re.compile(r"[.!?:][\"')\]]?$")


class SectionParser:
    def __init__(self, min_section_chars: int = 0):
        self.min_section_chars = min_section_chars

    def parse(self, blocks: list[Block], body_font: float) -> tuple[list[Section], dict]:
        """Returns (sections, front) where front = text blocks before the first section (title/authors/abstract)."""
        sections: list[Section] = []
        front: list[Block] = []
        cur: Section | None = None
        ref_mode = False
        # sort by page but keep intra-page order; table blocks appended last per page already
        prev_kind = None
        for b in blocks:
            if b.kind == "table_dup":
                continue
            if b.kind == "paragraph" or b.kind == "caption":
                abs_m = re.match(r"^\s*(abstract)\s*[:.\-—–]*\s*(.*)$", b.text, re.I | re.S)
                if abs_m and cur is None and len(b.text) > 20 and (abs_m.group(2) or b.bold):
                    # inline "Abstract— text" block
                    cur = Section(title="Abstract", canonical="abstract", page_start=b.page, page_end=b.page)
                    sections.append(cur)
                    body_txt = abs_m.group(2).strip()
                    if body_txt:
                        cur.paragraphs.append(Paragraph(body_txt, b.page, b.page))
                    continue
            strong = bool(_NUM_HEADING.match(b.text.strip()) or _KNOWN_TOP.match(b.text.strip()))
            # before the first real section, typography alone must not open a section (that would swallow the title)
            if (cur is not None or strong) and is_heading(b, body_font, prev_kind) and not (ref_mode and not _KNOWN_TOP.match(b.text)):
                num, title = _split_number(b.text)
                canon = canonicalize(b.text)
                if cur is not None and canon == "abstract" and cur.canonical == "abstract":
                    continue
                cur = Section(title=(f"{num} {title}" if num else title), number=num, level=_level(num),
                              canonical=canon, page_start=b.page, page_end=b.page)
                sections.append(cur)
                ref_mode = canon == "references"
                prev_kind = "heading"
                continue
            prev_kind = b.kind
            if cur is None:
                front.append(b)
                continue
            kind = b.kind if b.kind in ("equation", "table", "caption") else "paragraph"
            if ref_mode:
                kind = "reference"
            cur.page_end = max(cur.page_end, b.page)
            cur.paragraphs.append(Paragraph(b.text, b.page, b.page, kind))
        sections = [self._merge_paragraphs(s) for s in sections]
        self._propagate_canonical(sections)
        sections = [s for s in sections if s.paragraphs or s.canonical in ("abstract",)]
        return sections, {"front": front}

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _merge_paragraphs(s: Section) -> Section:
        """Join prose blocks that were broken by columns/pages (no sentence end + lowercase continuation)."""
        merged: list[Paragraph] = []
        for p in s.paragraphs:
            if p.kind == "equation":
                if len(p.text.strip()) < 6:
                    continue                                   # stray sub/superscript fragment ('i=1', '-1', 'k')
                if merged and merged[-1].kind == "equation" and merged[-1].page_end == p.page_start:
                    prev = merged[-1]                          # consecutive display-math fragments -> one equation
                    prev.text = prev.text.rstrip() + " " + p.text.strip()
                    prev.page_end = max(prev.page_end, p.page_end)
                    continue
            if (merged and p.kind == "paragraph" and merged[-1].kind == "paragraph"
                    and not _END_SENT.search(merged[-1].text) and p.text[:1].islower()):
                prev = merged[-1]
                prev.text = prev.text.rstrip() + " " + p.text
                prev.page_end = max(prev.page_end, p.page_end)
            else:
                merged.append(Paragraph(p.text, p.page_start, p.page_end, p.kind))
        s.paragraphs = merged
        return s

    @staticmethod
    def _propagate_canonical(sections: list[Section]) -> None:
        """Sub-sections inherit the parent's canonical label when they have none; unknown top-level
        sections between the introduction and the experiments are treated as 'method'."""
        parent_canon = "other"
        for s in sections:
            if s.level == 1:
                parent_canon = s.canonical
            elif (s.canonical == "other" or (s.canonical == "method" and parent_canon not in ("other", "method"))
                  or parent_canon in ("introduction", "related_work", "background", "abstract")):
                s.canonical = parent_canon
        seen_intro = seen_exp = False
        for s in sections:
            if s.level != 1:
                continue
            if s.canonical == "introduction":
                seen_intro = True
            elif s.canonical in ("experiments", "results", "discussion", "conclusion", "limitations"):
                seen_exp = True
            elif s.canonical == "other" and seen_intro and not seen_exp:
                s.canonical = "method"
        # re-propagate to children of re-labelled parents
        parent_canon = "other"
        for s in sections:
            if s.level == 1:
                parent_canon = s.canonical
            elif s.canonical == "other":
                s.canonical = parent_canon
