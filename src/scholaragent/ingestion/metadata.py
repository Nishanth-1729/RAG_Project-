"""Paper-level metadata: title / authors / abstract / year / DOI / arXiv id (heuristics + optional overrides)."""
from __future__ import annotations

import re
from pathlib import Path

from .models import Block, ParsedDocument

DOI_RE = re.compile(r"\b(10\.\d{4,9}/[^\s\"<>]+[^\s\"<>.,;)])", re.I)
ARXIV_ID_RE = re.compile(r"(?:arxiv[:\s/]*)?(\d{4}\.\d{4,5})(?:v\d+)?|([a-z\-]+(?:\.[A-Z]{2})?/\d{7})", re.I)
YEAR_RE = re.compile(r"\b(19[89]\d|20[0-3]\d)\b")
_BAD_TITLES = re.compile(r"(untitled|microsoft word|\.tex|\.doc|\.pdf|arxiv|preprint|manuscript|draft)", re.I)
_AFFIL_WORDS = re.compile(r"(universit|institute|department|laborator|college|school of|@|corresponding|google|microsoft|deepmind|nvidia|meta ai|openai|research|\bgmbh\b|\binc\b|\bltd\b|centre|center)", re.I)


def extract_title(front: list[Block], pdf_meta: dict) -> tuple[str, list[Block]]:
    """Title = the largest-font run of blocks on page 1 (top part); returns (title, used_blocks)."""
    p1 = [b for b in front if b.page == 1 and len(b.text) > 3]
    if p1:
        top = [b for b in p1 if b.bbox[1] < 0.45 * (max(x.bbox[3] for x in p1) or 800)] or p1
        max_size = max(b.font_size for b in top)
        cands = [b for b in top if b.font_size >= max_size - 0.6 and len(b.text) < 300]
        cands.sort(key=lambda b: b.bbox[1])
        # join vertically adjacent title lines
        title_blocks: list[Block] = []
        for b in cands:
            if not title_blocks or b.bbox[1] - title_blocks[-1].bbox[3] < max_size * 1.6:
                title_blocks.append(b)
        title = " ".join(b.text for b in title_blocks).strip()
        title = re.sub(r"\s+", " ", title)
        if len(title) >= 8:
            return title, title_blocks
    t = (pdf_meta or {}).get("title") or ""
    if t and not _BAD_TITLES.search(t) and len(t) > 8:
        return t.strip(), []
    return "", []


def extract_authors(front: list[Block], title_blocks: list[Block]) -> list[str]:
    if not title_blocks:
        return []
    ty = max(b.bbox[3] for b in title_blocks)
    cands = [b for b in front if b.page == 1 and b.bbox[1] >= ty - 2 and b.bbox[1] < ty + 160
             and b not in title_blocks and len(b.text) < 400]
    cands.sort(key=lambda b: (b.bbox[1], b.bbox[0]))
    names: list[str] = []
    for b in cands:
        t = b.text
        if re.match(r"^\s*abstract", t, re.I):
            break
        if _AFFIL_WORDS.search(t) and not re.search(r"\band\b|,", t):
            continue
        parts = re.split(r",|\band\b|\n|;|∗|\*|†|‡|§", t)
        for p in parts:
            p = re.sub(r"[\d†‡\*¹²³]+", "", p).strip(" ,.;")
            if (re.match(r"^[A-Z][\w'’.\-]+(?:\s+[A-Z][\w'’.\-]*){0,4}$", p) and 3 < len(p) < 40
                    and not _AFFIL_WORDS.search(p) and len(p.split()) >= 2):
                names.append(p)
    seen, out = set(), []
    for n in names:
        if n.lower() not in seen:
            seen.add(n.lower())
            out.append(n)
    return out[:25]


def find_ids(text: str) -> tuple[str | None, str | None]:
    doi = None
    m = DOI_RE.search(text)
    if m:
        doi = m.group(1).rstrip(".")
    arx = None
    m2 = re.search(r"arxiv[:\s/]*((?:\d{4}\.\d{4,5})|(?:[a-z\-]+(?:\.[A-Z]{2})?/\d{7}))", text, re.I)
    if m2:
        arx = m2.group(1)
    return doi, arx


def guess_year(text_head: str, filename: str, arxiv_id: str | None, pdf_meta: dict) -> int | None:
    if arxiv_id and re.match(r"^\d{4}\.", arxiv_id):
        yy = int(arxiv_id[:2])
        return 2000 + yy
    m = re.match(r"^(\d{2})(\d{2})\.\d{4,5}", Path(filename).stem)
    if m:
        return 2000 + int(m.group(1))
    cd = (pdf_meta or {}).get("creationDate", "")
    cands = [int(y) for y in YEAR_RE.findall(text_head)]
    cands = [y for y in cands if 1990 <= y <= 2030]
    if cands:
        return max(set(cands), key=cands.count)
    m = re.search(r"D:(\d{4})", cd)
    return int(m.group(1)) if m else None


def fill_metadata(doc: ParsedDocument, front: list[Block], pdf_meta: dict,
                  head_text: str, overrides: dict | None = None) -> ParsedDocument:
    title, tblocks = extract_title(front, pdf_meta)
    doc.title = title or doc.title or Path(doc.path).stem.replace("_", " ")
    doc.authors = extract_authors(front, tblocks) or doc.authors
    if not doc.authors and pdf_meta.get("author"):
        doc.authors = [a.strip() for a in re.split(r",|;|\band\b", pdf_meta["author"]) if a.strip()][:25]
    doi, arx = find_ids(head_text)
    doc.doi = doc.doi or doi
    doc.arxiv_id = doc.arxiv_id or arx
    m = re.match(r"^(\d{4}\.\d{4,5})", Path(doc.path).stem)
    if m and not doc.arxiv_id:
        doc.arxiv_id = m.group(1)
    doc.year = doc.year or guess_year(head_text[:3000], Path(doc.path).name, doc.arxiv_id, pdf_meta)
    for k, v in (overrides or {}).items():     # metadata from the search layer wins
        if v not in (None, "", []) and hasattr(doc, k):
            setattr(doc, k, v)
    return doc

