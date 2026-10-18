"""Page-aware PDF parsing with PyMuPDF.

Step 1 of ingestion: PDF -> list[Block] (text blocks that keep their page number, font size,
bold flag and a coarse kind: heading / paragraph / equation / table / caption).
Section detection lives in section_parser.py; metadata extraction in metadata.py.
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter
from pathlib import Path

import pymupdf as fitz  # PyMuPDF

from ..utils.logging import get_logger
from .models import Block

log = get_logger("pdf")

_MATH_CHARS = set("=+−-*/^_<>≤≥≈≠∈∉⊂⊆∪∩∑∏∫∂∇±×·√∞∀∃→←↔⇒⇔λμσθαβγδεζηπρτφχψωΩΣΔΘΛΠΦΨ‖|{}[]()")
_STRONG_MATH = set("=≤≥≈≠∈∑∏∫∂∇±×√∞∀∃→⇒⇔‖λμσθαβγδεζηπρτφχψω")
_EQ_NUM_RE = re.compile(r"\(\s*\d+(\.\d+)?\s*\)\s*$")
_CAPTION_RE = re.compile(r"^(figure|fig\.|table|algorithm)\s*\d+[.:]?", re.I)
_RUNIN_HEAD_RE = re.compile(r"^(?:(?:\d{1,2}(?:\.\d{1,2}){0,3}|[IVX]{1,5}|[A-H](?:\.\d{1,2})*)[.)]?\s+[A-Z][^\n]{1,100}|abstract|keywords?|key words|introduction|conclusions?)[.:]?$", re.I)
_PAGE_NUM_RE = re.compile(r"^\s*(page\s*)?\d{1,4}(\s*(of|/)\s*\d{1,4})?\s*$", re.I)


_PUA_RE = re.compile("[\ue000-\uf8ff\ufffd\u00ad\u200b]")
_NFKC_RANGES = re.compile("[\U0001d400-\U0001d7ff\ufb00-\ufb06\u2100-\u214f]")   # math alphanumerics, ligatures, letterlike symbols


def clean_text(t: str) -> str:
    """Normalise PDF extraction artefacts: math-italic letters -> plain letters, ligatures -> letters,
    unmappable glyphs (U+FFFD), private-use bracket pieces and soft hyphens removed."""
    if not t.isascii():
        t = _NFKC_RANGES.sub(lambda m: unicodedata.normalize("NFKC", m.group(0)), t)
        t = _PUA_RE.sub("", t)
    return t


def _is_bold(span: dict) -> bool:
    return bool(span["flags"] & 16) or "bold" in span["font"].lower() or "black" in span["font"].lower()


def _dehyphenate_join(lines: list[str]) -> str:
    out = ""
    for ln in lines:
        ln = ln.rstrip()
        if not ln:
            continue
        if out.endswith("-") and len(out) > 1 and out[-2].isalpha() and ln[:1].islower():
            out = out[:-1] + ln          # "approxi-" + "mation"
        elif out:
            out += " " + ln
        else:
            out = ln
    return re.sub(r"[ \t]+", " ", out).strip()


def _math_density(text: str) -> tuple[float, float]:
    """Returns (share of math-ish characters, share of alphabetic words with >=4 letters)."""
    s = re.sub(r"\s+", "", text)
    if not s:
        return 0.0, 0.0
    m = sum(1 for c in s if c in _MATH_CHARS or c in _STRONG_MATH)
    words = re.findall(r"[A-Za-z]{4,}", text)
    return m / len(s), len(words) / max(1, len(text.split()))


def classify_equation(text: str) -> bool:
    if len(text) > 400:
        return False
    dens, word_share = _math_density(text)
    strong = sum(1 for c in text if c in _STRONG_MATH)
    has_num = bool(_EQ_NUM_RE.search(text))
    if strong >= 2 and word_share < 0.35:
        return True
    if has_num and dens > 0.12 and word_share < 0.5:
        return True
    return dens > 0.30 and word_share < 0.25


class PDFParser:
    """Extract page-aware blocks from a PDF."""

    def __init__(self, detect_tables: bool = True, strip_headers: bool = True):
        self.detect_tables = detect_tables
        self.strip_headers = strip_headers

    # ---------------------------------------------------------------- public
    def parse(self, path: str | Path) -> tuple[list[Block], dict]:
        """Returns (blocks, info). info: n_pages, body_font_size, needs_ocr, warnings, pdf_meta."""
        path = Path(path)
        warnings: list[str] = []
        doc = fitz.open(path)
        try:
            n_pages = len(doc)
            raw_pages = [self._page_blocks(doc[i], i + 1) for i in range(n_pages)]
            pdf_meta = dict(doc.metadata or {})
            tables = self._tables(doc) if self.detect_tables else {}
        finally:
            doc.close()

        total_chars = sum(len(b.text) for pg in raw_pages for b in pg)
        needs_ocr = n_pages > 0 and (total_chars / n_pages) < 300
        if needs_ocr:
            warnings.append("very little extractable text - probably a scanned PDF (OCR needed)")

        body = self._body_font_size(raw_pages)
        blocks: list[Block] = []
        if self.strip_headers:
            raw_pages = self._strip_headers_footers(raw_pages)
        for pg_no, pg in enumerate(raw_pages, start=1):
            tbs = tables.get(pg_no, [])
            for b in pg:
                b.kind = self._classify(b, body, tbs)
                blocks.append(b)
            for tb in tbs:  # tables extracted separately keep their own block
                blocks.append(Block(page=pg_no, text=tb["text"], kind="table", bbox=tb["bbox"]))
        return blocks, {"n_pages": n_pages, "body_font_size": body, "needs_ocr": needs_ocr,
                        "warnings": warnings, "pdf_meta": pdf_meta}

    # --------------------------------------------------------------- internals
    def _page_blocks(self, page: "fitz.Page", page_no: int) -> list[Block]:
        out: list[Block] = []
        d = page.get_text("dict", flags=fitz.TEXT_DEHYPHENATE)
        for b in d["blocks"]:
            if b.get("type") != 0:
                continue
            # per-line typography, then split the block into runs of homogeneous font (a heading line that
            # PyMuPDF glued to the paragraph below it becomes its own block)
            lines = []
            for ln in b["lines"]:
                txt = "".join(s["text"] for s in ln["spans"])
                if not txt.strip():
                    continue
                allspans = [{**s_, "text": clean_text(s_["text"])} for s_ in ln["spans"]]
                # run-in heading: bold prefix ("2 Method. We ...") followed by regular-weight body text on the same line
                cut = next((i for i, s_ in enumerate(allspans) if s_["text"].strip() and not _is_bold(s_)), len(allspans))
                prefix = "".join(s_["text"] for s_ in allspans[:cut]).strip()
                if 0 < cut < len(allspans) and prefix and len(prefix) <= 110 and _RUNIN_HEAD_RE.match(prefix):
                    parts = [(allspans[:cut], True), (allspans[cut:], False)]
                else:
                    parts = [(allspans, None)]
                for sp, forced_bold in parts:
                    txt = "".join(s_["text"] for s_ in sp)
                    if not txt.strip():
                        continue
                    sizes, bold, tot = Counter(), 0, 0
                    for s_ in sp:
                        n = len(s_["text"].strip())
                        if n:
                            sizes[round(s_["size"], 1)] += n
                            tot += n
                            bold += n if _is_bold(s_) else 0
                    lines.append({"text": txt, "size": max(sizes, key=sizes.get) if sizes else 0.0,
                                  "bold": tot > 0 and bold / tot > 0.6, "bbox": ln["bbox"], "n": tot,
                                  "hard_break": forced_bold is not None})
            runs: list[list[dict]] = []
            for ln in lines:
                if runs and abs(runs[-1][-1]["size"] - ln["size"]) <= 0.8 and runs[-1][-1]["bold"] == ln["bold"]:
                    runs[-1].append(ln)
                elif (runs and ln["n"] < 12 and not ln["bold"] and abs(runs[-1][-1]["size"] - ln["size"]) <= 0.8):
                    runs[-1].append(ln)
                else:
                    runs.append([ln])
            for run in runs:
                text = _dehyphenate_join([l["text"] for l in run])
                if not text:
                    continue
                x0 = min(l["bbox"][0] for l in run); y0 = min(l["bbox"][1] for l in run)
                x1 = max(l["bbox"][2] for l in run); y1 = max(l["bbox"][3] for l in run)
                wsize = Counter()
                for l in run:
                    wsize[l["size"]] += l["n"]
                out.append(Block(page=page_no, text=text, font_size=max(wsize, key=wsize.get),
                                 bold=sum(l["n"] for l in run if l["bold"]) / max(1, sum(l["n"] for l in run)) > 0.6,
                                 bbox=(x0, y0, x1, y1)))
        out = self._join_number_labels(out)
        # PyMuPDF returns stream order, which follows reading order for LaTeX output;
        # for two-column pages whose stream order is scrambled, re-order by column.
        return self._reorder_columns(out, page.rect.width)

    @staticmethod
    def _join_number_labels(blocks: list[Block]) -> list[Block]:
        """Some templates put the section number ('2.1') and its title in separate text blocks on the same line."""
        num = re.compile(r"^(?:\d{1,2}(?:\.\d{1,2}){0,3}\.?|[IVX]{1,5}\.|[A-H](?:\.\d{1,2})+)$")
        out: list[Block] = []
        i = 0
        while i < len(blocks):
            b = blocks[i]
            if num.match(b.text.strip()) and i + 1 < len(blocks):
                nx = blocks[i + 1]
                h = max(1.0, b.bbox[3] - b.bbox[1])
                overlap = min(b.bbox[3], nx.bbox[3]) - max(b.bbox[1], nx.bbox[1])
                if overlap > 0.5 * h and -2 <= nx.bbox[0] - b.bbox[2] < 45 and len(nx.text) < 140:
                    out.append(Block(page=nx.page, text=f"{b.text.strip()} {nx.text}", font_size=max(nx.font_size, b.font_size),
                                     bold=nx.bold or b.bold, bbox=(min(b.bbox[0], nx.bbox[0]), min(b.bbox[1], nx.bbox[1]),
                                                                  max(b.bbox[2], nx.bbox[2]), max(b.bbox[3], nx.bbox[3]))))
                    i += 2
                    continue
            out.append(b)
            i += 1
        return out

    @staticmethod
    def _reorder_columns(blocks: list[Block], width: float) -> list[Block]:
        if len(blocks) < 6:
            return blocks
        mid = width / 2
        narrow = [b for b in blocks if (b.bbox[2] - b.bbox[0]) < 0.6 * width]
        left = [b for b in narrow if b.bbox[2] <= mid + 8]
        right = [b for b in narrow if b.bbox[0] >= mid - 8]
        if len(left) < 3 or len(right) < 3:
            return blocks
        # Deterministic ordering: full-width blocks split the page into horizontal bands;
        # inside a band the left column is read before the right column.
        wide = sorted([b for b in blocks if b not in left and b not in right], key=lambda b: b.bbox[1])
        bands: list[tuple[float, float]] = []
        prev = -1.0
        for w in wide:
            bands.append((prev, w.bbox[1]))
            prev = w.bbox[3]
        bands.append((prev, 1e9))
        ordered: list[Block] = []
        used: set[int] = set()
        for i, (y0, y1) in enumerate(bands):
            col_l = sorted([b for b in left if y0 <= b.bbox[1] < y1], key=lambda b: b.bbox[1])
            col_r = sorted([b for b in right if y0 <= b.bbox[1] < y1], key=lambda b: b.bbox[1])
            ordered += col_l + col_r
            used.update(id(b) for b in col_l + col_r)
            if i < len(wide):
                ordered.append(wide[i])
                used.add(id(wide[i]))
        ordered += [b for b in blocks if id(b) not in used]
        return ordered

    @staticmethod
    def _body_font_size(pages: list[list[Block]]) -> float:
        c: Counter = Counter()
        for pg in pages:
            for b in pg:
                if len(b.text) > 80:
                    c[round(b.font_size, 1)] += len(b.text)
        return max(c, key=c.get) if c else 10.0

    @staticmethod
    def _strip_headers_footers(pages: list[list[Block]]) -> list[list[Block]]:
        n = len(pages)
        if n < 3:
            return [[b for b in pg if not _PAGE_NUM_RE.match(b.text)] for pg in pages]
        norm = lambda t: re.sub(r"\d+", "#", re.sub(r"\s+", " ", t.strip().lower()))  # noqa: E731
        counts: Counter = Counter()
        for pg in pages:
            seen = set()
            for b in pg:
                if len(b.text) < 140:
                    seen.add(norm(b.text))
            counts.update(seen)
        repeated = {t for t, c in counts.items() if c >= max(3, int(0.4 * n)) and len(t) > 3}
        out = []
        for pg in pages:
            kept = []
            for b in pg:
                if _PAGE_NUM_RE.match(b.text):
                    continue
                if norm(b.text) in repeated and len(b.text) < 140:
                    continue
                kept.append(b)
            out.append(kept)
        return out

    def _tables(self, doc: "fitz.Document") -> dict[int, list[dict]]:
        res: dict[int, list[dict]] = {}
        for i, page in enumerate(doc):
            try:
                txt_hint = page.get_text("text")
                if not re.search(r"\bTable\s+\d", txt_hint):
                    continue
                found = page.find_tables()
                for t in found.tables:
                    rows = t.extract()
                    if not rows or len(rows) < 2 or len(rows[0]) < 2:
                        continue
                    md = "\n".join(" | ".join((c or "").replace("\n", " ").strip() for c in r) for r in rows)
                    if len(md) < 20:
                        continue
                    res.setdefault(i + 1, []).append({"text": md[:4000], "bbox": tuple(t.bbox)})
            except Exception as e:  # noqa: BLE001
                log.debug("table detection failed on page %s: %r", i + 1, e)
        return res

    def _classify(self, b: Block, body: float, tables: list[dict]) -> str:
        t = b.text
        for tb in tables:   # text blocks lying inside a detected table are dropped from prose
            x0, y0, x1, y1 = tb["bbox"]
            bx0, by0, bx1, by1 = b.bbox
            if bx0 >= x0 - 2 and by0 >= y0 - 2 and bx1 <= x1 + 2 and by1 <= y1 + 2:
                return "table_dup"
        if _CAPTION_RE.match(t):
            return "caption"
        if classify_equation(t):
            return "equation"
        return "paragraph"
