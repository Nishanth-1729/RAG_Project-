"""Chunking strategies. Every chunk keeps paper, section, page range and paper-level metadata.

Strategies (compared experimentally in evaluation/run_experiments.py):
  fixed     - sliding window of N tokens / overlap, ignoring structure (baseline)
  section   - structure aware: paragraphs are packed inside a section, never across sections
  semantic  - sentences grouped by embedding-similarity breakpoints inside a section
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from ..utils.text import approx_tokens, sentences
from .models import Chunk, Paragraph, ParsedDocument, Section

SKIP_CANON = {"references", "acknowledgments"}


@dataclass
class ChunkerConfig:
    strategy: str = "section"      # fixed | section | semantic
    max_tokens: int = 384
    overlap: int = 48
    min_tokens: int = 40
    include_appendix: bool = True
    semantic_percentile: float = 25.0   # lower percentile of adjacent-sentence similarity => breakpoint


def _mk(doc: ParsedDocument, text: str, sec: Section | None, ps: int, pe: int, ordinal: int,
        cfg: ChunkerConfig, eq: bool, tb: bool) -> Chunk:
    canon = sec.canonical if sec else "other"
    cid = f"{doc.paper_id}_{canon}_{ordinal:03d}"
    return Chunk(
        chunk_id=cid, paper_id=doc.paper_id, text=text.strip(), section=sec.title if sec else "",
        section_canonical=canon, page_start=ps, page_end=pe, ordinal=ordinal, strategy=cfg.strategy,
        n_tokens=approx_tokens(text), has_equation=eq, has_table=tb, title=doc.title, authors=list(doc.authors),
        year=doc.year, doi=doc.doi, source=doc.source,
    )


class Chunker:
    def __init__(self, cfg: ChunkerConfig | None = None,
                 embed_fn: Callable[[Sequence[str]], np.ndarray] | None = None):
        self.cfg = cfg or ChunkerConfig()
        self.embed_fn = embed_fn

    def chunk(self, doc: ParsedDocument) -> list[Chunk]:
        secs = [s for s in doc.sections if s.canonical not in SKIP_CANON
                and (self.cfg.include_appendix or s.canonical != "appendix")]
        strat = self.cfg.strategy
        if strat == "fixed":
            chunks = self._fixed(doc, secs)
        elif strat == "semantic":
            chunks = self._semantic(doc, secs)
        else:
            chunks = self._section(doc, secs)
        # contiguous ordinals + unique ids
        for i, c in enumerate(chunks):
            c.ordinal = i
            c.chunk_id = f"{doc.paper_id}_{c.section_canonical}_{i:03d}"
        return chunks

    # ------------------------------------------------------------------ fixed
    def _fixed(self, doc: ParsedDocument, secs: list[Section]) -> list[Chunk]:
        words: list[tuple[str, int, Section, str]] = []
        for s in secs:
            for p in s.paragraphs:
                for w in p.text.split():
                    words.append((w, p.page_start, s, p.kind))
        # window size in words, calibrated so that a window holds ~max_tokens tokens for this document
        ratio = approx_tokens(" ".join(w[0] for w in words[:4000])) / max(1, min(len(words), 4000))
        size = max(16, int(self.cfg.max_tokens / ratio))
        step = max(8, size - int(self.cfg.overlap / ratio))
        out: list[Chunk] = []
        for start in range(0, max(1, len(words)), step):
            win = words[start:start + size]
            if not win:
                break
            text = " ".join(w for w, *_ in win)
            if approx_tokens(text) < self.cfg.min_tokens and out:
                break
            sec = win[0][2]
            pages = [p for _, p, *_ in win]
            out.append(_mk(doc, text, sec, min(pages), max(pages), len(out), self.cfg,
                           any(k == "equation" for *_, k in win), any(k == "table" for *_, k in win)))
            if start + size >= len(words):
                break
        return out

    # ---------------------------------------------------------------- section
    def _section(self, doc: ParsedDocument, secs: list[Section]) -> list[Chunk]:
        out: list[Chunk] = []
        for s in secs:
            units = self._units(s)
            out.extend(self._pack(doc, s, units))
        return self._merge_tiny(out)

    def _units(self, s: Section) -> list[tuple[str, int, int, str]]:
        """Break a section into (text, page_start, page_end, kind) units no larger than max_tokens."""
        units: list[tuple[str, int, int, str]] = []
        pending_eq: list[Paragraph] = []
        for p in s.paragraphs:
            if p.kind == "equation":
                pending_eq.append(p)
                continue
            if pending_eq and p.kind == "paragraph":
                # equations stay glued to the following sentence so "where x is ..." keeps its context
                text = " ".join(e.text for e in pending_eq) + " " + p.text
                kind, ps = "equation", pending_eq[0].page_start
                pending_eq = []
                units.extend(self._split_big(text, ps, p.page_end, kind))
                continue
            units.extend(self._split_big(p.text, p.page_start, p.page_end, p.kind))
        for e in pending_eq:
            units.append((e.text, e.page_start, e.page_end, "equation"))
        return units

    def _split_big(self, text: str, ps: int, pe: int, kind: str) -> list[tuple[str, int, int, str]]:
        if approx_tokens(text) <= self.cfg.max_tokens:
            return [(text, ps, pe, kind)]
        sents = sentences(text)
        out, buf = [], []
        for st in sents:
            if approx_tokens(" ".join(buf + [st])) > self.cfg.max_tokens and buf:
                out.append((" ".join(buf), ps, pe, kind))
                buf = []
            if approx_tokens(st) > self.cfg.max_tokens:       # pathological single sentence
                words = st.split()
                step = int(self.cfg.max_tokens / 1.15)
                for i in range(0, len(words), step):
                    out.append((" ".join(words[i:i + step]), ps, pe, kind))
                continue
            buf.append(st)
        if buf:
            out.append((" ".join(buf), ps, pe, kind))
        return out

    def _pack(self, doc: ParsedDocument, s: Section, units: list[tuple[str, int, int, str]]) -> list[Chunk]:
        out: list[Chunk] = []
        buf: list[tuple[str, int, int, str]] = []

        def flush():
            if not buf:
                return
            text = "\n\n".join(u[0] for u in buf)
            out.append(_mk(doc, text, s, min(u[1] for u in buf), max(u[2] for u in buf), len(out), self.cfg,
                           any(u[3] == "equation" for u in buf), any(u[3] == "table" for u in buf)))

        def tail_overlap() -> list[tuple[str, int, int, str]]:
            if self.cfg.overlap <= 0 or not buf:
                return []
            last = buf[-1]
            sents = sentences(last[0])
            keep, tok = [], 0
            for st in reversed(sents):
                tok += approx_tokens(st)
                if tok > self.cfg.overlap:
                    break
                keep.insert(0, st)
            return [(" ".join(keep), last[1], last[2], "paragraph")] if keep and len(keep) < len(sents) else []

        for u in units:
            cur_tokens = approx_tokens("\n\n".join(x[0] for x in buf))
            if buf and cur_tokens + approx_tokens(u[0]) > self.cfg.max_tokens:
                ov = tail_overlap()
                flush()
                buf = ov
            buf.append(u)
        flush()
        return out

    def _merge_tiny(self, chunks: list[Chunk]) -> list[Chunk]:
        """Merge chunks below min_tokens into a neighbour of the same section."""
        out: list[Chunk] = []
        for c in chunks:
            if (out and c.n_tokens < self.cfg.min_tokens and out[-1].section == c.section
                    and out[-1].n_tokens + c.n_tokens <= int(self.cfg.max_tokens * 1.25)):
                p = out[-1]
                p.text += "\n\n" + c.text
                p.page_end = max(p.page_end, c.page_end)
                p.n_tokens = approx_tokens(p.text)
                p.has_equation |= c.has_equation
                p.has_table |= c.has_table
            else:
                out.append(c)
        return [c for c in out if c.n_tokens >= 8]

    # --------------------------------------------------------------- semantic
    def _semantic(self, doc: ParsedDocument, secs: list[Section]) -> list[Chunk]:
        if self.embed_fn is None:
            raise ValueError("semantic chunking needs embed_fn")
        out: list[Chunk] = []
        for s in secs:
            sents: list[tuple[str, int, str]] = []
            for p in s.paragraphs:
                for st in sentences(p.text):
                    sents.append((st, p.page_start, p.kind))
            if not sents:
                continue
            if len(sents) < 4:
                text = " ".join(x[0] for x in sents)
                out.append(_mk(doc, text, s, sents[0][1], s.paragraphs[-1].page_end, len(out), self.cfg,
                               any(x[2] == "equation" for x in sents), any(x[2] == "table" for x in sents)))
                continue
            emb = np.asarray(self.embed_fn([x[0] for x in sents]), dtype=np.float32)
            emb /= (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
            sims = (emb[:-1] * emb[1:]).sum(1)
            thr = np.percentile(sims, self.cfg.semantic_percentile)
            groups, cur = [], [0]
            for i, sim in enumerate(sims):
                tokens = approx_tokens(" ".join(sents[j][0] for j in cur))
                if (sim < thr and tokens >= self.cfg.min_tokens) or tokens >= self.cfg.max_tokens:
                    groups.append(cur)
                    cur = []
                cur.append(i + 1)
            groups.append(cur)
            for g in groups:
                text = " ".join(sents[j][0] for j in g)
                out.append(_mk(doc, text, s, min(sents[j][1] for j in g), max(sents[j][1] for j in g), len(out),
                               self.cfg, any(sents[j][2] == "equation" for j in g), any(sents[j][2] == "table" for j in g)))
        return self._merge_tiny(out)
