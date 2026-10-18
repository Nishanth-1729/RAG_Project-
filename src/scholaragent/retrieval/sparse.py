"""BM25 keyword retrieval, implemented from scratch with a tokenizer tuned for scientific text.

Exact terms matter in papers (GMRES, IFMM, H2-matrix, O(N log N), Equation 14), so the tokenizer
keeps identifiers intact AND adds their sub-tokens ("low-rank" -> low-rank, low, rank).
"""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import Sequence

import numpy as np

STOPWORDS = frozenset("""a about above after again all also am an and any are as at be because been before being below between
both but by can could did do does doing down during each few for from further had has have having he her here hers him his how
i if in into is it its itself just me more most my no nor not now of off on once only or other our out over own same she should so
some such than that the their them then there these they this those through to too under until up very was we were what when
where which while who whom why will with would you your us may might shall paper propose proposed show shown""".split())

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[-_.'][A-Za-z0-9]+)*")
_BIGO_RE = re.compile(r"O\s*\(([^)]{1,30})\)")


def stem(w: str) -> str:
    """Tiny suffix stripper (no external dependency); keeps acronyms and short tokens untouched."""
    if len(w) <= 4 or any(c.isdigit() for c in w):
        return w
    for suf, rep in (("ies", "y"), ("sses", "ss"), ("ing", ""), ("edly", ""), ("ed", ""), ("ly", ""), ("es", ""), ("s", "")):
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            if suf == "s" and w.endswith(("ss", "us", "is")):
                continue
            return w[: -len(suf)] + rep
    return w


def tokenize(text: str) -> list[str]:
    # "O(N log N)" -> "bigo n log n" so complexity statements are matchable
    text = _BIGO_RE.sub(lambda m: " bigo " + m.group(1) + " ", text)
    toks: list[str] = []
    for raw in _TOKEN_RE.findall(text):
        low = raw.lower()
        parts = re.split(r"[-_.']", low)
        cands = [low] + (parts if len(parts) > 1 else [])
        for t in cands:
            if t in STOPWORDS or len(t) < 2 and not t.isdigit():
                continue
            toks.append(stem(t))
    return toks


class BM25:
    def __init__(self, k1: float = 1.2, b: float = 0.75):
        self.k1, self.b = k1, b
        self.ids: list[str] = []
        self.paper_ids: np.ndarray = np.array([], dtype=object)
        self.doc_len: np.ndarray = np.zeros(0)
        self.avgdl = 0.0
        self.postings: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self.idf: dict[str, float] = {}

    def fit(self, ids: Sequence[str], paper_ids: Sequence[str], texts: Sequence[str]) -> "BM25":
        self.ids = list(ids)
        self.paper_ids = np.array(list(paper_ids), dtype=object)
        post: dict[str, list[tuple[int, int]]] = defaultdict(list)
        lens = np.zeros(len(texts))
        for i, t in enumerate(texts):
            toks = tokenize(t)
            lens[i] = len(toks)
            for term, tf in Counter(toks).items():
                post[term].append((i, tf))
        self.doc_len = lens
        self.avgdl = float(lens.mean()) if len(lens) else 0.0
        n = len(texts)
        self.postings, self.idf = {}, {}
        for term, lst in post.items():
            arr = np.array(lst, dtype=np.int64)
            self.postings[term] = (arr[:, 0], arr[:, 1].astype(np.float32))
            df = len(lst)
            self.idf[term] = math.log(1.0 + (n - df + 0.5) / (df + 0.5))
        return self

    def scores(self, query: str) -> np.ndarray:
        s = np.zeros(len(self.ids), dtype=np.float32)
        if not self.ids:
            return s
        norm = self.k1 * (1 - self.b + self.b * self.doc_len / max(self.avgdl, 1e-9))
        for term, qtf in Counter(tokenize(query)).items():
            p = self.postings.get(term)
            if p is None:
                continue
            idx, tf = p
            s[idx] += self.idf[term] * (tf * (self.k1 + 1)) / (tf + norm[idx]) * (1 + 0.1 * (qtf - 1))
        return s

    def search(self, query: str, k: int = 30, paper_ids: set[str] | None = None,
               mask: np.ndarray | None = None) -> list[tuple[str, float]]:
        s = self.scores(query)
        if paper_ids is not None:
            s = np.where(np.isin(self.paper_ids, list(paper_ids)), s, 0.0)
        if mask is not None:
            s = np.where(mask, s, 0.0)
        if not s.any():
            return []
        k = min(k, int((s > 0).sum()))
        top = np.argpartition(-s, k - 1)[:k]
        top = top[np.argsort(-s[top])]
        return [(self.ids[i], float(s[i])) for i in top if s[i] > 0]
