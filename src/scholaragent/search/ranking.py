"""Relevance scoring for candidate papers.

S = w1*semantic + w2*title_overlap + w3*abstract_overlap + w4*recency + w5*citations_per_year
Citations are deliberately a small weight and age-normalised, so new papers are not punished.
"""
from __future__ import annotations

import math
import re
from datetime import date
from typing import Callable, Sequence

import numpy as np

from .models import PaperRecord

STOP = set("a an the of for in on and or to with by from using via is are be as at into towards toward based study approach new".split())
WEIGHTS = {"semantic": 0.50, "title": 0.15, "abstract": 0.15, "recency": 0.10, "citations": 0.10}


def _toks(s: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", s.lower()) if t not in STOP and len(t) > 1}


def score_papers(query: str, records: list[PaperRecord], embed_fn: Callable[[Sequence[str]], np.ndarray] | None,
                 year_from: int | None = None, year_to: int | None = None,
                 weights: dict[str, float] | None = None) -> list[PaperRecord]:
    w = {**WEIGHTS, **(weights or {})}
    if not records:
        return []
    q = _toks(query)
    now = date.today().year
    sem = np.zeros(len(records))
    if embed_fn is not None:
        qe = np.asarray(embed_fn([query]), dtype=np.float32)[0]
        de = np.asarray(embed_fn([f"{r.title}. {r.abstract[:1200]}" for r in records]), dtype=np.float32)
        qe /= (np.linalg.norm(qe) + 1e-9)
        de /= (np.linalg.norm(de, axis=1, keepdims=True) + 1e-9)
        sem = de @ qe
        lo, hi = float(np.percentile(sem, 5)), float(np.percentile(sem, 99))
        sem = np.clip((sem - lo) / (hi - lo + 1e-9), 0, 1)
    for i, r in enumerate(records):
        t_ov = len(q & _toks(r.title)) / max(1, len(q))
        a_ov = len(q & _toks(r.abstract)) / max(1, len(q)) if r.abstract else 0.0
        yr = r.year or (now - 5)
        span = max(1, (year_to or now) - (year_from or now - 10))
        rec = min(1.0, max(0.0, (yr - (year_from or now - 10)) / span))
        age = max(1, now - yr + 1)
        cpy = (r.citation_count or 0) / age
        cit = min(1.0, math.log1p(cpy) / math.log1p(200))
        parts = {"semantic": float(sem[i]), "title": t_ov, "abstract": a_ov, "recency": rec, "citations": cit}
        r.score_parts = {k: round(v, 4) for k, v in parts.items()}
        r.score = round(sum(w[k] * v for k, v in parts.items()), 4)
        if not r.abstract:
            r.score *= 0.9        # slightly distrust records we cannot read
        if len(r.sources) > 1:
            r.score = round(r.score + 0.02 * (len(r.sources) - 1), 4)
    if year_from or year_to:
        records = [r for r in records if r.year is None or ((year_from or 0) <= r.year <= (year_to or 9999))]
    return sorted(records, key=lambda r: r.score, reverse=True)
