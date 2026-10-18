"""Retrieval metrics: Recall@K, Precision@K, MRR, NDCG@K - with chunking-independent relevance labels.

A retrieved chunk is RELEVANT to a question iff it comes from an evidence paper AND contains that evidence span
(>= 85% of the span's tokens appear in the chunk, in the normalised text). This makes systems with different
chunkers directly comparable. Graded relevance (for NDCG): 3 = contains the span, 2 = contains >= 50% of the
evidence paragraph, 1 = same paper and same section, 0 = otherwise.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Sequence

import numpy as np

from ..utils.text import normalize_for_match


@lru_cache(maxsize=200_000)
def _toks(text: str) -> frozenset[str]:
    return frozenset(normalize_for_match(text).split())


@lru_cache(maxsize=200_000)
def _norm(text: str) -> str:
    return normalize_for_match(text)


def contains_span(chunk_text: str, span: str, thr: float = 0.85) -> bool:
    ns, nc = _norm(span), _norm(chunk_text)
    if ns in nc:
        return True
    st = _toks(span)
    return bool(st) and len(st & _toks(chunk_text)) / len(st) >= thr and _ordered_overlap(ns, nc)


def _ordered_overlap(ns: str, nc: str) -> bool:
    """Guard against bag-of-words false positives: 3 consecutive words of the span's start AND of its end must appear."""
    w = ns.split()
    if len(w) < 8:
        return True
    return " ".join(w[:3]) in nc and " ".join(w[-3:]) in nc


def grade(chunk, evidences: Sequence[dict]) -> int:
    best = 0
    for ev in evidences:
        if chunk.paper_id != ev["paper_id"]:
            continue
        if contains_span(chunk.text, ev["span"]):
            return 3
        pt = _toks(ev["paragraph"])
        if pt and len(pt & _toks(chunk.text)) / len(pt) >= 0.5:
            best = max(best, 2)
        elif normalize_for_match(chunk.section) == normalize_for_match(ev["section"]):
            best = max(best, 1)
    return best


@dataclass
class QResult:
    qid: str
    qtype: str
    first_rank: int | None             # 1-based rank of the first relevant chunk (None = miss)
    relevant_flags: list[bool]         # per retrieved chunk
    grades: list[int]
    coverage: dict[int, float]         # K -> fraction of evidence units covered in top-K (comparison questions)
    ctx_tokens: int = 0
    n_evidence: int = 1
    ideal_grades: list[int] = field(default_factory=list)   # grades of ALL corpus chunks for this question, best first


def evaluate_ranking(question: dict, chunks: Sequence, max_k: int = 10, n_relevant_in_corpus: int | None = None,
                     ideal_grades: Sequence[int] | None = None) -> QResult:
    evs = question["evidences"]
    flags, grades = [], []
    for c in chunks[:max_k]:
        g = grade(c, evs)
        grades.append(g)
        flags.append(g == 3)
    first = next((i + 1 for i, f in enumerate(flags) if f), None)
    cov = {}
    for k in (1, 3, 5, 10):
        covered = sum(any(contains_span(c.text, e["span"]) and c.paper_id == e["paper_id"] for c in chunks[:k]) for e in evs)
        cov[k] = covered / len(evs)
    return QResult(question["qid"], question["type"], first, flags, grades, cov, sum(c.n_tokens for c in chunks[:5]), max(len(evs), n_relevant_in_corpus or 0), list(ideal_grades or []))


def ndcg_at_k(grades: Sequence[int], k: int, n_relevant: int = 1, ideal_grades: Sequence[int] | None = None) -> float:
    """Graded NDCG. The ideal ranking = the best possible ordering of ALL graded chunks in the corpus (falls back to
    `n_relevant` fully relevant chunks when the full grading is not available)."""
    dcg = sum((2 ** x - 1) / math.log2(i + 2) for i, x in enumerate(list(grades[:k])))
    ideal = sorted(ideal_grades, reverse=True)[:k] if ideal_grades else [3] * min(max(1, n_relevant), k)
    idcg = sum((2 ** x - 1) / math.log2(i + 2) for i, x in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def aggregate(results: Sequence[QResult], ks=(1, 3, 5, 10)) -> dict:
    n = len(results)
    if n == 0:
        return {}
    out: dict = {"n": n}
    for k in ks:
        out[f"recall@{k}"] = float(np.mean([r.first_rank is not None and r.first_rank <= k for r in results]))
        out[f"precision@{k}"] = float(np.mean([sum(r.relevant_flags[:k]) / k for r in results]))
        out[f"ndcg@{k}"] = float(np.mean([ndcg_at_k(r.grades, k, r.n_evidence, r.ideal_grades) for r in results]))
        out[f"coverage@{k}"] = float(np.mean([r.coverage.get(k, 0.0) for r in results]))
    out["mrr"] = float(np.mean([1.0 / r.first_rank if r.first_rank else 0.0 for r in results]))
    out["avg_ctx_tokens@5"] = float(np.mean([r.ctx_tokens for r in results]))
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in out.items()}


def by_type(results: Sequence[QResult]) -> dict[str, dict]:
    types = sorted({r.qtype for r in results})
    return {t: aggregate([r for r in results if r.qtype == t], ks=(1, 5, 10)) for t in types}


def bootstrap_ci(values: Sequence[float], n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    v = np.asarray(values, dtype=float)
    if len(v) == 0:
        return (0.0, 0.0)
    means = [rng.choice(v, size=len(v), replace=True).mean() for _ in range(n_boot)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))
