"""Rank fusion."""
from __future__ import annotations

from typing import Sequence


def rrf(rankings: Sequence[Sequence[str]], k: int = 60, weights: Sequence[float] | None = None) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion:  RRF(d) = sum_r w_r / (k + rank_r(d)),  rank starting at 1."""
    weights = weights or [1.0] * len(rankings)
    score: dict[str, float] = {}
    for w, ranking in zip(weights, rankings):
        for rank, doc in enumerate(ranking, start=1):
            score[doc] = score.get(doc, 0.0) + w / (k + rank)
    return sorted(score.items(), key=lambda x: x[1], reverse=True)


def max_per_group(items: list, key, limit: int | None) -> list:
    """Keep at most `limit` items per group (e.g. per paper) while preserving order."""
    if not limit:
        return items
    counts: dict = {}
    out = []
    for it in items:
        g = key(it)
        if counts.get(g, 0) < limit:
            out.append(it)
            counts[g] = counts.get(g, 0) + 1
    return out


def rsf(dense: Sequence[tuple[str, float]], sparse: Sequence[tuple[str, float]], weights: Sequence[float] = (1.0, 1.0)) -> list[tuple[str, float]]:
    """Relative Score Fusion: min-max normalise each retriever's scores to [0,1], then weighted sum."""
    def norm(lst):
        if not lst:
            return {}
        vals = [s for _, s in lst]
        lo, hi = min(vals), max(vals)
        return {d: (s - lo) / (hi - lo + 1e-9) for d, s in lst}
    a, b = norm(dense), norm(sparse)
    score = {d: weights[0] * a.get(d, 0.0) + weights[1] * b.get(d, 0.0) for d in set(a) | set(b)}
    return sorted(score.items(), key=lambda x: x[1], reverse=True)
