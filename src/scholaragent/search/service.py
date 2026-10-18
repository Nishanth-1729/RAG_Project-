"""Multi-source literature search: query -> candidates -> dedup -> relevance ranking."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Sequence

import numpy as np

from ..utils.logging import get_logger, trace
from . import arxiv, crossref, openalex, semantic_scholar
from .dedup import deduplicate
from .models import PaperRecord
from .ranking import score_papers

log = get_logger("search")
SOURCES = {
    "arxiv": lambda q, n, a, b: arxiv.search(q, n, a, b),
    "semantic_scholar": lambda q, n, a, b: semantic_scholar.search(q, n, a, b),
    "crossref": lambda q, n, a, b: crossref.search(q, n, a, b),
    "openalex": lambda q, n, a, b: openalex.search(q, n, a, b),
}


def search_literature(queries: Sequence[str], sources: Sequence[str] = ("arxiv", "semantic_scholar", "crossref"),
                      per_source: int = 20, year_from: int | None = None, year_to: int | None = None,
                      rank_query: str | None = None, embed_fn: Callable[[Sequence[str]], np.ndarray] | None = None,
                      min_score: float = 0.0) -> tuple[list[PaperRecord], dict]:
    """Returns (ranked deduplicated records, stats). Failing sources never abort the search."""
    srcs = [s for s in sources if s in SOURCES]
    raw: list[PaperRecord] = []

    def run_source(s: str):
        """One thread per source; queries to the same source run sequentially (respects per-host rate limits)."""
        got: list[PaperRecord] = []
        fails = 0
        for q in queries:
            try:
                res = SOURCES[s](q, per_source, year_from, year_to)
            except Exception as e:  # noqa: BLE001
                log.warning("%s failed for %r: %r", s, q, e)
                res = []
            if not res:
                fails += 1
                if fails >= 3 and not got:      # source is down/throttled: stop wasting time on it
                    break
            for r in res:
                r.queries = [q]
            got.extend(res)
        return s, got

    with ThreadPoolExecutor(max_workers=max(1, len(srcs))) as ex:
        results = list(ex.map(run_source, srcs))
    per_src: dict[str, int] = {}
    for s, res in results:
        per_src[s] = per_src.get(s, 0) + len(res)
        raw.extend(res)
    uniq = deduplicate(raw)
    ranked = score_papers(rank_query or " ".join(queries[:2]), uniq, embed_fn, year_from, year_to)
    if min_score:
        ranked = [r for r in ranked if r.score >= min_score]
    stats = {"queries": list(queries), "sources": list(sources), "candidates_raw": len(raw), "per_source": per_src,
             "after_dedup": len(uniq), "after_filter": len(ranked)}
    trace("search_literature", **{k: v for k, v in stats.items() if k != "queries"})
    return ranked, stats
