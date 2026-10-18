"""Shared multi-query retrieval used by the pipeline and by the answer evaluation."""
from __future__ import annotations

from ..retrieval.fusion import max_per_group, rrf
from ..retrieval.hybrid import Hit, Retriever
from .query import decompose_query, rewrite_query


def multi_query_retrieve(rt: Retriever, question: str, mode: str, k: int = 8, rewrite: bool = False, llm=None,
                         n_candidates: int = 30, flt=None, max_per_paper: int | None = None) -> tuple[list[Hit], list[str]]:
    queries = [question]
    if rewrite:
        rw = rewrite_query(question, None, llm)
        queries = [rw.standalone_question] + [q for q in rw.search_queries if q != rw.standalone_question]
    if len(queries) == 1:
        return rt.search(question, k=k, mode=mode, flt=flt, n_candidates=n_candidates, max_per_paper=max_per_paper), queries
    pools = [rt.search(q, k=n_candidates, mode="hybrid" if mode == "hybrid_rerank" else mode, flt=flt, n_candidates=n_candidates) for q in queries]
    fused = rrf([[h.chunk.chunk_id for h in p] for p in pools])
    by_id = {h.chunk.chunk_id: h for p in reversed(pools) for h in p}
    cands = [by_id[cid] for cid, _ in fused[:n_candidates]]
    if mode == "hybrid_rerank" and rt.reranker is not None:
        sc = rt.reranker.score(queries[0], [rt._rerank_text(h.chunk) for h in cands])
        for h, s in zip(cands, sc):
            h.scores["rerank"], h.score = float(s), float(s)
        cands.sort(key=lambda h: h.score, reverse=True)
    cands = max_per_group(cands, lambda h: h.chunk.paper_id, max_per_paper)[:k]
    for i, h in enumerate(cands, 1):
        h.rank = i
    return cands, queries


def decomposed_retrieve(rt: Retriever, question: str, mode: str, k: int = 8, llm=None, n_candidates: int = 30, flt=None,
                        per_sub: int = 4) -> tuple[list[Hit], list[str]]:
    """Comparison / multi-part questions: retrieve for every sub-query separately, then interleave (round robin) so each
    compared entity is represented in the evidence; the full question's own hits fill the remaining slots."""
    subs = decompose_query(question, llm)
    main = rt.search(question, k=k, mode=mode, flt=flt, n_candidates=n_candidates)
    if len(subs) < 2:
        return main, [question]
    lists = [rt.search(q, k=per_sub, mode=mode, flt=flt, n_candidates=n_candidates) for q in subs]
    merged, seen = [], set()
    for rank in range(per_sub):
        for lst in lists:
            if rank < len(lst) and lst[rank].chunk.chunk_id not in seen:
                seen.add(lst[rank].chunk.chunk_id)
                merged.append(lst[rank])
    for h in main:
        if h.chunk.chunk_id not in seen and len(merged) < k:
            seen.add(h.chunk.chunk_id)
            merged.append(h)
    merged = merged[:k]
    for i, h in enumerate(merged, 1):
        h.rank = i
    return merged, [question, *subs]
