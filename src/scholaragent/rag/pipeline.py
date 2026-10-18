"""RAG pipeline: (rewrite) -> hybrid retrieval -> (rerank) -> grounded answer with citations."""
from __future__ import annotations

import time
from typing import Sequence

from ..llm.client import LLM, get_llm
from ..retrieval.hybrid import Hit
from ..retrieval.library import Library
from .answer import Answer, AnswerGenerator
from .fusion_helpers import decomposed_retrieve, multi_query_retrieve


class RagPipeline:
    def __init__(self, library: Library, llm: LLM | None = None):
        self.lib = library
        self.llm = llm or get_llm()
        self.gen = AnswerGenerator(self.llm, reflect=True)

    def retrieve(self, question: str, paper_ids: Sequence[str] | None = None, k: int = 8, mode: str = "hybrid_rerank",
                 rewrite: bool = False, history=None, n_candidates: int = 30, max_per_paper: int | None = None,
                 decompose: bool = False) -> tuple[list[Hit], list[str]]:
        from ..retrieval.vectorstore import Filter
        self.lib.load()
        flt = Filter(paper_ids=set(paper_ids)) if paper_ids is not None else None
        if mode == "hybrid_rerank" and self.lib.reranker is None:
            mode = "hybrid"
        if decompose and not rewrite:
            return decomposed_retrieve(self.lib.retriever, question, mode, k=k, llm=self.llm, n_candidates=n_candidates, flt=flt)
        return multi_query_retrieve(self.lib.retriever, question, mode, k=k, rewrite=rewrite, llm=self.llm,
                                    n_candidates=n_candidates, flt=flt, max_per_paper=max_per_paper)

    def ask(self, question: str, paper_ids: Sequence[str] | None = None, k: int = 8, mode: str = "hybrid_rerank",
            rewrite: bool = False, history=None, instructions: str = "", max_per_paper: int | None = None,
            decompose: bool = False) -> tuple[Answer, dict]:
        t0 = time.perf_counter()
        hits, queries = self.retrieve(question, paper_ids, k, mode, rewrite, history, max_per_paper=max_per_paper, decompose=decompose)
        t1 = time.perf_counter()
        ans = self.gen.answer(question, hits, instructions)
        ans.queries = queries
        return ans, {"retrieval_s": round(t1 - t0, 3), "generation_s": round(time.perf_counter() - t1, 3)}
