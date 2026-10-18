"""Search Agent - only finds literature (no summarising, no gap finding)."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Sequence

from pydantic import BaseModel, Field

from ..llm.client import LLM, get_llm
from ..retrieval.embedder import Embedder
from ..search.fetcher import fetch_pdf
from ..search.models import PaperRecord
from ..search.service import search_literature
from ..utils.logging import agent_scope, get_logger

log = get_logger("search_agent")


class QueryPlan(BaseModel):
    queries: list[str] = Field(description="6-10 diverse academic search queries (2-6 words each): synonyms, sub-topics, method names, task names")


QUERY_SYS = """You generate search queries for academic databases (arXiv, Semantic Scholar, Crossref).
Given a research question produce diverse, precise queries: the core topic, synonyms, specific method families, neighbouring terminology and typical benchmark/task names.
Queries must be short keyword phrases (no full sentences, no boolean operators, no quotes)."""


class Relevance(BaseModel):
    score: int = Field(description="0 unrelated, 1 tangential (same broad field, different problem), 2 relevant, 3 directly addresses the research question")


REL_SYS = """You screen candidate papers for a literature review. Given the research question and a paper's title and abstract, rate its relevance:
3 = directly addresses the question, 2 = relevant (a method/study clearly within the topic), 1 = tangential (shares keywords or a broad field but studies a different problem),
0 = unrelated. Judge by what the paper actually does, not by shared words (e.g. 'preconditioning' in optimisers is not about preconditioners for linear solvers)."""


class SearchAgent:
    def __init__(self, embedder: Embedder, llm: LLM | None = None):
        self.emb = embedder
        self.llm = llm or get_llm()

    def plan_queries(self, question: str, n: int = 8, avoid: Sequence[str] = (), hint: str = "") -> list[str]:
        with agent_scope("search_agent"):
            extra = f"\nQueries already used (do not repeat; find new angles): {list(avoid)}" if avoid else ""
            out = self.llm.generate_or_none(QueryPlan, QUERY_SYS, f"RESEARCH QUESTION: {question}\n{hint}{extra}",
                                            max_tokens=300, tag="plan_queries", temperature=0.0 if not avoid else 0.3)
        qs = [q.strip() for q in (out.queries if out else []) if 3 <= len(q.strip()) <= 90]
        if not qs:
            qs = [question[:120]]
        if question[:120] not in qs and not avoid:
            qs.insert(0, question[:120])
        return list(dict.fromkeys(qs))[:n + 1]

    def search(self, question: str, queries: Sequence[str], *, year_from=None, year_to=None,
               sources=("arxiv", "semantic_scholar", "crossref"), per_source: int = 20) -> tuple[list[PaperRecord], dict]:
        with agent_scope("search_agent"):
            return search_literature(queries, sources=sources, per_source=per_source, year_from=year_from, year_to=year_to,
                                     rank_query=question, embed_fn=self.emb)

    def filter_relevant(self, question: str, records: list[PaperRecord], keep: int, min_score: int = 2) -> list[PaperRecord]:
        """LLM screening of the top candidates (title + abstract) so off-topic keyword matches never reach the corpus."""
        with agent_scope("search_agent"):
            pool = records[: max(keep * 3, 12)]

            def judge(r: PaperRecord):
                out = self.llm.generate_or_none(Relevance, REL_SYS, f"RESEARCH QUESTION: {question}\n\nTITLE: {r.title}\nABSTRACT: {(r.abstract or 'n/a')[:1200]}",
                                                max_tokens=40, tag="relevance", retries=1)
                return out.score if out else 1
            scores = self.llm.map(judge, pool, workers=16)
            scores = [s if isinstance(s, int) else 1 for s in scores]
            for r, sc in zip(pool, scores):
                r.score_parts["llm_relevance"] = float(sc)
            good = [r for r, sc in zip(pool, scores) if sc >= min_score]
            if len(good) < min(3, keep):                                   # too strict for this topic: admit tangential papers
                good += [r for r, sc in zip(pool, scores) if sc == 1 and r not in good][: keep - len(good)]
            return good                                                      # never pad with unscreened papers

    @staticmethod
    def fetch_fulltext(records: list[PaperRecord], want: int, workers: int = 3) -> tuple[list[tuple[PaperRecord, object]], list[PaperRecord]]:
        """Download PDFs in rank order until `want` succeeded. Returns (successes[(record, path)], failures)."""
        ok: list[tuple[PaperRecord, object]] = []
        failed: list[PaperRecord] = []
        i = 0
        with ThreadPoolExecutor(max_workers=workers) as ex:
            while len(ok) < want and i < len(records):
                batch = records[i:i + max(workers, want - len(ok))]
                i += len(batch)
                for r, p in zip(batch, ex.map(fetch_pdf, batch)):
                    (ok.append((r, p)) if p else failed.append(r))
        ok = ok[:want]
        return ok, failed + records[i:]
