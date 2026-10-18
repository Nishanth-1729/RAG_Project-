"""Query understanding: rewrite follow-up questions and expand into retrieval queries."""
from __future__ import annotations

from pydantic import BaseModel, Field

from ..llm.client import LLM, get_llm
from ..utils.logging import agent_scope


class RewriteOut(BaseModel):
    standalone_question: str
    search_queries: list[str] = Field(description="2-4 keyword-style queries covering different phrasings and key technical terms")


REWRITE_SYSTEM = """You prepare questions for a scientific paper retrieval system.
Given a user question (and optionally the chat history), produce:
- standalone_question: the question rewritten so it can be understood without the history (keep all technical terms).
- search_queries: 2-4 short queries using precise technical terminology and synonyms (acronyms spelled out AND abbreviated) that would match passages of papers answering the question."""


def rewrite_query(question: str, history: list[tuple[str, str]] | None = None, llm: LLM | None = None) -> RewriteOut:
    llm = llm or get_llm()
    hist = "\n".join(f"{r}: {t}" for r, t in (history or [])[-4:])
    user = (f"CHAT HISTORY:\n{hist}\n\n" if hist else "") + f"QUESTION: {question}"
    with agent_scope("query_rewriter"):
        out = llm.generate_or_none(RewriteOut, REWRITE_SYSTEM, user, max_tokens=300, tag="rewrite")
    return out or RewriteOut(standalone_question=question, search_queries=[question])


class Decomposition(BaseModel):
    is_multi_part: bool = Field(description="true if the question compares/contrasts two or more specific methods, papers or settings, or asks several distinct things")
    sub_queries: list[str] = Field(description="one self-contained retrieval query per compared entity/part (2-3 items), each naming the entity AND the aspect asked about; empty if not multi-part")


DECOMP_SYS = """You split multi-part scientific questions into retrieval queries.
If the question compares or contrasts specific methods/papers/settings (or asks distinct things), return one self-contained query for EACH of them
that names the entity and the aspect asked about (e.g. 'BandPC training configuration', 'Neural Preconditioned Born Series training configuration').
For ordinary single-focus questions return is_multi_part=false and an empty list."""


def decompose_query(question: str, llm: LLM | None = None) -> list[str]:
    llm = llm or get_llm()
    with agent_scope("query_decomposer"):
        out = llm.generate_or_none(Decomposition, DECOMP_SYS, f"QUESTION: {question}", max_tokens=200, tag="decompose")
    return [q for q in (out.sub_queries if out and out.is_multi_part else []) if len(q) > 8][:3]
