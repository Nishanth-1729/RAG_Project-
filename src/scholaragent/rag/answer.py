"""Grounded answer generation: answer ONLY from the supplied evidence, cite every claim, admit gaps."""
from __future__ import annotations

from dataclasses import dataclass, field

from ..llm.client import LLM, get_llm
from ..retrieval.hybrid import Hit
from ..utils.logging import agent_scope
from .citations import Evidence, extract_citation_ids, render_references, strip_invalid_citations
from .context_builder import build_context

ANSWER_SYSTEM = """You are a careful scientific research assistant. Answer the question using ONLY the numbered evidence passages provided.

Rules:
1. Every factual statement must end with the citation(s) of the passage(s) that support it, written as [1] or [1][3]. Use only the passage numbers you were given.
2. Never add facts from your own knowledge, never invent numbers, equations, names or citations.
3. If the evidence only partly answers the question, answer the supported part and clearly say what the evidence does not cover.
4. If the evidence does not answer the question at all, reply exactly: INSUFFICIENT_EVIDENCE: <one sentence on what is missing>.
5. Be precise and concise. Preserve exact terminology, numbers and units from the passages. When comparing papers, name each paper explicitly.
6. Do not cite passages that you did not use. Do not write a reference list."""


@dataclass
class Answer:
    question: str
    text: str
    evidence: list[Evidence]
    cited_ids: list[int] = field(default_factory=list)
    insufficient: bool = False
    removed_citations: list[int] = field(default_factory=list)
    queries: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        used = set(self.cited_ids)
        return {"question": self.question, "answer": self.text, "insufficient_evidence": self.insufficient,
                "cited_ids": self.cited_ids, "queries": self.queries,
                "evidence": [e.to_dict() for e in self.evidence],
                "references": render_references(self.evidence, used or None)}


REFLECT_NOTE = """Your previous attempt concluded that the evidence is insufficient. Look again, passage by passage: if ANY passage contains information that answers the question
(even only partly), answer from it, cite it, and say explicitly which part is not covered. Reply INSUFFICIENT_EVIDENCE only if no passage is relevant at all."""


class AnswerGenerator:
    def __init__(self, llm: LLM | None = None, max_context_tokens: int = 5500, reflect: bool = False):
        self.llm = llm or get_llm()
        self.max_context_tokens = max_context_tokens
        self.reflect = reflect          # give the model a second look before it abstains

    def answer(self, question: str, hits: list[Hit], instructions: str = "", max_tokens: int = 900) -> Answer:
        with agent_scope("answer"):
            ctx, evidence = build_context(hits, self.max_context_tokens)
            if not evidence:
                return Answer(question, "INSUFFICIENT_EVIDENCE: no relevant passages were retrieved.", [], [], True)
            user = f"EVIDENCE PASSAGES:\n{ctx}\n\nQUESTION: {question}\n{instructions}\nAnswer:"
            text = self.llm.complete(ANSWER_SYSTEM, user, max_tokens=max_tokens, tag="answer").strip()
            if self.reflect and text.upper().startswith("INSUFFICIENT_EVIDENCE"):
                text2 = self.llm.complete(ANSWER_SYSTEM, user.replace("Answer:", f"{REFLECT_NOTE}\nAnswer:"), max_tokens=max_tokens, tag="answer_reflect").strip()
                text = text2 or text
            insufficient = text.upper().startswith("INSUFFICIENT_EVIDENCE")
            text, removed = strip_invalid_citations(text, {e.id for e in evidence})
            cited = sorted(set(extract_citation_ids(text)))
            return Answer(question, text, evidence, cited, insufficient, removed)
