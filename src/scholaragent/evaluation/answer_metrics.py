"""Answer-level metrics: correctness, faithfulness, citation accuracy/completeness, hallucination rate.

The evaluation judge uses its own rubric/prompt (different from the production verifier) and can be complemented
by the fine-tuned NLI model (an independent architecture) to reduce self-judging bias.
"""
from __future__ import annotations

from typing import Literal, Sequence

from pydantic import BaseModel, Field

from ..agents.verifier_agent import VerifierAgent
from ..llm.client import LLM, get_llm
from ..rag.citations import Evidence


class Correct(BaseModel):
    label: Literal["correct", "partially_correct", "incorrect"]
    reason: str = Field(description="one short sentence")


class Entail(BaseModel):
    verdict: Literal["entailed", "not_entailed", "contradicted"] = Field(
        description="entailed: the passages state or directly imply the claim; contradicted: they state the opposite; not_entailed: otherwise")


CORRECT_SYS = """You grade answers to questions about scientific papers against a reference answer written by an expert.
- correct: the answer contains the key facts of the reference and nothing that contradicts it.
- partially_correct: it contains some but not all key facts, or adds a wrong detail.
- incorrect: wrong, off-topic, or says the information is not available (INSUFFICIENT_EVIDENCE) when the reference has an answer.
Ignore citation markers like [1]. Judge content only."""

ENTAIL_SYS = """Natural-language-inference check. Decide whether the PASSAGES entail the CLAIM.
Entailed = every specific element of the claim (entities, numbers, relations) is stated in or directly implied by the passages. Otherwise not_entailed; use contradicted only if the passages state the opposite."""


class EvalJudge:
    def __init__(self, llm: LLM | None = None):
        self.llm = llm or get_llm()

    def correctness(self, question: str, reference: str, answer: str) -> float:
        if answer.strip().upper().startswith("INSUFFICIENT_EVIDENCE"):
            return 0.0
        r = self.llm.generate_or_none(Correct, CORRECT_SYS, f"QUESTION: {question}\n\nREFERENCE ANSWER: {reference}\n\nANSWER TO GRADE: {answer}", max_tokens=120, tag="eval_correct")
        return {"correct": 1.0, "partially_correct": 0.5, "incorrect": 0.0}.get(r.label, 0.0) if r else 0.0

    def entails(self, claim: str, passages: Sequence[Evidence]) -> str:
        if not passages:
            return "not_entailed"
        ctx = "\n\n".join(f"PASSAGE {i}: {e.text[:1600]}" for i, e in enumerate(passages, 1))
        r = self.llm.generate_or_none(Entail, ENTAIL_SYS, f"CLAIM: {claim}\n\n{ctx}", max_tokens=60, tag="eval_entail")
        return r.verdict if r else "not_entailed"


def claim_metrics(text: str, evidence: list[Evidence], verifier: VerifierAgent, judge: EvalJudge, nli=None) -> dict:
    """Decompose `text` into claims and score them against (a) the passages each claim cites and (b) all passages."""
    ev = {e.id: e for e in evidence}
    claims = verifier.extract_claims(text)
    if not claims:
        return {"n_claims": 0}
    n_cited = cit_ok = faith_ok = halluc = 0
    nli_ok = 0
    detail = []
    for c in claims:
        cited = [ev[i] for i in c.cited if i in ev]
        v_all = judge.entails(c.claim, evidence)
        faith_ok += v_all == "entailed"
        halluc += v_all != "entailed"
        detail.append({"claim": c.claim, "cited": c.cited, "llm_entailed": v_all == "entailed"})
        if cited:
            n_cited += 1
            cit_ok += judge.entails(c.claim, cited) == "entailed"
        if nli is not None:
            nli_ok += nli.predict_label(c.claim, [e.text for e in evidence]) in ("SUPPORTED", "PARTIALLY_SUPPORTED")
    n = len(claims)
    out = {"n_claims": n, "n_cited_claims": n_cited, "citation_correct": cit_ok, "faithful": faith_ok,
           "hallucinated": halluc, "detail": detail, "evidence_texts": [e.text for e in evidence]}
    if nli is not None:
        out["nli_faithful"] = nli_ok
    return out


def aggregate_claims(rows: Sequence[dict]) -> dict:
    n = sum(r.get("n_claims", 0) for r in rows)
    nc = sum(r.get("n_cited_claims", 0) for r in rows)
    out = {"claims": n,
           "faithfulness": round(sum(r.get("faithful", 0) for r in rows) / n, 4) if n else None,
           "hallucination_rate": round(sum(r.get("hallucinated", 0) for r in rows) / n, 4) if n else None,
           "citation_accuracy": round(sum(r.get("citation_correct", 0) for r in rows) / nc, 4) if nc else None,
           "citation_completeness": round(nc / n, 4) if n else None}
    if any("nli_faithful" in r for r in rows):
        out["faithfulness_nli"] = round(sum(r.get("nli_faithful", 0) for r in rows) / n, 4) if n else None
    return out
