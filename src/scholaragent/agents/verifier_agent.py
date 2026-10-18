"""Verification Agent: claim-level fact checking with a self-correcting loop.

draft answer -> atomic claims -> evidence retrieval (cited passages + fresh retrieval) -> verdict
SUPPORTED / PARTIALLY_SUPPORTED / UNSUPPORTED / CONTRADICTED
   not SUPPORTED -> rewrite query -> retrieve again -> re-judge
       found evidence      -> keep / correct the claim
       still no evidence   -> delete the claim (or qualify it if partially supported)
-> corrected answer built ONLY from verified claims, with a claim-evidence matrix.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from pydantic import BaseModel, Field

from ..llm.client import LLM, get_llm
from ..rag.answer import Answer
from ..rag.citations import Evidence, evidence_from_hits, extract_citation_ids, strip_invalid_citations
from ..retrieval.hybrid import Hit
from ..retrieval.library import Library
from ..utils.logging import agent_scope, get_logger
from .schemas import ClaimVerdict

log = get_logger("verifier")


class ClaimItem(BaseModel):
    claim: str = Field(description="one atomic, self-contained factual statement (names the method/paper explicitly; no pronouns)")
    cited: list[int] = Field(default_factory=list, description="citation numbers attached to the claim in the draft")


class ClaimList(BaseModel):
    claims: list[ClaimItem]


class QueryOut(BaseModel):
    queries: list[str] = Field(description="2-3 search queries (technical keywords) that would find a passage proving or refuting the claim")


class Rewrite(BaseModel):
    corrected_claim: str = Field(description="the claim corrected so that it is fully consistent with the passage; keep it minimal")


EXTRACT_SYS = """Split the answer into atomic factual claims. Each claim must be a single checkable statement that stands alone (replace pronouns/'the method' by names), preserving numbers exactly.
Skip pure transitions, meta statements ('the evidence does not cover ...') and questions. Record the citation numbers [n] that the answer attaches to each claim. Max 14 claims."""

JUDGE_SYS = """You are a strict scientific fact-checker. Decide whether the CLAIM is supported by the numbered PASSAGES.

SUPPORTED: every element of the claim (entities, numbers, direction of effect, scope) is stated in, or directly follows from, the passages.
PARTIALLY_SUPPORTED: the core is supported but some detail is not (e.g. a number, an unqualified 'always/all/arbitrary', an extra property).
UNSUPPORTED: the passages do not contain the information needed to judge the claim.
CONTRADICTED: the passages state something incompatible with the claim (different number, opposite result, explicit limitation that negates it).

Do not use outside knowledge. Do not be lenient: if a specific number or comparison in the claim is absent, it is not SUPPORTED."""

REWRITE_SYS = """The CLAIM conflicts with or overstates what the PASSAGE says. Rewrite the claim so it is exactly consistent with the passage (fix numbers, scope, direction). Keep it one sentence."""

COMPOSE_SYS = """You rewrite a draft answer into a final answer using ONLY the VERIFIED CLAIMS provided.
- Keep the structure and tone of the draft, but include only information contained in the verified claims.
- Keep the citation markers [n] given with each claim; put them at the end of the sentence they support. Do not invent new numbers.
- Where a claim is flagged PARTIAL, state it with the appropriate caveat.
- If a CORRECTED note is given, use the corrected wording.
- Do not mention the verification process."""


@dataclass
class VerifiedClaim:
    claim: str
    cited: list[int]
    initial: str = "UNSUPPORTED"
    final: str = "UNSUPPORTED"
    action: str = "kept"             # kept | re-retrieved | corrected | removed | qualified
    rationale: str = ""
    evidence_ids: list[int] | None = None
    corrected: str | None = None

    def to_dict(self, ev: dict[int, Evidence]) -> dict:
        support = [ev[i].to_dict() for i in (self.evidence_ids or []) if i in ev]
        return {"claim": self.claim, "cited": self.cited, "initial_status": self.initial, "final_status": self.final,
                "action": self.action, "rationale": self.rationale, "corrected_claim": self.corrected, "evidence": support}


class VerifierAgent:
    def __init__(self, library: Library, llm: LLM | None = None, nli=None, nli_mode: str = "primary", nli_conf: float = 0.6):
        """nli: optional fine-tuned claim-verification cross-encoder (training/train_verifier.py).
        nli_mode: 'primary' - the NLI model decides whenever it is confident (>= nli_conf), the LLM judge only breaks ties;
                  'veto'    - LLM decides, NLI may only downgrade a SUPPORTED verdict; 'off' - LLM only."""
        self.lib = library
        self.llm = llm or get_llm()
        self.nli = nli
        self.nli_mode, self.nli_conf = nli_mode, nli_conf

    # ------------------------------------------------------------------ claims
    def extract_claims(self, text: str) -> list[ClaimItem]:
        out = self.llm.generate_or_none(ClaimList, EXTRACT_SYS, f"ANSWER:\n{text}", max_tokens=1400, tag="extract_claims")
        return [c for c in (out.claims if out else []) if len(c.claim) > 12][:14]

    def judge(self, claim: str, passages: Sequence[Evidence]) -> ClaimVerdict:
        if not passages:
            return ClaimVerdict(label="UNSUPPORTED", best_passage=None, rationale="no passages available")
        if self.nli is not None and self.nli_mode == "primary":
            P = self.nli.proba([claim] * len(passages), [e.text for e in passages])
            for j, label in ((0, "SUPPORTED"), (1, "PARTIALLY_SUPPORTED"), (3, "CONTRADICTED")):     # strongest evidence class first
                if P[:, j].max() > 0.5:
                    k = int(P[:, j].argmax())
                    if P[k, j] >= self.nli_conf:
                        return ClaimVerdict(label=label, best_passage=k + 1, rationale=f"entailment model: {label.lower().replace('_', ' ')} "
                                            f"(p={P[k, j]:.2f}) by passage [{passages[k].id}]")
                    break
            else:
                if P[:, 2].min() >= self.nli_conf:           # every passage says 'unrelated' with confidence
                    return ClaimVerdict(label="UNSUPPORTED", best_passage=None, rationale="entailment model: no passage supports the claim")
        ctx = "\n\n".join(f"[{e.id}] ({e.short_ref}, {e.location})\n{e.text[:1800]}" for e in passages)
        v = self.llm.generate_or_none(ClaimVerdict, JUDGE_SYS, f"CLAIM: {claim}\n\nPASSAGES:\n{ctx}", max_tokens=250, tag="judge_claim")
        if v is None:
            return ClaimVerdict(label="UNSUPPORTED", best_passage=None, rationale="verifier failed to produce a verdict")
        if self.nli is not None and self.nli_mode == "veto" and v.label == "SUPPORTED":          # independent second opinion: veto only when very confident
            veto = self.nli.strict_veto(claim, [e.text for e in passages])
            if veto:
                return ClaimVerdict(label=veto, best_passage=None, rationale=f"NLI verifier veto ({veto.lower()}); LLM said: {v.rationale}")
        return v

    # ---------------------------------------------------------------- pipeline
    def verify_answer(self, answer: Answer, paper_ids: Sequence[str] | None = None, max_rounds: int = 1) -> dict:
        """Verify + self-correct an Answer. Returns report dict incl. 'final_answer', 'claims', 'summary', 'evidence'."""
        with agent_scope("verifier"):
            if answer.insufficient or answer.text.strip().upper().startswith("INSUFFICIENT_EVIDENCE"):
                # the generator already declined to answer from the evidence: nothing to verify, keep the honest abstention
                return {"final_answer": answer.text, "draft_answer": answer.text, "claims": [], "summary": _summary([]),
                        "evidence": [e.to_dict() for e in answer.evidence], "cited_ids": [], "note": "abstained"}
            pool: dict[int, Evidence] = {e.id: e for e in answer.evidence}
            next_id = max(pool, default=0) + 1
            claims = self.extract_claims(answer.text)
            if not claims:
                return {"final_answer": answer.text, "claims": [], "summary": _summary([]), "evidence": [e.to_dict() for e in answer.evidence],
                        "note": "no verifiable claims found"}
            vclaims = [VerifiedClaim(c.claim, c.cited) for c in claims]

            def check(vc: VerifiedClaim):
                return self._check_one(vc, pool, paper_ids, max_rounds)

            new_evs: list[list[Evidence]] = []
            results = self.llm.map(check, vclaims, workers=8)
            for vc, res in zip(vclaims, results):
                if isinstance(res, Exception):
                    vc.final, vc.action, vc.rationale = "UNSUPPORTED", "removed", f"verification error: {res!r}"[:200]
                    new_evs.append([])
                else:
                    new_evs.append(res)
            # register newly retrieved evidence with fresh ids (stable, deduped by chunk_id)
            by_chunk = {e.chunk_id: e.id for e in pool.values()}
            for vc, evs in zip(vclaims, new_evs):
                ids = []
                for e in evs:
                    if e.chunk_id not in by_chunk:
                        e.id = next_id
                        next_id += 1
                        pool[e.id] = e
                        by_chunk[e.chunk_id] = e.id
                    ids.append(by_chunk[e.chunk_id])
                vc.evidence_ids = ids
            final_text = self._compose(answer, vclaims, pool)
            used = set(extract_citation_ids(final_text))
            summary = _summary(vclaims)
            return {"final_answer": final_text, "draft_answer": answer.text, "claims": [vc.to_dict(pool) for vc in vclaims],
                    "summary": summary, "evidence": [e.to_dict() for e in sorted(pool.values(), key=lambda e: e.id)],
                    "cited_ids": sorted(used)}

    def _check_one(self, vc: VerifiedClaim, pool: dict[int, Evidence], paper_ids, max_rounds: int) -> list[Evidence]:
        # 1) judge against the cited passages + top fresh retrieval for the claim itself
        cited = [pool[i] for i in vc.cited if i in pool]
        fresh = self._retrieve(vc.claim, paper_ids, k=4, exclude={e.chunk_id for e in cited}, start_id=1000)
        cand = (cited + fresh)[:6]
        for i, e in enumerate(cand, 1):             # local numbering for the judge prompt
            e = Evidence(**{**e.__dict__, "id": i})
            cand[i - 1] = e
        verdict = self.judge(vc.claim, cand)
        vc.initial = vc.final = verdict.label
        vc.rationale = verdict.rationale
        best = [cand[verdict.best_passage - 1]] if verdict.best_passage and 1 <= verdict.best_passage <= len(cand) else []
        if verdict.label == "SUPPORTED":
            return best or cand[:1]
        # 2) self-correction: retrieve again with rewritten queries over a wider scope
        for _ in range(max_rounds):
            qs = self.llm.generate_or_none(QueryOut, "Write search queries that would find the passage proving or refuting this claim in the papers.",
                                           f"CLAIM: {vc.claim}", max_tokens=150, tag="claim_queries")
            more: list[Evidence] = []
            seen = {e.chunk_id for e in cand}
            for q in (qs.queries if qs else [])[:3]:
                for e in self._retrieve(q, paper_ids, k=3, exclude=seen, start_id=2000):
                    seen.add(e.chunk_id)
                    more.append(e)
            if not more:
                break
            cand2 = [Evidence(**{**e.__dict__, "id": i}) for i, e in enumerate(more[:6], 1)]
            v2 = self.judge(vc.claim, cand2)
            if v2.label in ("SUPPORTED", "PARTIALLY_SUPPORTED", "CONTRADICTED") and _rank(v2.label) > _rank(verdict.label) - (1 if verdict.label == "CONTRADICTED" else 0):
                verdict, cand = v2, cand2
                vc.final, vc.rationale, vc.action = v2.label, v2.rationale, "re-retrieved"
                best = [cand[v2.best_passage - 1]] if v2.best_passage and 1 <= v2.best_passage <= len(cand) else cand[:1]
                if v2.label == "SUPPORTED":
                    return best
                break
        # 3) resolve remaining problems
        if vc.final == "CONTRADICTED" and best:
            rw = self.llm.generate_or_none(Rewrite, REWRITE_SYS, f"CLAIM: {vc.claim}\n\nPASSAGE:\n{best[0].text[:1800]}", max_tokens=200, tag="rewrite_claim")
            if rw:
                v3 = self.judge(rw.corrected_claim, best)
                if v3.label == "SUPPORTED":
                    vc.corrected, vc.final, vc.action = rw.corrected_claim, "SUPPORTED", "corrected"
                    return best
            vc.action = "removed"
            return best
        if vc.final == "PARTIALLY_SUPPORTED":
            vc.action = "qualified" if vc.action != "re-retrieved" else vc.action
            return best
        vc.action = "removed"          # UNSUPPORTED after retry: delete the claim
        return []

    def _retrieve(self, query: str, paper_ids, k: int, exclude: set[str], start_id: int) -> list[Evidence]:
        hits = self.lib.search(query, paper_ids, k=k + len(exclude), mode="hybrid_rerank" if self.lib.reranker else "hybrid")
        hits = [h for h in hits if h.chunk.chunk_id not in exclude][:k]
        return evidence_from_hits(hits, start_id)

    # ------------------------------------------------------------------ compose
    def _compose(self, answer: Answer, vclaims: list[VerifiedClaim], pool: dict[int, Evidence]) -> str:
        keep = [vc for vc in vclaims if vc.final in ("SUPPORTED", "PARTIALLY_SUPPORTED")]
        if not keep:
            return "INSUFFICIENT_EVIDENCE: none of the statements in the draft could be verified against the source papers."
        lines = []
        for vc in keep:
            cites = "".join(f"[{i}]" for i in (vc.evidence_ids or []) if i in pool)
            tag = "PARTIAL" if vc.final == "PARTIALLY_SUPPORTED" else "OK"
            wording = vc.corrected or vc.claim
            note = f" (CORRECTED from: {vc.claim})" if vc.corrected else ""
            lines.append(f"- [{tag}] {wording} {cites}{note}")
        user = f"QUESTION: {answer.question}\n\nDRAFT ANSWER:\n{answer.text}\n\nVERIFIED CLAIMS:\n" + "\n".join(lines) + "\n\nFinal answer:"
        text = self.llm.complete(COMPOSE_SYS, user, max_tokens=900, tag="compose_final").strip()
        text, _ = strip_invalid_citations(text, set(pool))
        return text


_ORDER = {"CONTRADICTED": 0, "UNSUPPORTED": 1, "PARTIALLY_SUPPORTED": 2, "SUPPORTED": 3}


def _rank(label: str) -> int:
    return _ORDER[label]


def _summary(vclaims: list[VerifiedClaim]) -> dict:
    from collections import Counter
    ini = Counter(v.initial for v in vclaims)
    fin = Counter(v.final for v in vclaims)
    return {"claims_checked": len(vclaims),
            "initial": {k: ini.get(k, 0) for k in _ORDER},
            "final": {k: fin.get(k, 0) for k in _ORDER},
            "removed": sum(v.action == "removed" for v in vclaims),
            "corrected": sum(v.action == "corrected" for v in vclaims),
            "re_retrieved": sum(v.action == "re-retrieved" for v in vclaims)}
