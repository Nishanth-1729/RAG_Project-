"""Benchmark / training-data generation.

Questions are generated from *paragraphs* (evidence units), so relevance labels do not depend on how a retrieval
system chunks the papers: a retrieved chunk is relevant iff it contains the evidence span (see retrieval_metrics.py).

Question types follow the project spec: factual, methodological, mathematical, results, limitations, comparison.
"""
from __future__ import annotations

import json
import random
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

from pydantic import BaseModel, Field

from ..ingestion.models import ParsedDocument
from ..llm.client import LLM, get_llm
from ..rag.citations import quote_support
from ..utils.logging import get_logger
from ..utils.text import normalize_for_match, sentences

log = get_logger("dataset")

TYPES = ["factual", "methodological", "mathematical", "results", "limitations", "comparison"]
TYPE_SECTIONS = {
    "factual": {"abstract", "introduction", "background", "related_work"},
    "methodological": {"method"},
    "mathematical": {"method", "background"},
    "results": {"experiments", "results"},
    "limitations": {"discussion", "limitations", "conclusion"},
}
TYPE_GUIDE = {
    "factual": "a factual question about the problem, motivation, definitions, claims or contributions stated in the passage",
    "methodological": "a question about HOW the method/algorithm works or is set up (steps, architecture, training procedure, design choice)",
    "mathematical": "a question about a formula, quantity, complexity bound, assumption or theorem in the passage",
    "results": "a question about experimental setup or a quantitative/qualitative result (numbers, comparisons with baselines, speed-ups)",
    "limitations": "a question about a limitation, failure case, assumption that restricts the method, or future work",
}


@dataclass
class EvidenceUnit:
    paper_id: str
    title: str
    section: str
    canonical: str
    page: int
    text: str
    kind: str = "paragraph"


class QAOut(BaseModel):
    question: str = Field(description="self-contained question")
    reference_answer: str = Field(description="answer in 1-3 sentences using only the passage")
    evidence_span: str = Field(description="the 1-2 consecutive sentences of the passage, copied VERBATIM, that contain the answer")
    answerable: bool = Field(description="false if the passage does not contain a clear, specific answerable fact")


class ComparisonOut(BaseModel):
    question: str = Field(description="a question that requires BOTH passages to answer (comparing or contrasting the two works)")
    reference_answer: str
    evidence_span_a: str = Field(description="verbatim 1-2 sentences from passage A")
    evidence_span_b: str = Field(description="verbatim 1-2 sentences from passage B")
    answerable: bool


QA_SYS = """You write evaluation questions for a retrieval system over a large library of scientific papers.
Given ONE passage from a paper, write ONE {qtype} question: {guide}.

Hard requirements:
- The question must be answerable from the passage alone.
- The question must be SELF-CONTAINED and SPECIFIC: someone searching a library of hundreds of papers must be able to identify the right passage from the question. Name the specific method, problem, quantity or technique (e.g. 'the Flexible GMRES preconditioner' rather than 'the method'). NEVER write 'this paper', 'the paper', 'the authors', 'the passage', 'this passage', 'the text', 'the section', 'the proposed method' or 'the above' - the reader has no access to the passage.
- Do not copy long phrases from the passage; paraphrase naturally, like a researcher would ask.
- evidence_span must be copied verbatim from the passage (1-2 consecutive sentences containing the answer).
- If the passage is only references, notation lists, boilerplate or has no clear fact, set answerable=false."""

CMP_SYS = """You write evaluation questions for a retrieval system over a library of scientific papers.
You get TWO passages (A and B) from different papers. Write ONE question that can only be answered by combining BOTH (e.g. compare or contrast the approaches, or ask what both share).
It must be self-contained and specific: name the two methods/problems explicitly. NEVER write 'passage A/B', 'the passages', 'the two papers', 'the first/second paper', 'this paper' or 'the authors'. Give a reference answer using only the passages and copy a verbatim 1-2 sentence evidence span from each.
If the two passages cannot be meaningfully compared, set answerable=false."""


class Quality(BaseModel):
    specificity: int = Field(description="1-5; 5 = names a specific method/quantity/setting so that exactly one passage in a large library would answer it; 1 = generic textbook question many papers answer")
    answer_correct: bool = Field(description="is the reference answer correct and complete according to the passage(s)?")
    self_contained: bool = Field(description="can the question be understood without seeing the passage (no dangling references like 'the proposed method' or 'Table 5')?")


QUAL_SYS = """You audit evaluation questions for a retrieval system over a library of hundreds of scientific papers about machine learning and numerical linear algebra.
Given the evidence passage(s), a question and its reference answer, rate: specificity (1-5), whether the reference answer is correct and complete, and whether the question is self-contained.
Be strict about specificity: 'What does the matrix A represent?' or 'What is the system Ax=b?' are generic (1-2) because dozens of papers answer them."""


def quality_filter(items: list[dict], llm: LLM | None = None, min_spec: int = 4) -> list[dict]:
    llm = llm or get_llm()

    def one(it):
        ps = "\n\n".join(f"PASSAGE {i + 1} (paper: {e['title']}):\n{e['paragraph']}" for i, e in enumerate(it["evidences"]))
        q = llm.generate_or_none(Quality, QUAL_SYS, f"{ps}\n\nQUESTION: {it['question']}\nREFERENCE ANSWER: {it['reference_answer']}",
                                 max_tokens=150, tag="quality", retries=1)
        if q is None or q.specificity < min_spec or not q.answer_correct or not q.self_contained:
            return None
        return {**it, "quality": {"specificity": q.specificity}}
    return [r for r in llm.map(one, items, workers=24) if isinstance(r, dict)]


class Ambig(BaseModel):
    answered_by: list[int] = Field(description="indices of OTHER passages that answer the question about as well as the source passage (empty if none)")


AMBIG_SYS = """You check whether an evaluation question is AMBIGUOUS in a library of scientific papers.
You get a question, the SOURCE passage it was written from, and several OTHER passages retrieved from the library.
List the indices of OTHER passages that contain the specific information needed to answer the question just as well as the source passage
(same facts for the same method/setting - not merely the same topic). Return an empty list if the question can only be answered from the source."""


def ambiguity_filter(items: list[dict], candidates_fn, llm: LLM | None = None) -> list[dict]:
    """candidates_fn(item) -> list of competing passages (str) that do NOT contain the evidence span."""
    llm = llm or get_llm()

    def one(it):
        others = candidates_fn(it)
        if not others:
            return it
        src = "\n\n".join(f"SOURCE PASSAGE {i + 1}: {e['paragraph']}" for i, e in enumerate(it["evidences"]))
        oth = "\n\n".join(f"[{i}] {t[:1100]}" for i, t in enumerate(others, 1))
        r = llm.generate_or_none(Ambig, AMBIG_SYS, f"{src}\n\nQUESTION: {it['question']}\n\nOTHER PASSAGES:\n{oth}", max_tokens=80, tag="ambiguity", retries=1)
        return None if (r is None or r.answered_by) else {**it, "n_competitors_checked": len(others)}
    return [r for r in llm.map(one, items, workers=24) if isinstance(r, dict)]


META_REF = re.compile(
    r"\b(this|the|that|these|those|above|following|given|provided|first|second|both|two)\s+(passages?|papers?|texts?|sections?|works?|studies|study|articles?|documents?|excerpts?|snippets?|authors?)\b"
    r"|\bpassages?\s+[AB12]\b|\b(the|this) proposed (method|approach|model|framework|algorithm)\b|\bthe authors?\b", re.I)


def has_meta_reference(question: str) -> bool:
    return bool(META_REF.search(question))


def evidence_units(doc: ParsedDocument, min_chars: int = 260, max_chars: int = 1500) -> list[EvidenceUnit]:
    out = []
    for s in doc.sections:
        if s.canonical in ("references", "acknowledgments"):
            continue
        for p in s.paragraphs:
            if p.kind not in ("paragraph", "equation") or not (min_chars <= len(p.text) <= max_chars):
                continue
            alpha = sum(c.isalpha() for c in p.text) / len(p.text)
            if alpha < 0.6:
                continue
            out.append(EvidenceUnit(doc.paper_id, doc.title, s.title, s.canonical, p.page_start, p.text, p.kind))
    return out


def _check_span(span: str, passage: str, thr: float = 92.0) -> bool:
    return len(span) >= 25 and quote_support(span, passage) >= thr


def _leaks(question: str, passage: str, n: int = 9) -> bool:
    qt, pt = normalize_for_match(question).split(), normalize_for_match(passage)
    return any(" ".join(qt[i:i + n]) in pt for i in range(max(0, len(qt) - n + 1)))


def generate_qa(units: list[EvidenceUnit], qtype: str, llm: LLM | None = None, temperature: float = 0.3) -> list[dict]:
    llm = llm or get_llm()
    sys = QA_SYS.format(qtype=qtype, guide=TYPE_GUIDE[qtype])

    def one(u: EvidenceUnit):
        user = f"PAPER TITLE: {u.title}\nSECTION: {u.section}\n\nPASSAGE:\n{u.text}"
        r = llm.generate_or_none(QAOut, sys, user, max_tokens=900, temperature=temperature, tag="gen_qa", retries=1)
        if not r or not r.answerable or len(r.question) < 20 or not _check_span(r.evidence_span, u.text) or _leaks(r.question, u.text):
            return None
        if has_meta_reference(r.question):
            return None
        return {"type": qtype, "question": r.question.strip(), "reference_answer": r.reference_answer.strip(),
                "evidences": [{"paper_id": u.paper_id, "title": u.title, "section": u.section, "page": u.page,
                               "span": r.evidence_span.strip(), "paragraph": u.text}]}
    res = llm.map(one, units, workers=24)
    return [r for r in res if isinstance(r, dict)]


def generate_comparison(pairs: list[tuple[EvidenceUnit, EvidenceUnit]], llm: LLM | None = None) -> list[dict]:
    llm = llm or get_llm()

    def one(pair):
        a, b = pair
        user = f"PASSAGE A (paper: {a.title}):\n{a.text}\n\nPASSAGE B (paper: {b.title}):\n{b.text}"
        r = llm.generate_or_none(ComparisonOut, CMP_SYS, user, max_tokens=1000, temperature=0.3, tag="gen_cmp", retries=1)
        if not r or not r.answerable or not _check_span(r.evidence_span_a, a.text) or not _check_span(r.evidence_span_b, b.text):
            return None
        if has_meta_reference(r.question):
            return None
        return {"type": "comparison", "question": r.question.strip(), "reference_answer": r.reference_answer.strip(),
                "evidences": [{"paper_id": u.paper_id, "title": u.title, "section": u.section, "page": u.page, "span": sp.strip(),
                               "paragraph": u.text} for u, sp in ((a, r.evidence_span_a), (b, r.evidence_span_b))]}
    res = llm.map(one, pairs, workers=24)
    return [r for r in res if isinstance(r, dict)]


def sample_units(units: list[EvidenceUnit], qtype: str, n: int, rng: random.Random, per_paper_cap: int = 6) -> list[EvidenceUnit]:
    secs = TYPE_SECTIONS[qtype]
    pool = [u for u in units if u.canonical in secs]
    if qtype == "mathematical":
        pool = [u for u in pool if re.search(r"[=≤≥∑∫∂∇]|\bO\(|complexity|bound|theorem|lemma|eigenvalue|condition number|norm", u.text)]
    if qtype == "results":
        pool = [u for u in pool if re.search(r"\d", u.text)]
    if qtype == "limitations":
        pool = [u for u in units if u.canonical in secs or re.search(r"\b(limitation|however|future work|cannot|fails?|drawback|restrict|not (?:able|applicable))\b", u.text, re.I)]
    rng.shuffle(pool)
    out, cnt = [], {}
    for u in pool:
        if cnt.get(u.paper_id, 0) >= per_paper_cap:
            continue
        out.append(u)
        cnt[u.paper_id] = cnt.get(u.paper_id, 0) + 1
        if len(out) >= n:
            break
    return out


def dedupe_questions(items: list[dict]) -> list[dict]:
    from rapidfuzz import fuzz
    kept: list[dict] = []
    for it in items:
        if any(fuzz.token_set_ratio(it["question"], k["question"]) > 88 for k in kept):
            continue
        kept.append(it)
    return kept


def save_jsonl(items: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for i, it in enumerate(items):
            it = {"qid": it.get("qid") or f"q{i:04d}", **{k: v for k, v in it.items() if k != "qid"}}
            f.write(json.dumps(it, ensure_ascii=False) + "\n")


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
