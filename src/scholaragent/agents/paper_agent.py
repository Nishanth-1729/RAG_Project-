"""Paper Analyst Agent: structured, evidence-linked understanding of ONE paper.

For each group of fields it (1) retrieves the most relevant chunks *inside the paper* (section prior + hybrid
retrieval), (2) asks the LLM for schema-constrained JSON with passage ids and verbatim quotes, and
(3) verifies every quote against the source chunk. Items whose quote cannot be found are dropped, so every
surviving field has page/section/chunk provenance.
"""
from __future__ import annotations

from typing import Any

from ..llm.client import LLM, get_llm
from ..rag.citations import Evidence, evidence_from_hits, quote_support
from ..rag.context_builder import build_context
from ..retrieval.library import Library
from ..retrieval.vectorstore import Filter
from ..utils.logging import agent_scope, get_logger
from .schemas import (ALL_FIELDS, FIELDS_CORE, FIELDS_EXP, FIELDS_LIM, AnalysisCore, AnalysisExperiments, AnalysisLimits, Limitations)

log = get_logger("paper_agent")

GROUPS = {
    "core": dict(schema=AnalysisCore, fields=FIELDS_CORE,
                 sections={"abstract", "introduction", "method", "background", "other"},
                 queries=["what problem does the paper address and why it matters", "the proposed method and main idea",
                          "key equation formulation algorithm", "computational complexity cost scaling"]),
    "exp": dict(schema=AnalysisExperiments, fields=FIELDS_EXP,
                sections={"experiments", "results", "discussion", "method", "other"},
                queries=["datasets and benchmark problems used in experiments", "baseline methods compared against",
                         "evaluation metrics", "main quantitative results speedup accuracy improvement"]),
    "lim": dict(schema=AnalysisLimits, fields=FIELDS_LIM,
                sections={"discussion", "limitations", "conclusion", "introduction", "abstract", "results"},
                queries=["limitations and weaknesses of the approach", "future work and open problems",
                         "main contributions of this paper", "assumptions required, fails when, does not generalize"]),
}

SYSTEM = """You are a meticulous scientific paper analyst. You read numbered evidence passages from ONE paper and extract structured information.

Rules:
- Use ONLY the passages. If a field is not supported by the passages, leave it empty (null / empty list). Never guess.
- For every item give: `text` (concise paraphrase, max 40 words), `passage` (the passage number) and `quote` (a short VERBATIM quote of at most 30 words copied exactly from that passage).
- Prefer concrete content (numbers, names of datasets/baselines, explicit limitations) over generic statements.
- Keep lists short and non-redundant (max 5 items per list)."""


LIMITS_SYSTEM = """You read passages from ONE paper and list the LIMITATIONS of the approach proposed in that paper.
A limitation is any stated or implied shortcoming, restriction or cost: an explicit one ('only tested on', 'cannot', 'does not', 'unlikely to', 'expensive', 'requires', 'limited by'),
or one implied by a future-work item ('we plan to extend to X' => 'currently does not support X'; 'future work: multi-GPU training' => 'limited by single-GPU memory').
Write each as a short statement of the shortcoming (max 30 words) with the passage number and a verbatim quote (<= 30 words) copied from that passage. Give 2-5 items; return an empty list only if the passages truly contain nothing."""


class PaperAnalyst:
    def __init__(self, library: Library, llm: LLM | None = None):
        self.lib = library
        self.llm = llm or get_llm()

    # ------------------------------------------------------------------ public
    def analyze(self, paper_id: str, force: bool = False) -> dict | None:
        store = self.lib.store
        if not force and (cached := store.get_analysis(paper_id)):
            return cached
        meta = store.get_paper(paper_id)
        if not meta:
            return None
        with agent_scope("paper_analyst"):
            chunks = self.lib.chunks_for(paper_id)
            if not chunks:
                return None
            out: dict[str, Any] = {"paper_id": paper_id, "title": meta["title"], "authors": meta["authors"],
                                   "year": meta["year"], "venue": meta.get("venue"), "doi": meta.get("doi"),
                                   "status": meta["status"], "fields": {}, "stats": {"dropped": 0, "kept": 0}}
            groups = ["core"] if meta["status"] == "abstract_only" else list(GROUPS)
            for g in groups:
                try:
                    self._run_group(paper_id, g, out)
                except Exception as e:  # noqa: BLE001
                    log.warning("analysis group %s failed for %s: %r", g, paper_id, e)
                    out.setdefault("errors", []).append(f"{g}: {e!r}"[:200])
            if not out["fields"]:
                return None
            store.save_analysis(paper_id, out, self.llm.model)
            return out

    # ---------------------------------------------------------------- internals
    def _select_chunks(self, paper_id: str, group: dict, k: int = 7):
        flt_sec = Filter(paper_ids={paper_id}, sections=group["sections"])
        flt_all = Filter(paper_ids={paper_id})
        seen: dict[str, Any] = {}
        for q in group["queries"]:
            for flt in (flt_sec, flt_all):
                hits = self.lib.retriever.search(q, k=3, mode="hybrid_rerank" if self.lib.reranker else "hybrid",
                                                 flt=flt, n_candidates=12)
                for h in hits:
                    if h.chunk.chunk_id not in seen or h.score > seen[h.chunk.chunk_id].score:
                        seen[h.chunk.chunk_id] = h
                if hits and flt is flt_sec:
                    break
        hits = sorted(seen.values(), key=lambda h: h.score, reverse=True)[:k]
        hits.sort(key=lambda h: h.chunk.ordinal)          # reading order helps the LLM
        return hits

    def _run_group(self, paper_id: str, gname: str, out: dict) -> None:
        group = GROUPS[gname]
        hits = self._select_chunks(paper_id, group)
        if gname == "core" and out["status"] == "abstract_only":
            hits = hits[:1]
        ctx, evidence = build_context(hits, max_tokens=3600)
        ev = {e.id: e for e in evidence}
        if not ev:
            return
        user = f"PAPER: {out['title']}\n\nEVIDENCE PASSAGES:\n{ctx}\n\nExtract the structured analysis."
        res = self.llm.generate(group["schema"], SYSTEM, user, max_tokens=1600, tag=f"analyze_{gname}")
        data = res.model_dump()
        if gname == "lim":          # limitations get their own, narrower pass (a 7B model skips them when they are one field among many)
            lim = self.llm.generate_or_none(Limitations, LIMITS_SYSTEM, f"PAPER: {out['title']}\n\nEVIDENCE PASSAGES:\n{ctx}\n\nList the limitations.",
                                            max_tokens=700, tag="analyze_limitations")
            data["limitations"] = lim.model_dump()["limitations"] if lim else []
        for field in group["fields"]:
            raw = data.get(field)
            items = raw if isinstance(raw, list) else ([raw] if raw else [])
            verified = []
            for it in items:
                e: Evidence | None = ev.get(it.get("passage"))
                if e is None:
                    out["stats"]["dropped"] += 1
                    continue
                score = quote_support(it.get("quote", ""), e.text)
                if score < 85 or len(it.get("quote", "")) < 8:
                    out["stats"]["dropped"] += 1
                    continue
                out["stats"]["kept"] += 1
                verified.append({"text": it["text"].strip(), "evidence": {
                    "chunk_id": e.chunk_id, "section": e.section, "page_start": e.page_start, "page_end": e.page_end,
                    "quote": it["quote"].strip(), "quote_match": round(score, 1)}})
            if verified:
                out["fields"][field] = verified
