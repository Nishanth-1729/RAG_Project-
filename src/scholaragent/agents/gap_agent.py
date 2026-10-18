"""Research-Gap Agent - deliberately conservative.

limitations + future work (evidence-linked) -> embed & cluster -> keep recurring themes -> candidate direction
-> targeted novelty SEARCH (literature APIs + the indexed corpus) -> LLM judge: already addressed? -> reject / keep.
Output is always phrased as a *potential research direction*, never as 'nobody has done X'.
"""
from __future__ import annotations

import re

from pydantic import BaseModel, Field

from ..llm.client import LLM, get_llm
from ..rag.citations import paper_label
from ..retrieval.embedder import Embedder
from ..search.service import search_literature
from ..utils.logging import agent_scope, get_logger

log = get_logger("gap")


class ThemeOut(BaseModel):
    name: str = Field(description="specific 3-8 word name of the recurring limitation, e.g. 'Poor generalisation to unseen matrix distributions' (never generic words like 'shared problem')")
    description: str = Field(description="1-2 sentences describing the shared unresolved problem")


class DirectionOut(BaseModel):
    direction: str = Field(description="one sentence describing a concrete potential research direction that addresses the limitation")
    search_queries: list[str] = Field(description="2-3 literature-search queries (keyword style) that would find prior work ALREADY doing this")


class NoveltyJudgement(BaseModel):
    already_addressed: bool = Field(description="true if any listed paper clearly already does what the direction proposes")
    closest_indices: list[int] = Field(description="indices of the 1-3 closest listed papers (can be empty)")
    reasoning: str = Field(description="1-2 sentences")


THEME_SYS = "You summarise a cluster of limitations/open problems that were extracted from different research papers. Describe the shared underlying problem only; do not add anything not present in the items."
DIR_SYS = """You propose ONE concrete, specific potential research direction that would address the recurring limitation given.
It must be technically specific (what to do, not 'improve X'). Also give search queries that would find prior work that already does it."""
JUDGE_SYS = """You judge whether a proposed research direction is ALREADY addressed by existing papers.
You see the direction and candidate papers (title + abstract snippet). Say already_addressed=true only if a candidate clearly and directly does what the direction proposes."""


class GapAgent:
    def __init__(self, embedder: Embedder, llm: LLM | None = None):
        self.emb = embedder
        self.llm = llm or get_llm()

    def find_gaps(self, analyses: list[dict], corpus_titles: dict[str, dict], question: str, *, min_support: int = 2,
                  max_themes: int = 6, year_from: int | None = None, search_sources=("arxiv", "semantic_scholar"),
                  progress=None) -> dict:
        with agent_scope("gap"):
            items = self._collect(analyses)
            stats = {"n_papers": len(analyses), "n_limitation_items": len(items)}
            if len(items) < 3:
                return {"gaps": [], "rejected": [], "stats": stats, "note": "too few limitation statements to analyse"}
            themes = self._cluster(items, min_support, max_themes)
            stats["n_themes"] = len(themes)
            gaps, rejected = [], []
            for th in themes:
                if progress:
                    progress(f"Examining recurring limitation ({len(th['items'])} mentions)")
                res = self._examine(th, question, analyses, corpus_titles, year_from, search_sources)
                (rejected if res.get("rejected") else gaps).append(res)
            gaps.sort(key=lambda g: (g["confidence_score"]), reverse=True)
            stats.update(n_candidates=len(themes), n_rejected=len(rejected))
            return {"gaps": gaps, "rejected": rejected, "stats": stats}

    # ------------------------------------------------------------------ steps
    @staticmethod
    def _collect(analyses: list[dict]) -> list[dict]:
        out = []
        for a in analyses:
            for fname in ("limitations", "future_work"):
                for it in a["fields"].get(fname, []):
                    out.append({"paper_id": a["paper_id"], "paper": paper_label(a.get("authors") or [], a.get("year")),
                                "title": a["title"], "kind": fname, "text": it["text"], "evidence": it["evidence"]})
        return out

    def _cluster(self, items: list[dict], min_support: int, max_themes: int) -> list[dict]:
        """Coherent clusters only: average-linkage with a cosine-distance threshold (no forced cluster count)."""
        from sklearn.cluster import AgglomerativeClustering
        X = self.emb([it["text"] for it in items])
        cl = AgglomerativeClustering(n_clusters=None, distance_threshold=0.42, metric="cosine", linkage="average").fit(X)
        groups: dict[int, list[int]] = {}
        for i, l in enumerate(cl.labels_):
            groups.setdefault(int(l), []).append(i)
        themes = []
        for idxs in groups.values():
            papers = {items[i]["paper_id"] for i in idxs}
            if len(papers) < min_support:          # recurring = mentioned by >= min_support DIFFERENT papers
                continue
            themes.append({"items": [items[i] for i in idxs], "papers": sorted(papers)})
        themes.sort(key=lambda t: (len(t["papers"]), len(t["items"])), reverse=True)
        return themes[:max_themes]

    def _examine(self, th: dict, question: str, analyses: list[dict], corpus: dict[str, dict], year_from, sources) -> dict:
        lim_text = "\n".join(f"- ({it['paper']}) {it['text']}" for it in th["items"][:10])
        theme = self.llm.generate_or_none(ThemeOut, THEME_SYS, f"LIMITATION ITEMS:\n{lim_text}", max_tokens=250, tag="gap_theme")
        if theme is None or re.search(r"shared (underlying )?problem|^limitation|various|general", theme.name, re.I):
            return {"rejected": True, "reason": "cluster too heterogeneous to describe as one specific limitation", "theme": theme.name if theme else "?"}
        d = self.llm.generate_or_none(DirectionOut, DIR_SYS,
                                      f"RESEARCH AREA: {question}\nRECURRING LIMITATION: {theme.name}. {theme.description}\nEVIDENCE:\n{lim_text}",
                                      max_tokens=350, tag="gap_direction")
        if d is None:
            return {"rejected": True, "reason": "direction generation failed", "theme": theme.name}
        # --- novelty search: external literature + papers of this run
        cands, sstats = search_literature(d.search_queries[:3], sources=sources, per_source=10, year_from=year_from,
                                          rank_query=d.direction, embed_fn=self.emb)
        cands = cands[:12]
        listing = "\n".join(f"[{i}] {c.title} ({c.year}): {(c.abstract or '')[:350]}" for i, c in enumerate(cands))
        judge = self.llm.generate_or_none(NoveltyJudgement, JUDGE_SYS,
                                          f"PROPOSED DIRECTION: {d.direction}\n\nCANDIDATE PAPERS:\n{listing or '(none found)'}",
                                          max_tokens=300, tag="gap_judge") if cands else None
        closest = [cands[i] for i in (judge.closest_indices if judge else []) if 0 <= i < len(cands)][:3]
        base = {"theme": theme.name, "theme_description": theme.description, "direction": d.direction,
                "n_supporting_papers": len(th["papers"]), "n_papers_total": len(analyses),
                "support": [{"paper": it["paper"], "paper_id": it["paper_id"], "title": it["title"], "text": it["text"],
                             "evidence": it["evidence"]} for it in th["items"][:8]],
                "novelty_search": {"queries": d.search_queries[:3], "papers_examined": len(cands),
                                   "closest_work": [{"title": c.title, "year": c.year, "url": c.url or (f"https://arxiv.org/abs/{c.arxiv_id}" if c.arxiv_id else None)} for c in closest]}}
        if judge and judge.already_addressed:
            return {**base, "rejected": True, "reason": judge.reasoning}
        frac = len(th["papers"]) / max(1, len(analyses))
        score = min(1.0, 0.5 * min(1.0, frac * 3) + 0.3 * min(1.0, len(cands) / 10) + (0.2 if not closest else 0.05))
        if not cands:                       # the novelty check could not run - never present that as 'no prior work found'
            score = min(score, 0.3)
            statement = ("The novelty search returned no papers (sources unavailable or rate limited), so whether this direction is "
                         "already addressed could NOT be assessed; treat it as an unverified candidate.")
        else:
            statement = (f"Our search of {len(cands)} related papers did not identify work directly addressing this "
                         f"within the searched corpus; it is a potential research direction, not a proven gap.")
        conf = "HIGH" if score > 0.75 and len(th["papers"]) >= 4 else ("MEDIUM" if score > 0.45 else "LOW")
        return {**base, "rejected": False, "confidence": conf, "confidence_score": round(score, 3), "statement": statement,
                "judge_reasoning": judge.reasoning if judge else "no candidate papers retrieved"}
