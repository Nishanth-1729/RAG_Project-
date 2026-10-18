"""Comparison Agent: turns per-paper analyses into an evidence-linked comparison table + landscape of approaches.

Cells are condensed from analysis items (which already carry verified quotes/pages), and each cell keeps the ids of
the items it was built from - so the table never contains an un-sourced comparison.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from ..llm.client import LLM, get_llm
from ..rag.citations import paper_label
from ..utils.logging import agent_scope, get_logger

log = get_logger("comparison")

COLUMNS = ["problem", "method", "complexity", "datasets", "main_result", "strength", "limitation"]


class Cell(BaseModel):
    text: str = Field(description="max 25 words; '-' if the items say nothing about it")
    items: list[str] = Field(default_factory=list, description="ids (like F3) of the analysis items used")


class Row(BaseModel):
    problem: Cell
    method: Cell
    complexity: Cell
    datasets: Cell
    main_result: Cell
    strength: Cell
    limitation: Cell
    approach_category: str = Field(description="2-4 word name of the family of approaches this paper belongs to")


class Landscape(BaseModel):
    class Approach(BaseModel):
        name: str
        description: str = Field(description="one or two sentences")
        paper_ids: list[str]
    approaches: list[Approach]
    synthesis: str = Field(description="3-6 sentences comparing the families: trade-offs, where they agree/disagree; refer to papers by their tag like [P2]")


ROW_SYS = """You condense the structured analysis of one scientific paper into a comparison-table row.
Use ONLY the listed analysis items (each has an id like F3). Every cell: max 25 words, plain language, keep numbers exact, and list the ids of the items you used.
If the items contain nothing for a cell, write '-' with no ids. 'strength' must be justified by the results/claims items; 'limitation' by the limitation items.
`approach_category` is a short name (2-4 words) for the family of methods (e.g. 'learned preconditioners', 'neural operators')."""

LAND_SYS = """You are a research analyst. Given short descriptions of papers (tagged [P1], [P2], ...), group them into 3-7 families of approaches
and write a brief synthesis of the trade-offs between the families. Use only the provided descriptions. Every paper must appear in exactly one approach."""


def _flatten(analysis: dict) -> tuple[str, dict[str, dict]]:
    lines, idmap = [], {}
    n = 0
    for fname, items in analysis["fields"].items():
        for it in items:
            n += 1
            fid = f"F{n}"
            idmap[fid] = {"field": fname, **it}
            lines.append(f"{fid} [{fname}] {it['text']}")
    return "\n".join(lines), idmap


class ComparisonAgent:
    def __init__(self, llm: LLM | None = None):
        self.llm = llm or get_llm()

    def compare(self, analyses: list[dict], question: str = "") -> dict:
        with agent_scope("comparison"):
            tags = {a["paper_id"]: f"P{i}" for i, a in enumerate(analyses, 1)}

            def one(a: dict) -> dict | None:
                text, idmap = _flatten(a)
                user = f"PAPER: {a['title']}\nRESEARCH QUESTION: {question or 'n/a'}\n\nANALYSIS ITEMS:\n{text}"
                row = self.llm.generate_or_none(Row, ROW_SYS, user, max_tokens=900, tag="compare_row")
                if row is None:
                    return None
                d = row.model_dump()
                cells = {}
                for col in COLUMNS:
                    c = d[col]
                    ev = [idmap[i]["evidence"] for i in c["items"] if i in idmap]
                    cells[col] = {"text": c["text"], "evidence": ev}
                return {"paper_id": a["paper_id"], "tag": tags[a["paper_id"]], "title": a["title"],
                        "label": paper_label(a.get("authors") or [], a.get("year")), "year": a.get("year"),
                        "category": d["approach_category"], "cells": cells}

            rows = [r for r in self.llm.map(one, analyses, workers=8) if isinstance(r, dict)]
            landscape = self._landscape(rows, question) if len(rows) >= 2 else None
            return {"columns": COLUMNS, "rows": rows, "landscape": landscape}

    def _landscape(self, rows: list[dict], question: str) -> dict | None:
        desc = "\n".join(f"[{r['tag']}] {r['title']} — approach: {r['category']}; method: {r['cells']['method']['text']}; "
                         f"limitation: {r['cells']['limitation']['text']}" for r in rows)
        out = self.llm.generate_or_none(Landscape, LAND_SYS, f"RESEARCH QUESTION: {question}\n\nPAPERS:\n{desc}",
                                        max_tokens=1500, tag="landscape")
        if out is None:
            return None
        tag2pid = {r["tag"]: r["paper_id"] for r in rows}
        d = out.model_dump()
        for ap in d["approaches"]:
            ap["paper_ids"] = [tag2pid[t.strip("[]")] for t in ap["paper_ids"] if t.strip("[]") in tag2pid]
        return d
