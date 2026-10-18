"""Report Generator: evidence-grounded literature report.

Every narrative section is written from evidence-linked facts, then claim-verified (and self-corrected) by the
Verifier. Citations are mapped to ONE global evidence index so each statement links back to paper/section/page.
"""
from __future__ import annotations

import datetime as dt
import re
from collections import OrderedDict
from typing import Callable

from ..llm.client import LLM, get_llm
from ..rag.answer import Answer
from ..rag.citations import Evidence, extract_citation_ids, paper_label
from ..retrieval.library import Library
from ..utils.logging import agent_scope, get_logger
from ..utils.text import slugify
from .verifier_agent import VerifierAgent

log = get_logger("report")

SECTION_SYS = """You write one section of a scientific literature report. Use ONLY the numbered FACTS (each is a verified statement from a paper, with its source).
Rules: cite the supporting fact number(s) like [3][7] after every factual sentence; never cite a number that is not listed; never add outside knowledge;
compare papers by name; be precise and concise; plain Markdown paragraphs or short bullet lists; no headings (the heading is added separately).
If the facts are insufficient for part of the task, say so briefly."""

ABSTRACT_SYS = "Write a 120-170 word abstract for the literature report from the section texts provided. Use only their content; keep citation markers [n] attached to facts exactly as they appear in the sections."


class ReportAgent:
    def __init__(self, library: Library, verifier: VerifierAgent, llm: LLM | None = None):
        self.lib = library
        self.verifier = verifier
        self.llm = llm or get_llm()

    # ------------------------------------------------------------------ facts
    def _facts(self, analyses: dict[str, dict], fields: list[str], per_paper: int = 2, max_facts: int = 40,
               paper_ids: list[str] | None = None) -> list[Evidence]:
        chunks = self.lib.index.chunks
        out: list[Evidence] = []
        for pid, a in analyses.items():
            if paper_ids is not None and pid not in paper_ids:
                continue
            n = 0
            for f in fields:
                for it in a["fields"].get(f, [])[:per_paper]:
                    ev = it["evidence"]
                    c = chunks.get(ev["chunk_id"])
                    text = f"{it['text']}\n(Source passage: {c.text[:900] if c else ev['quote']})"
                    out.append(Evidence(id=len(out) + 1, chunk_id=ev["chunk_id"], paper_id=pid, title=a["title"],
                                        authors=a.get("authors") or [], year=a.get("year"), section=ev["section"],
                                        page_start=ev["page_start"], page_end=ev["page_end"], text=text, doi=a.get("doi")))
                    n += 1
                    if n >= per_paper * 2:
                        break
            if len(out) >= max_facts:
                break
        return out[:max_facts]

    def _fact_listing(self, facts: list[Evidence]) -> str:
        return "\n".join(f"[{e.id}] ({e.short_ref}; {e.location}) {e.text.split(chr(10))[0]}" for e in facts)

    # ---------------------------------------------------------------- sections
    def _write_section(self, name: str, task: str, facts: list[Evidence], question: str, verify: bool,
                       registry: "Registry", paper_ids: list[str]) -> dict:
        if not facts:
            return {"name": name, "text": "_No sufficiently supported evidence was found for this section._", "claims": [], "summary": None}
        user = f"RESEARCH QUESTION: {question}\nSECTION: {name}\nTASK: {task}\n\nFACTS:\n{self._fact_listing(facts)}\n\nWrite the section:"
        text = _clean_section(self.llm.complete(SECTION_SYS, user, max_tokens=900, tag=f"section_{slugify(name)}"), name)
        ans = Answer(question=task, text=text, evidence=facts, cited_ids=sorted(set(extract_citation_ids(text))))
        report = None
        if verify:
            report = self.verifier.verify_answer(ans, paper_ids=paper_ids)
            text = _clean_section(report["final_answer"], name)
            pool = {e["id"]: e for e in report["evidence"]}
            claims = report["claims"]
            summary = report["summary"]
        else:
            pool = {e.id: e.to_dict() for e in facts}
            claims, summary = [], None
        text = registry.remap(text, pool)
        for c in claims:
            c["evidence"] = [{**e, "global_id": registry.gid(e["chunk_id"])} for e in c["evidence"]]
        return {"name": name, "text": text, "claims": claims, "summary": summary}

    # ---------------------------------------------------------------- main
    def generate(self, *, question: str, analyses: dict[str, dict], comparison: dict | None, gaps: dict | None,
                 search_stats: dict | None, graph: dict | None, paper_meta: dict[str, dict], verify: bool = True,
                 progress: Callable[[str], None] | None = None) -> dict:
        with agent_scope("report"):
            registry = Registry()
            pids = list(analyses)
            prog = progress or (lambda m: None)
            sections: "OrderedDict[str, dict]" = OrderedDict()

            plan = [
                ("Major Approaches", "Describe the main families of approaches in this literature and what each paper contributes. Group papers by approach.",
                 ["main_idea", "method"], 2),
                ("Technical Analysis", "Explain the key technical ideas, formulations and complexity claims, and how the approaches differ technically.",
                 ["method", "key_equations", "complexity"], 2),
                ("Experimental Results Across Literature", "Summarise the quantitative results, datasets and baselines used across the papers, and what they imply.",
                 ["main_results", "datasets", "baselines"], 2),
                ("Limitations", "Summarise recurring limitations acknowledged or evidenced in the papers, grouped by theme.",
                 ["limitations"], 3),
            ]
            for name, task, fields, per in plan:
                prog(f"Writing section: {name}")
                facts = self._facts(analyses, fields, per_paper=per, max_facts=36)
                sections[name] = self._write_section(name, task, facts, question, verify, registry, pids)

            # gaps are already evidence-linked and novelty-checked; they are rendered from structure, not free text
            sections["Potential Research Directions"] = self._gaps_section(gaps, registry)
            abstract = self._abstract(question, sections)

            verification = self._verification_overview(sections, analyses, paper_meta, search_stats)
            md = self._render_markdown(question, abstract, sections, comparison, search_stats, analyses, paper_meta,
                                       registry, verification, graph)
            return {"title": self._title(question), "abstract": abstract, "sections": sections,
                    "evidence_index": registry.index(), "verification": verification, "markdown": md,
                    "bibtex": make_bibtex(paper_meta, pids), "generated_at": dt.datetime.now().isoformat(timespec="seconds")}

    # ------------------------------------------------------------------ parts
    def _title(self, q: str) -> str:
        return "Literature Report: " + q.strip().rstrip("?.")

    def _abstract(self, question: str, sections: dict) -> str:
        body = "\n\n".join(f"## {n}\n{s['text']}" for n, s in sections.items() if s["text"] and not s["text"].startswith("_No"))
        if not body.strip():
            return ""
        return self.llm.complete(ABSTRACT_SYS, f"RESEARCH QUESTION: {question}\n\nSECTIONS:\n{body[:7000]}", max_tokens=350, tag="abstract").strip()

    def _gaps_section(self, gaps: dict | None, registry: "Registry") -> dict:
        if not gaps or not gaps.get("gaps"):
            note = "Our analysis did not produce a research direction that survived the novelty check."
            if gaps and gaps.get("rejected"):
                note += f" {len(gaps['rejected'])} candidate direction(s) were rejected because existing work already addresses them."
            return {"name": "Potential Research Directions", "text": note, "claims": [], "summary": None, "gaps": gaps}
        lines = []
        for i, g in enumerate(gaps["gaps"], 1):
            sup = ", ".join(sorted({s["paper"] for s in g["support"]})[:5])
            cites = "".join(f"[{registry.add_evidence_dict(s['evidence'], s['paper_id'], s['title'], s['paper'])}]" for s in g["support"][:3])
            ns = g["novelty_search"]
            closest = "; ".join(f"{c['title']} ({c['year']})" for c in ns["closest_work"]) or "none identified"
            lines.append(f"**Direction {i}: {g['direction']}**\n\n- Recurring limitation: *{g['theme']}* — {g['theme_description']} {cites}\n"
                         f"- Mentioned by {g['n_supporting_papers']}/{g['n_papers_total']} analysed papers ({sup}).\n"
                         f"- Novelty search: {ns['papers_examined']} related papers examined; closest work: {closest}.\n"
                         f"- Confidence: **{g['confidence']}**. {g['statement']}")
        return {"name": "Potential Research Directions", "text": "\n\n".join(lines), "claims": [], "summary": None, "gaps": gaps}

    def _verification_overview(self, sections: dict, analyses: dict, paper_meta: dict, search_stats) -> dict:
        tot = {"claims_checked": 0, "SUPPORTED": 0, "PARTIALLY_SUPPORTED": 0, "UNSUPPORTED": 0, "CONTRADICTED": 0,
               "removed": 0, "corrected": 0}
        for s in sections.values():
            sm = s.get("summary")
            if not sm:
                continue
            tot["claims_checked"] += sm["claims_checked"]
            for k in ("SUPPORTED", "PARTIALLY_SUPPORTED", "UNSUPPORTED", "CONTRADICTED"):
                tot[k] += sm["initial"][k]
            tot["removed"] += sm["removed"]
            tot["corrected"] += sm["corrected"]
        full = sum(1 for p in paper_meta.values() if p["status"] == "indexed")
        tot.update(papers_analyzed=len(analyses), papers_full_text=full, papers_abstract_only=len(paper_meta) - full)
        n = max(1, tot["claims_checked"])
        tot["claim_support_rate_initial"] = round((tot["SUPPORTED"] + 0.5 * tot["PARTIALLY_SUPPORTED"]) / n, 3)
        return tot

    def _render_markdown(self, question, abstract, sections, comparison, search_stats, analyses, paper_meta, registry,
                         verification, graph) -> str:
        L: list[str] = [f"# {self._title(question)}", "", f"*Generated {dt.date.today().isoformat()} by ScholarAgent*", ""]
        if abstract:
            L += ["## Abstract", abstract, ""]
        L += ["## 1. Research Question", question, ""]
        L += ["## 2. Search Methodology"]
        if search_stats:
            ss = search_stats
            L += [f"- Search date: {dt.date.today().isoformat()}",
                  f"- Databases: {', '.join(ss.get('sources', []))}",
                  f"- Queries issued: {len(ss.get('queries', []))}" + (f" ({'; '.join(ss['queries'][:8])})" if ss.get('queries') else ""),
                  f"- Candidates retrieved: {ss.get('candidates_raw', '?')}  →  after de-duplication: {ss.get('after_dedup', '?')}  →  selected: {ss.get('selected', '?')}",
                  f"- Full text obtained and analysed: {verification['papers_full_text']} papers; abstract only: {verification['papers_abstract_only']}"]
            if len(analyses) < 5:
                L += [f"- **Low coverage:** only {len(analyses)} relevant paper(s) could be analysed, so the conclusions below are tentative and may not represent the literature."]
            L += [""]
        else:
            L += [f"- Papers analysed: {len(analyses)} (user-provided collection)", ""]
        L += ["## 3. Literature Landscape"]
        if comparison and comparison.get("landscape"):
            ls = comparison["landscape"]
            tag = {r["paper_id"]: r for r in comparison["rows"]}
            for ap in ls["approaches"]:
                names = ", ".join(tag[p]["label"] for p in ap["paper_ids"] if p in tag)
                L.append(f"- **{ap['name']}** — {ap['description']} *({names})*")
            L += ["", ls["synthesis"], ""]
        n = 4
        for name in ("Major Approaches", "Technical Analysis"):
            L += [f"## {n}. {name}", sections[name]["text"], ""]
            n += 1
        L += [f"## {n}. Paper Comparison"]
        L += [comparison_table_md(comparison) if comparison else "_not available_", ""]
        n += 1
        for name in ("Experimental Results Across Literature", "Limitations", "Potential Research Directions"):
            L += [f"## {n}. {name}", sections[name]["text"], ""]
            n += 1
        L += [f"## {n}. Evidence Quality & Verification"]
        v = verification
        L += [f"- Claims checked: **{v['claims_checked']}**  |  supported: {v['SUPPORTED']}  |  partially supported: {v['PARTIALLY_SUPPORTED']}  |  "
              f"unsupported: {v['UNSUPPORTED']}  |  contradicted: {v['CONTRADICTED']}",
              f"- Self-correction: {v['corrected']} claim(s) corrected, {v['removed']} claim(s) removed from the final text.",
              f"- Papers analysed: {v['papers_analyzed']} ({v['papers_full_text']} full text, {v['papers_abstract_only']} abstract-only).",
              "- Limits: findings are restricted to the retrieved corpus; absence of evidence in this corpus is not evidence of absence in the literature.", ""]
        n += 1
        L += [f"## {n}. Conclusion", conclusion_stub(sections), ""]
        L += ["## References"]
        for pid, m in paper_meta.items():
            if pid in analyses:
                ids = [x for x in (("doi:" + m["doi"]) if m.get("doi") else "", ("arXiv:" + m["arxiv_id"]) if m.get("arxiv_id") else "") if x]
                venue = m.get("venue") if m.get("venue") not in (None, "arXiv") else ""
                L.append(f"- {', '.join(m['authors'][:6]) or 'Unknown'} ({m['year'] or 'n.d.'}). *{m['title']}*. {venue + '. ' if venue else ''}{'; '.join(ids)}")
        L += ["", "## Evidence Index", "_Numbers in square brackets in the text refer to these source passages._", ""]
        for e in registry.index():
            L.append(f"**[{e['id']}]** {e['label']} — *{e['title']}*  \n> {e['quote']}")
        return "\n".join(L)


def _clean_section(text: str, name: str) -> str:
    """Normalise model output: drop a repeated section title / markdown heading and un-flatten bullet lists."""
    t = text.strip()
    t = re.sub(r"^#+\s*" + re.escape(name) + r"\s*:?\s*", "", t, flags=re.I)
    t = re.sub(r"^#+\s*[^\n]{0,80}\n", "", t)
    t = re.sub(r"\s+(#{2,4} [A-Z])", r"\n\n\1", t)
    t = re.sub(r"(?<=[.\]:*])[ \t]+-[ \t]+(?=\*\*|[A-Z])", "\n- ", t)          # '... text. - **Item** - **Item2**' -> real bullets
    t = re.sub(r"(\*\*[^*\n]{3,80}:\*\*)[ \t]*(?=[A-Z])", r"\n\n\1 ", t)       # '**Heading:** text' starts a new paragraph
    t = re.sub(r"(?<=[.\])])[ \t]+(\*\*[^*\n]{3,80}:\*\*)", r"\n\n\1", t)                   # bold sub-heading glued behind a sentence
    t = re.sub(r"(\*\*[^*\n]{3,80}:\*\*)[ \t]*\n(?=- )", r"\1\n\n", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


class Registry:
    """Global evidence numbering: the same source chunk always gets the same [n] in the whole report."""

    def __init__(self):
        self._ids: dict[str, int] = {}
        self._items: list[dict] = []

    def gid(self, chunk_id: str) -> int | None:
        return self._ids.get(chunk_id)

    def _add(self, e: dict) -> int:
        cid = e["chunk_id"]
        if cid not in self._ids:
            self._ids[cid] = len(self._ids) + 1
            quote = e.get("text", "").split("\n(Source passage:")[0] if e.get("text") else ""
            self._items.append({"id": self._ids[cid], "chunk_id": cid, "paper_id": e["paper_id"], "title": e["title"],
                                "label": e.get("label") or "", "quote": " ".join(quote.split())[:320], "section": e.get("section"),
                                "page_start": e.get("page_start"), "page_end": e.get("page_end")})
        return self._ids[cid]

    def add_evidence_dict(self, ev: dict, paper_id: str, title: str, label: str) -> int:
        pg = ev["page_start"]
        return self._add({"chunk_id": ev["chunk_id"], "paper_id": paper_id, "title": title,
                          "label": f"{label}, {ev['section']}, p. {pg}", "text": ev["quote"], "section": ev["section"],
                          "page_start": ev["page_start"], "page_end": ev["page_end"]})

    def remap(self, text: str, pool: dict[int, dict]) -> str:
        def sub(m: re.Match) -> str:
            ids = []
            for p in re.split(r"[,;]", m.group(1)):
                p = p.strip()
                if p.isdigit() and int(p) in pool:
                    ids.append(self._add(pool[int(p)]))
            return "".join(f"[{i}]" for i in dict.fromkeys(ids))
        return re.sub(r"\[(\d+(?:\s*[,;]\s*\d+)*)\]", sub, text)

    def index(self) -> list[dict]:
        return self._items


def comparison_table_md(comp: dict) -> str:
    cols = comp["columns"]
    head = "| Paper | " + " | ".join(c.replace("_", " ").title() for c in cols) + " |"
    sep = "|" + "---|" * (len(cols) + 1)
    rows = [f"| **{r['tag']}** {r['label']} | " + " | ".join(r["cells"][c]["text"].replace("|", "/") for c in cols) + " |" for r in comp["rows"]]
    return "\n".join([head, sep, *rows])


def conclusion_stub(sections: dict) -> str:
    return ("This report synthesised the retrieved literature with every statement linked to a source passage. "
            "See *Potential Research Directions* for recurring, novelty-checked open problems.")


def make_bibtex(paper_meta: dict[str, dict], pids: list[str]) -> str:
    out = []
    for pid in pids:
        m = paper_meta.get(pid)
        if not m:
            continue
        key = slugify((m["authors"][0].split()[-1] if m["authors"] else "anon") + str(m["year"] or ""), 30) + "_" + pid[-6:]
        fields = [f"  title = {{{m['title']}}}", f"  author = {{{' and '.join(m['authors'])}}}", f"  year = {{{m['year'] or ''}}}"]
        if m.get("venue"):
            fields.append(f"  journal = {{{m['venue']}}}")
        if m.get("doi"):
            fields.append(f"  doi = {{{m['doi']}}}")
        if m.get("arxiv_id"):
            fields += [f"  eprint = {{{m['arxiv_id']}}}", "  archivePrefix = {arXiv}"]
        out.append("@article{" + key + ",\n" + ",\n".join(fields) + "\n}")
    return "\n\n".join(out)
