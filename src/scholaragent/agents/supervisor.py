"""Supervisor Agent (LangGraph).

The supervisor does NOT research. It plans, then repeatedly decides which specialist to call next
(search / ingest / analyze / compare / build_graph / find_gaps / answer / report) from what the state still lacks,
and it may decide to search again when too few papers were found. Every decision is recorded.
"""
from __future__ import annotations

import time
import traceback
from typing import Any, Callable, Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from ..config import settings
from ..llm.client import meter
from ..utils.logging import agent_scope, get_logger, set_run, trace
from .services import Services

log = get_logger("supervisor")

ACTIONS = ["search", "ingest", "analyze", "compare", "build_graph", "find_gaps", "answer", "report", "finish"]
MAX_STEPS = 24
MAX_SEARCH_ROUNDS = 3


class State(TypedDict, total=False):
    run_id: str
    question: str
    mode: str                      # deep | my_papers
    params: dict
    plan: dict
    intent: str
    queries: list[str]
    used_queries: list[str]
    candidates: list[dict]
    selected: list[dict]
    pending_records: list[dict]
    search_stats: dict
    search_rounds: int
    paper_ids: list[str]
    ingest_report: dict
    analyses: dict
    comparison: dict
    graph: dict
    gaps: dict
    answer: dict
    report: dict
    done: list[str]
    decisions: list[dict]
    steps: int
    next: str
    errors: list[str]
    timings: dict


class Plan(BaseModel):
    intent: Literal["survey", "comparison", "qa", "gaps"] = Field(
        description="survey = broad literature review; comparison = compare specific methods/papers; qa = a specific factual question; gaps = limitations and open directions")
    needs_literature_search: bool
    coherent_research_request: bool = Field(description="false if the request is not a meaningful, answerable scientific research question (gibberish, mixed-up unrelated concepts, not about science/engineering/literature)")
    sub_questions: list[str] = Field(description="1-4 sub-questions that together answer the request")
    focus_terms: list[str] = Field(description="key technical terms/methods named in the request")


class Decision(BaseModel):
    next: str
    reason: str


PLAN_SYS = """You are the planning module of a scientific research agent. Classify the user's request into exactly one intent.

- survey: asks what is known / what methods or approaches exist in a research area, for an overview or state of the art.
  e.g. "What are recent methods for protein structure prediction?", "Give an overview of battery degradation modelling", "How is reinforcement learning used in robotics?"
- comparison: explicitly asks to compare / contrast / differentiate TWO OR MORE named methods, papers or settings.
  e.g. "Compare CRISPR-Cas9 with base editing", "LSTM versus Transformer for forecasting", "What is the difference between method A and method B?"
- qa: a narrow factual or explanatory question about one specific thing that a few passages can answer.
  e.g. "What is a Kalman filter?", "Which optimizer does the ResNet paper use?", "How many layers does the encoder have?"
- gaps: asks specifically about limitations, challenges, open problems or promising future research directions.
  e.g. "What are the limitations of current quantum error correction and what directions remain?", "What open problems exist in federated learning?"

A request that merely contains the word 'methods' or 'approaches' is a survey, NOT a comparison.
Set coherent_research_request=false for gibberish or nonsensical combinations of unrelated concepts (e.g. "emotional quarterly revenue of rotating sandwiches"), even if some words are scientific. needs_literature_search is false when the user already supplied their own papers."""

DECIDE_SYS = """You are the supervisor of a research agent team. Choose the SINGLE next action from the allowed list.
Actions: search (find papers in academic databases), ingest (download+parse+index selected papers), analyze (structured per-paper analysis),
compare (comparison table and landscape), build_graph (citation/knowledge graph), find_gaps (limitations -> candidate research directions with novelty check),
answer (grounded, verified answer to the question), report (final evidence-grounded report), finish.
Prefer the natural order for the intent, but choose 'search' again if too few relevant papers were indexed. Reply with next + a short reason."""


def default_policy(s: State, allowed: list[str]) -> str:
    intent = s.get("intent", "survey")
    order = {"survey": ["search", "ingest", "analyze", "compare", "build_graph", "find_gaps", "report", "finish"],
             "gaps": ["search", "ingest", "analyze", "compare", "build_graph", "find_gaps", "report", "finish"],
             "comparison": ["search", "ingest", "analyze", "compare", "answer", "finish"],
             "qa": ["search", "ingest", "answer", "finish"]}[intent]
    for a in order:
        if a in allowed:
            return a
    return "finish"


def allowed_actions(s: State) -> list[str]:
    if s.get("plan", {}).get("coherent_research_request") is False:
        return ["finish"]                         # nothing meaningful to research - decline instead of fabricating a report
    done = set(s.get("done", []))
    mode, intent = s.get("mode", "deep"), s.get("intent", "survey")
    n_idx = len(s.get("paper_ids", []))
    target = s.get("params", {}).get("max_papers", 15)
    out: list[str] = []
    pending = bool(s.get("pending_records"))
    if mode == "deep" and s.get("plan", {}).get("needs_literature_search", True) and not pending:
        first = "search" not in done
        too_few = n_idx < max(3, int(0.6 * target)) and "ingest" in done and s.get("search_rounds", 0) < MAX_SEARCH_ROUNDS
        if first or too_few:
            out.append("search")
    if pending:
        out.append("ingest")
    if n_idx and not pending:
        if intent != "qa" and "analyze" not in done:
            out.append("analyze")
        if intent != "qa" and "analyze" in done and "compare" not in done:
            out.append("compare")
        if intent in ("survey", "gaps") and "analyze" in done and "build_graph" not in done:
            out.append("build_graph")
        if intent in ("survey", "gaps") and "analyze" in done and "find_gaps" not in done:
            out.append("find_gaps")
        if intent in ("comparison", "qa") and "answer" not in done and (intent == "qa" or "compare" in done):
            out.append("answer")
        if intent in ("survey", "gaps") and "analyze" in done and "report" not in done:
            out.append("report")
    # a deep run that has nothing indexed yet must search/ingest first
    if (("report" in done) or ("answer" in done) or not out) and "finish" not in out:
        out.append("finish")
    if not n_idx and "search" not in out and "ingest" not in out:
        out = ["finish"]
    return list(dict.fromkeys(out))


class Supervisor:
    def __init__(self, svc: Services):
        self.svc = svc
        self.graph = self._build()

    # ------------------------------------------------------------ plumbing
    @staticmethod
    def emit(config, stage: str, message: str, **data) -> None:
        cb: Callable | None = config["configurable"].get("emit")
        if cb:
            cb({"stage": stage, "message": message, "ts": time.strftime("%H:%M:%S"), **data})

    def _build(self):
        g = StateGraph(State)
        g.add_node("plan", self.n_plan)
        g.add_node("supervisor", self.n_supervisor)
        for name in ACTIONS[:-1]:
            g.add_node(name, getattr(self, f"n_{name}"))
        g.add_edge(START, "plan")
        g.add_edge("plan", "supervisor")
        g.add_conditional_edges("supervisor", lambda s: s["next"], {a: (END if a == "finish" else a) for a in ACTIONS})
        for name in ACTIONS[:-1]:
            g.add_edge(name, "supervisor")
        return g.compile()

    def run(self, question: str, *, mode: str = "deep", params: dict | None = None, paper_ids: list[str] | None = None,
            run_id: str | None = None, emit: Callable | None = None) -> State:
        params = {"max_papers": 15, "year_from": None, "year_to": None, "sources": ["arxiv", "semantic_scholar", "crossref"],
                  "verify": True, **(params or {})}
        init: State = {"run_id": run_id or "adhoc", "question": question, "mode": mode, "params": params,
                       "paper_ids": list(paper_ids or []), "done": [], "decisions": [], "steps": 0, "errors": [],
                       "analyses": {}, "search_rounds": 0, "used_queries": [], "timings": {}, "pending_records": []}
        set_run(init["run_id"])
        t0 = time.perf_counter()
        final = self.graph.invoke(init, config={"configurable": {"emit": emit}, "recursion_limit": 100})
        final["timings"]["total_s"] = round(time.perf_counter() - t0, 1)
        return final

    def _timed(self, s: State, name: str, fn):
        t0 = time.perf_counter()
        try:
            return fn()
        finally:
            tm = dict(s.get("timings", {}))
            tm[name] = round(tm.get(name, 0) + time.perf_counter() - t0, 1)
            s["timings"] = tm

    # --------------------------------------------------------------- nodes
    def n_plan(self, s: State, config) -> dict:
        with agent_scope("supervisor"):
            self.emit(config, "plan", "Understanding the research intent")
            q, mode = s["question"], s["mode"]
            user = f"REQUEST: {q}\nUSER SUPPLIED OWN PAPERS: {mode == 'my_papers'} ({len(s.get('paper_ids', []))} papers)"
            plan = self.svc.llm.generate_or_none(Plan, PLAN_SYS, user, max_tokens=400, tag="plan")
            if plan is None:
                plan = Plan(intent="qa" if mode == "my_papers" else "survey", needs_literature_search=mode == "deep",
                            coherent_research_request=True, sub_questions=[q], focus_terms=[])
            if mode == "my_papers":
                plan.needs_literature_search = False
            if s["params"].get("intent"):
                plan.intent = s["params"]["intent"]
            self.emit(config, "plan", f"Intent: {plan.intent}; literature search: {plan.needs_literature_search}", plan=plan.model_dump())
            return {"plan": plan.model_dump(), "intent": plan.intent}

    def n_supervisor(self, s: State, config) -> dict:
        with agent_scope("supervisor"):
            steps = s.get("steps", 0) + 1
            allowed = allowed_actions(s)
            fallback = default_policy(s, allowed)
            choice, reason = fallback, "default policy"
            if steps >= MAX_STEPS:
                choice, reason = "finish", "step budget exhausted"
            elif len(allowed) > 1:
                hist = ", ".join(d["action"] for d in s.get("decisions", [])) or "(nothing yet)"
                user = (f"REQUEST: {s['question']}\nINTENT: {s['intent']}\nDONE SO FAR: {hist}\n"
                        f"Indexed papers: {len(s.get('paper_ids', []))} (target {s['params'].get('max_papers')}); search rounds: {s.get('search_rounds', 0)}\n"
                        f"ALLOWED ACTIONS: {allowed}\nSuggested by the default workflow: {fallback}")
                d = self.svc.llm.generate_or_none(Decision, DECIDE_SYS, user, max_tokens=120, tag="decide")
                if d and d.next in allowed:
                    choice, reason = d.next, d.reason
            decisions = [*s.get("decisions", []), {"step": steps, "action": choice, "reason": reason, "allowed": allowed}]
            trace("supervisor_decision", step=steps, action=choice, allowed=allowed)
            self.emit(config, "decision", f"Supervisor → {choice}: {reason}", action=choice)
            return {"next": choice, "steps": steps, "decisions": decisions}

    def _mark(self, s: State, name: str) -> list[str]:
        return [*s.get("done", []), name]

    # ---- search
    def n_search(self, s: State, config) -> dict:
        p = s["params"]
        try:
            with agent_scope("search_agent"):
                rnd = s.get("search_rounds", 0)
                self.emit(config, "search", "Searching academic databases" + (f" (round {rnd + 1})" if rnd else ""))
                qs = self.svc.search_agent.plan_queries(s["question"], avoid=s.get("used_queries", []),
                                                        hint="Focus: " + ", ".join(s["plan"].get("focus_terms", [])))
                recs, stats = self.svc.search_agent.search(s["question"], qs, year_from=p.get("year_from"), year_to=p.get("year_to"),
                                                           sources=p.get("sources"), per_source=max(15, p["max_papers"]))
                already = set(s.get("paper_ids", [])) | {c["paper_id"] for c in s.get("candidates", []) if c.get("_ingested")}
                recs = [r for r in recs if r.paper_id not in already]
                # relevance cut: keep papers with a reasonable score
                prelim = [r for r in recs if r.score >= 0.2] or recs
                keep = self.svc.search_agent.filter_relevant(s["question"], prelim, p["max_papers"])[: int(p["max_papers"] * 2.0)]
                self.emit(config, "search", f"Found {stats['candidates_raw']} candidates → {stats['after_dedup']} unique → {len(keep)} relevant",
                          stats=stats)
                merged = dict(s.get("search_stats") or {})
                merged_q = [*merged.get("queries", []), *qs]
                prev_raw, prev_dedup = merged.get("candidates_raw", 0), merged.get("after_dedup", 0)
                merged.update(queries=merged_q, sources=stats["sources"], candidates_raw=prev_raw + stats["candidates_raw"],
                              after_dedup=prev_dedup + stats["after_dedup"], per_source=stats["per_source"])
                return {"pending_records": [r.to_dict() for r in keep], "candidates": [*s.get("candidates", []), *[r.to_dict() for r in recs]],
                        "search_stats": merged, "used_queries": [*s.get("used_queries", []), *qs], "queries": qs,
                        "search_rounds": s.get("search_rounds", 0) + 1, "done": self._mark(s, "search")}
        except Exception as e:  # noqa: BLE001
            log.error("search failed: %s", traceback.format_exc())
            return {"errors": [*s.get("errors", []), f"search: {e!r}"], "done": self._mark(s, "search"), "search_rounds": s.get("search_rounds", 0) + 1}

    # ---- ingest
    def n_ingest(self, s: State, config) -> dict:
        from ..search.models import PaperRecord
        p = s["params"]
        lib = self.svc.lib
        recs = [PaperRecord.from_dict(d) for d in s.get("pending_records", [])]
        have = list(s.get("paper_ids", []))
        want = max(0, p["max_papers"] - len(have))
        report = {"ok": [], "cached": 0, "failed": [], "abstract_only": []}
        with agent_scope("ingest"):
            cid = s["params"].get("collection_id") or lib.store.create_collection(s["question"][:80], "research", collection_id=f"run_{s['run_id']}")
            if recs:
                self.emit(config, "ingest", f"Downloading full text for up to {want} papers")
                ok, failed = self.svc.search_agent.fetch_fulltext(recs, want)
                for i, (r, path) in enumerate(ok, 1):
                    res = lib.ingest_pdf(path, overrides={"citation_count": r.citation_count, "venue": r.venue, "source": ",".join(r.sources)},
                                         collection_id=cid)
                    if res["ok"]:
                        report["ok"].append(res["paper_id"])
                        report["cached"] += int(res.get("cached", False))
                        have.append(res["paper_id"])
                        lib.store.add_to_collection(cid, [res["paper_id"]], {res["paper_id"]: r.score})
                    else:
                        report["failed"].append({"title": r.title, "error": res.get("error")})
                        failed.insert(0, r)
                    self.emit(config, "ingest", f"Indexed {i}/{len(ok)}: {r.title[:70]}", progress=[i, len(ok)])
                # abstract-only fallback so a paper with unavailable full text is still represented
                room = max(0, p["max_papers"] - len(have))
                for r in [x for x in failed if x.abstract and len(x.abstract) > 300][: min(room, max(0, int(0.25 * p["max_papers"])))]:
                    lib.add_metadata_only(r.to_dict())
                    lib.store.add_to_collection(cid, [r.paper_id], {r.paper_id: r.score})
                    have.append(r.paper_id)
                    report["abstract_only"].append(r.paper_id)
        stats = dict(s.get("search_stats") or {})
        stats["selected"] = len(have)
        self.emit(config, "ingest", f"{len(have)} papers ready ({len(report['abstract_only'])} abstract-only, {len(report['failed'])} failed)")
        return {"paper_ids": list(dict.fromkeys(have)), "pending_records": [], "ingest_report": report, "search_stats": stats,
                "done": self._mark(s, "ingest")}

    # ---- analyze
    def n_analyze(self, s: State, config) -> dict:
        pids = s["paper_ids"]
        self.emit(config, "analyze", f"Analysing {len(pids)} papers", progress=[0, len(pids)])
        results: dict[str, dict] = dict(s.get("analyses", {}))
        done = [0]

        def work(pid):
            a = self.svc.analyst.analyze(pid)
            done[0] += 1
            self.emit(config, "analyze", f"Analysed {done[0]}/{len(pids)}", progress=[done[0], len(pids)])
            return pid, a
        for r in self.svc.llm.map(work, pids, workers=4):
            if isinstance(r, tuple) and r[1]:
                results[r[0]] = r[1]
        failed = [p for p in pids if p not in results]
        errs = [*s.get("errors", []), *[f"analysis failed: {p}" for p in failed]]
        return {"analyses": results, "errors": errs, "done": self._mark(s, "analyze")}

    # ---- compare
    def n_compare(self, s: State, config) -> dict:
        self.emit(config, "compare", "Comparing papers")
        comp = self.svc.comparison.compare(list(s["analyses"].values()), s["question"])
        return {"comparison": comp, "done": self._mark(s, "compare")}

    # ---- graph
    def n_build_graph(self, s: State, config) -> dict:
        self.emit(config, "graph", "Building citation graph")
        try:
            g = self.svc.graph.build(s["paper_ids"], s.get("comparison"), s["analyses"])
        except Exception as e:  # noqa: BLE001
            log.warning("graph failed: %r", e)
            g = {"nodes": [], "links": [], "stats": {}}
        return {"graph": g, "done": self._mark(s, "build_graph")}

    # ---- gaps
    def n_find_gaps(self, s: State, config) -> dict:
        self.emit(config, "gaps", "Extracting limitations and testing candidate research directions against the literature")
        corpus = {p["paper_id"]: p for p in self.svc.lib.store.list_papers(s["paper_ids"])}
        try:
            gaps = self.svc.gap.find_gaps(list(s["analyses"].values()), corpus, s["question"], year_from=s["params"].get("year_from"),
                                          progress=lambda m: self.emit(config, "gaps", m))
        except Exception as e:  # noqa: BLE001
            log.error("gap agent failed: %s", traceback.format_exc())
            gaps = {"gaps": [], "rejected": [], "stats": {}, "error": repr(e)}
        return {"gaps": gaps, "done": self._mark(s, "find_gaps")}

    # ---- answer (+verification)
    def n_answer(self, s: State, config) -> dict:
        self.emit(config, "answer", "Retrieving evidence and answering")
        ans, timing = self.svc.rag.ask(s["question"], s["paper_ids"], k=10, mode="hybrid_rerank", rewrite=False, max_per_paper=3, decompose=True)
        out = {"draft": ans.to_dict(), "timing": timing}
        if s["params"].get("verify", True):
            self.emit(config, "verify", "Verifying every claim against the sources")
            rep = self.svc.verifier.verify_answer(ans, s["paper_ids"])
            out["verification"] = rep
            out["final_answer"] = rep["final_answer"]
            out["evidence"] = rep["evidence"]
        else:
            out["final_answer"], out["evidence"] = ans.text, [e.to_dict() for e in ans.evidence]
        return {"answer": out, "done": self._mark(s, "answer")}

    # ---- report
    def n_report(self, s: State, config) -> dict:
        self.emit(config, "report", "Writing and verifying the research report")
        meta = {p["paper_id"]: p for p in self.svc.lib.store.list_papers(s["paper_ids"])}
        rep = self.svc.report.generate(question=s["question"], analyses=s["analyses"], comparison=s.get("comparison"),
                                       gaps=s.get("gaps"), search_stats=s.get("search_stats") if s["mode"] == "deep" else None,
                                       graph=s.get("graph"), paper_meta=meta, verify=s["params"].get("verify", True),
                                       progress=lambda m: self.emit(config, "report", m))
        return {"report": rep, "done": self._mark(s, "report")}


def summarize_state(s: State, svc: Services) -> dict:
    """JSON-serialisable result of a run for the API/UI."""
    meta = {p["paper_id"]: p for p in svc.lib.store.list_papers(s.get("paper_ids", []))}
    return {
        "question": s["question"], "mode": s["mode"], "intent": s.get("intent"), "plan": s.get("plan"),
        "params": s["params"], "search_stats": s.get("search_stats"), "ingest_report": s.get("ingest_report"),
        "papers": list(meta.values()), "analyses": s.get("analyses"), "comparison": s.get("comparison"),
        "graph": s.get("graph"), "gaps": s.get("gaps"), "answer": s.get("answer"), "report": s.get("report"),
        "decisions": s.get("decisions"), "errors": s.get("errors"), "timings": s.get("timings"), "usage": meter.snapshot(),
        "note": ("The request does not look like a meaningful scientific research question, so no search or report was produced. "
                 "Please rephrase it in terms of a research topic, method or problem.")
        if (s.get("plan") or {}).get("coherent_research_request") is False else None,
    }
