"""ScholarAgent - Streamlit UI (talks to the FastAPI backend)."""
import os
import time

import networkx as nx
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

API = os.getenv("API_URL", "http://127.0.0.1:8100")
st.set_page_config(page_title="ScholarAgent", page_icon="🔬", layout="wide")

STAGES = [("plan", "Understanding the research intent"), ("search", "Searching literature"), ("ingest", "Downloading & indexing papers"),
          ("analyze", "Analysing papers"), ("compare", "Comparing papers"), ("graph", "Building citation graph"),
          ("gaps", "Finding research directions"), ("answer", "Answering"), ("verify", "Verifying claims"), ("report", "Writing report")]


# ----------------------------------------------------------------------------- api helpers
def api(method, path, **kw):
    try:
        r = requests.request(method, f"{API}{path}", timeout=kw.pop("timeout", 600), **kw)
        r.raise_for_status()
        ct = r.headers.get("content-type", "")
        return r.json() if "json" in ct else r.text
    except requests.HTTPError as e:
        st.error(f"API error: {e.response.status_code} {e.response.text[:300]}")
    except Exception as e:  # noqa: BLE001
        st.error(f"Cannot reach the API at {API}: {e}")
    return None


@st.cache_data(ttl=10)
def health():
    try:
        return requests.get(f"{API}/health", timeout=20).json()
    except Exception:  # noqa: BLE001
        return None


def evidence_panel(evidence, cited=None, key="ev"):
    st.markdown("##### Evidence")
    for e in evidence:
        if cited and e["id"] not in cited:
            continue
        with st.expander(f"[{e['id']}] {e.get('label') or e['title']}"):
            st.caption(f"**{e['title']}**  ·  {e['section']}  ·  pages {e['page_start']}–{e['page_end']}")
            st.write(e["text"])


def show_answer(res, key):
    st.markdown("#### Answer")
    ans = res.get("final_answer") or res.get("answer")
    if res.get("insufficient_evidence"):
        st.warning("The retrieved evidence is insufficient to answer fully.")
    st.markdown(ans)
    ver = res.get("verification")
    if ver:
        s = ver["summary"]
        c = st.columns(5)
        c[0].metric("Claims checked", s["claims_checked"])
        c[1].metric("Supported", s["initial"]["SUPPORTED"])
        c[2].metric("Partial", s["initial"]["PARTIALLY_SUPPORTED"])
        c[3].metric("Unsupported", s["initial"]["UNSUPPORTED"] + s["initial"]["CONTRADICTED"])
        c[4].metric("Removed / corrected", f"{s['removed']} / {s['corrected']}")
        with st.expander("Claim–evidence matrix"):
            rows = [{"Claim": c["claim"], "Initial": c["initial_status"], "Final": c["final_status"], "Action": c["action"],
                     "Evidence": "; ".join(f"[{e['id']}] {e['label']}" for e in c["evidence"]) or "—"} for c in ver["claims"]]
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        evidence_panel(ver["evidence"], set(ver.get("cited_ids") or []) or None, key)
    elif res.get("evidence"):
        evidence_panel(res["evidence"], set(res.get("cited_ids") or []) or None, key)
    if res.get("timing"):
        st.caption(f"retrieval {res['timing']['retrieval_s']}s · generation {res['timing']['generation_s']}s")


# ----------------------------------------------------------------------------- results dashboard
def graph_figure(g):
    G = nx.DiGraph()
    for n in g["nodes"]:
        G.add_node(n["id"], **n)
    for l in g["links"]:
        G.add_edge(l["source"], l["target"], relation=l["relation"])
    pos = nx.spring_layout(G, seed=7, k=1.2 / max(1, len(G) ** 0.5))
    colors = {"Paper": "#2b6cb0", "Approach": "#d69e2e", "Dataset": "#38a169"}
    fig = go.Figure()
    for rel, col in (("CITES", "#718096"), ("PROPOSES", "#d69e2e"), ("EVALUATES_ON", "#38a169")):
        xs, ys = [], []
        for u, v, d in G.edges(data=True):
            if d["relation"] == rel:
                xs += [pos[u][0], pos[v][0], None]
                ys += [pos[u][1], pos[v][1], None]
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(width=1 if rel != "CITES" else 1.6, color=col), hoverinfo="none", name=rel))
    for t, col in colors.items():
        ns = [n for n, d in G.nodes(data=True) if d.get("type") == t]
        if ns:
            fig.add_trace(go.Scatter(x=[pos[n][0] for n in ns], y=[pos[n][1] for n in ns], mode="markers",
                                     marker=dict(size=[10 + 4 * G.in_degree(n) for n in ns], color=col, line=dict(width=1, color="white")),
                                     text=[(G.nodes[n].get("label") or n)[:90] + (f" ({G.nodes[n].get('year')})" if G.nodes[n].get("year") else "") for n in ns],
                                     hoverinfo="text", name=t))
    fig.update_layout(height=620, showlegend=True, margin=dict(l=0, r=0, t=10, b=0), xaxis=dict(visible=False), yaxis=dict(visible=False))
    return fig


def render_results(run_id, res):
    papers = {p["paper_id"]: p for p in res.get("papers", [])}
    tabs = st.tabs(["Overview", "Papers", "Methods", "Comparison", "Citation graph", "Research directions", "Ask", "Report", "Agent trace"])
    ss = res.get("search_stats") or {}
    with tabs[0]:
        c = st.columns(5)
        c[0].metric("Candidates found", ss.get("candidates_raw", "—"))
        c[1].metric("After de-duplication", ss.get("after_dedup", "—"))
        c[2].metric("Papers analysed", len(res.get("analyses") or {}))
        rep = res.get("report") or {}
        v = rep.get("verification") or {}
        c[3].metric("Claims verified", v.get("claims_checked", (res.get("answer") or {}).get("verification", {}).get("summary", {}).get("claims_checked", "—")))
        c[4].metric("Runtime (s)", (res.get("timings") or {}).get("total_s", "—"))
        if res.get("answer"):
            show_answer(res["answer"], "overview")
        if rep.get("abstract"):
            st.markdown("#### Summary"); st.write(rep["abstract"])
        if v:
            st.info(f"Verification — checked **{v['claims_checked']}** claims: {v['SUPPORTED']} supported, {v['PARTIALLY_SUPPORTED']} partially supported, "
                    f"{v['UNSUPPORTED']} unsupported, {v['CONTRADICTED']} contradicted · {v['removed']} removed · {v['corrected']} corrected.")
        if res.get("errors"):
            with st.expander(f"{len(res['errors'])} non-fatal problems"):
                st.write(res["errors"])
    with tabs[1]:
        an = res.get("analyses") or {}
        for pid, a in an.items():
            m = papers.get(pid, {})
            with st.container(border=True):
                st.markdown(f"**{a['title']}**  \n{', '.join((a.get('authors') or [])[:4])} — {a.get('year')}  ·  {a.get('venue') or ''}  ·  citations: {m.get('citation_count', '—')}")
                f = a["fields"]
                for label, key in (("Main idea", "main_idea"), ("Key result", "main_results"), ("Limitation", "limitations")):
                    items = f.get(key) or []
                    if items:
                        it = items[0]
                        st.markdown(f"**{label}:** {it['text']}  \n<small>_{it['evidence']['section']}, p. {it['evidence']['page_start']}_ — “{it['evidence']['quote']}”</small>", unsafe_allow_html=True)
                with st.expander("Full structured analysis"):
                    for k, items in f.items():
                        st.markdown(f"**{k.replace('_', ' ').title()}**")
                        for it in items:
                            st.markdown(f"- {it['text']} <small>({it['evidence']['section']}, p. {it['evidence']['page_start']})</small>", unsafe_allow_html=True)
    with tabs[2]:
        comp = res.get("comparison") or {}
        ls = comp.get("landscape")
        if ls:
            st.markdown("#### Major approaches")
            tag = {r["paper_id"]: r for r in comp["rows"]}
            for ap in ls["approaches"]:
                st.markdown(f"**{ap['name']}** — {ap['description']}")
                st.caption(", ".join(tag[p]["label"] for p in ap["paper_ids"] if p in tag))
            st.markdown("#### Synthesis"); st.write(ls["synthesis"])
        else:
            st.info("No landscape available for this run.")
    with tabs[3]:
        comp = res.get("comparison")
        if comp:
            df = pd.DataFrame([{"Paper": f"{r['tag']} · {r['label']}", "Approach": r["category"], **{c.replace('_', ' ').title(): r["cells"][c]["text"] for c in comp["columns"]}} for r in comp["rows"]])
            st.dataframe(df, width="stretch", hide_index=True, height=520)
            st.download_button("Download CSV", df.to_csv(index=False), "comparison.csv")
            with st.expander("Cell evidence"):
                for r in comp["rows"]:
                    st.markdown(f"**{r['tag']} {r['label']}**")
                    for c in comp["columns"]:
                        for e in r["cells"][c]["evidence"][:1]:
                            st.markdown(f"- *{c}* → {e['section']}, p. {e['page_start']}: “{e['quote']}”")
        else:
            st.info("No comparison for this run.")
    with tabs[4]:
        g = res.get("graph")
        if g and g.get("nodes"):
            st.plotly_chart(graph_figure(g), width="stretch")
            st.json(g.get("stats", {}), expanded=False)
            st.download_button("Neo4j Cypher", requests.get(f"{API}/runs/{run_id}/graph?fmt=cypher").text, "graph.cypher")
        else:
            st.info("No graph for this run.")
    with tabs[5]:
        gaps = res.get("gaps") or {}
        if not gaps.get("gaps"):
            st.info("No research direction survived the novelty check." if gaps else "Gap analysis was not part of this run.")
        for i, gp in enumerate(gaps.get("gaps", []), 1):
            with st.container(border=True):
                st.markdown(f"### Direction {i}: {gp['direction']}")
                st.markdown(f"**Recurring limitation:** {gp['theme']} — {gp['theme_description']}")
                st.markdown(f"**Support:** {gp['n_supporting_papers']}/{gp['n_papers_total']} papers  ·  **Confidence:** `{gp['confidence']}`")
                ns = gp["novelty_search"]
                st.markdown(f"**Novelty search:** {ns['papers_examined']} papers examined; closest work: " + ("; ".join(f"{c['title']} ({c['year']})" for c in ns["closest_work"]) or "none identified"))
                st.caption(gp["statement"])
                with st.expander("Evidence"):
                    for s in gp["support"]:
                        st.markdown(f"- **{s['paper']}** — {s['text']} <small>({s['evidence']['section']}, p. {s['evidence']['page_start']})</small>", unsafe_allow_html=True)
        if gaps.get("rejected"):
            with st.expander(f"{len(gaps['rejected'])} candidate directions rejected (already addressed in the literature)"):
                for r in gaps["rejected"]:
                    st.markdown(f"- ~~{r.get('direction', r.get('theme'))}~~ — {r.get('reason', '')}")
    with tabs[6]:
        q = st.text_input("Ask a question about the papers in this run", key=f"ask_{run_id}")
        c1, c2 = st.columns(2)
        verify = c1.checkbox("Verify claims", True, key=f"v_{run_id}")
        rewrite = c2.checkbox("Rewrite query", False, key=f"r_{run_id}")
        if st.button("Ask", key=f"askb_{run_id}") and q:
            with st.spinner("Retrieving evidence, answering and verifying…"):
                out = api("POST", "/ask", json={"question": q, "paper_ids": list(papers), "verify": verify, "rewrite": rewrite, "k": 8})
            if out:
                show_answer(out, "ask")
    with tabs[7]:
        rep = res.get("report")
        if rep:
            c = st.columns(5)
            for col, (fmt, lab) in zip(c, [("md", "Markdown"), ("html", "HTML"), ("tex", "LaTeX"), ("bib", "BibTeX"), ("json", "JSON")]):
                col.download_button(lab, requests.get(f"{API}/runs/{run_id}/report.{fmt}").text, f"report.{fmt}")
            st.markdown(rep["markdown"])
        else:
            st.info("This run did not generate a full report.")
    with tabs[8]:
        st.markdown("#### Supervisor decisions")
        st.dataframe(pd.DataFrame([{"step": d["step"], "action": d["action"], "reason": d["reason"]} for d in res.get("decisions", [])]), hide_index=True, width="stretch")
        st.markdown("#### Usage"); st.json(res.get("usage", {}), expanded=False)
        st.json(res.get("timings", {}), expanded=False)


def follow_run(run_id):
    box = st.container()
    bar = box.progress(0.0)
    status = box.empty()
    log = box.empty()
    while True:
        r = api("GET", f"/runs/{run_id}")
        if not r:
            return None
        evs = r["events"]
        stages_seen = {e["stage"] for e in evs}
        done = sum(1 for k, _ in STAGES if k in stages_seen)
        bar.progress(min(1.0, done / len(STAGES)))
        last = evs[-1]["message"] if evs else "Queued…"
        status.info(f"**{r['status'].upper()}** — {last}")
        lines = []
        for k, label in STAGES:
            if k in stages_seen:
                e = [x for x in evs if x["stage"] == k][-1]
                prog = e.get("progress")
                lines.append(f"✓ {label}" + (f"  {prog[0]}/{prog[1]}" if prog else ""))
        log.code("\n".join(lines) or "starting…")
        if r["status"] in ("done", "failed"):
            bar.progress(1.0)
            if r["status"] == "failed":
                st.error(r.get("error"))
                return None
            return api("GET", f"/runs/{run_id}?with_result=true")["result"]
        time.sleep(2)


# ----------------------------------------------------------------------------- pages
def page_research():
    st.header("Deep Research")
    st.caption("Give a research question — ScholarAgent finds the papers, reads them, compares them, and writes an evidence-grounded report.")
    q = st.text_area("Research question", "What are recent methods for using machine learning to accelerate numerical linear algebra solvers?", height=80)
    c1, c2, c3 = st.columns([2, 1, 2])
    years = c1.slider("Research period", 2010, 2026, (2020, 2026))
    n = c2.number_input("Maximum papers", 3, 40, 12)
    sources = c3.multiselect("Sources", ["arxiv", "semantic_scholar", "crossref", "openalex"], default=["arxiv", "semantic_scholar", "crossref"])
    verify = st.checkbox("Verify every claim against the sources (recommended)", True)
    if st.button("Start research", type="primary"):
        r = api("POST", "/research", json={"question": q, "mode": "deep", "max_papers": int(n), "year_from": years[0], "year_to": years[1], "sources": sources, "verify": verify})
        if r:
            st.session_state.run_id = r["run_id"]
            st.session_state.pop("result", None)
    rid = st.session_state.get("run_id")
    if rid:
        if "result" not in st.session_state or st.session_state.get("result_run") != rid:
            res = follow_run(rid)
            if res:
                st.session_state.result, st.session_state.result_run = res, rid
                st.rerun()
        else:
            render_results(rid, st.session_state.result)


def page_my_papers():
    st.header("My Papers")
    up = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
    if up and st.button("Index uploaded papers"):
        with st.spinner("Parsing, chunking and embedding…"):
            out = api("POST", "/papers/upload", files=[("files", (f.name, f.getvalue(), "application/pdf")) for f in up], data={"collection_id": "my_papers"})
        if out:
            for r in out["results"]:
                (st.success if r["ok"] else st.error)(f"{r.get('title')}: " + (f"{r.get('n_chunks')} chunks" if r["ok"] else r.get("error", "")))
    papers = api("GET", "/papers", params={"collection_id": "my_papers"}) or []
    if not papers:
        st.info("No papers yet — upload some PDFs.")
        return
    df = pd.DataFrame([{"Title": p["title"], "Year": p["year"], "Pages": p["n_pages"], "Chunks": p["n_chunks"], "id": p["paper_id"]} for p in papers])
    sel = st.multiselect("Papers to use", df["id"], default=list(df["id"]), format_func=lambda i: df.set_index("id").loc[i, "Title"][:90])
    st.dataframe(df.drop(columns="id"), hide_index=True, width="stretch")
    t1, t2 = st.tabs(["Ask", "Analyse & compare"])
    with t1:
        q = st.text_input("Question", placeholder="Which papers use low-rank approximations?")
        c = st.columns(4)
        mode = c[0].selectbox("Retrieval", ["hybrid_rerank", "hybrid", "dense", "bm25"])
        k = c[1].slider("Passages", 3, 15, 8)
        rewrite = c[2].checkbox("Rewrite query")
        verify = c[3].checkbox("Verify claims", True)
        if st.button("Ask", type="primary") and q:
            with st.spinner("Working…"):
                out = api("POST", "/ask", json={"question": q, "paper_ids": sel, "mode": mode, "k": k, "rewrite": rewrite, "verify": verify})
            if out:
                show_answer(out, "mp")
    with t2:
        q2 = st.text_input("Analysis request", "Compare these papers and identify their common limitations.")
        if st.button("Run analysis"):
            r = api("POST", "/research", json={"question": q2, "mode": "my_papers", "paper_ids": sel, "max_papers": len(sel), "verify": True, "intent": "gaps"})
            if r:
                st.session_state.mp_run = r["run_id"]
        if st.session_state.get("mp_run"):
            res = follow_run(st.session_state.mp_run)
            if res:
                render_results(st.session_state.mp_run, res)


def page_eval():
    st.header("Evaluation dashboard")
    res = api("GET", "/evaluation/results")
    if not res:
        st.info("No evaluation results yet. Run `python -m scholaragent.evaluation.run_experiments`.")
        return
    for name, data in res.items():
        with st.expander(name, expanded=name in ("retrieval_systems", "answer_systems")):
            if isinstance(data, dict) and data and all(isinstance(v, dict) for v in data.values()):
                try:
                    st.dataframe(pd.DataFrame(data).T, width="stretch")
                    continue
                except Exception:  # noqa: BLE001
                    pass
            st.json(data, expanded=False)
    for img in ("retrieval_systems.png", "chunking.png", "embedders.png", "ablation.png", "error_analysis.png"):
        try:
            st.image(f"{API}/results-files/{img}", caption=img)
        except Exception:  # noqa: BLE001
            pass


def page_history():
    st.header("Previous runs")
    runs = api("GET", "/runs") or []
    for r in runs:
        c = st.columns([5, 1, 1])
        c[0].write(f"**{r['question'][:110]}**  \n<small>{r['created_at'][:16]} · {r['mode']}</small>", unsafe_allow_html=True)
        c[1].write(r["status"])
        if r["status"] == "done" and c[2].button("Open", key=r["run_id"]):
            st.session_state.run_id = r["run_id"]
            st.session_state.result = api("GET", f"/runs/{r['run_id']}?with_result=true")["result"]
            st.session_state.result_run = r["run_id"]
            st.success("Loaded — open the Deep Research page.")


# ----------------------------------------------------------------------------- main
st.sidebar.title("🔬 ScholarAgent")
st.sidebar.caption("Agentic scientific intelligence")
h = health()
if h:
    st.sidebar.success("API online") if h["status"] == "ok" else st.sidebar.warning("API up, LLM offline")
    st.sidebar.caption(f"LLM: {h['llm']['model']}  \nGPU: {h['gpu'].get('name', 'n/a')} ({h['gpu'].get('free_gb', '?')} GB free)  \n"
                       f"Papers: {h.get('library', {}).get('papers', '?')} · chunks: {h.get('library', {}).get('chunks', '?')}")
else:
    st.sidebar.error("API offline")
page = st.sidebar.radio("Navigate", ["Deep Research", "My Papers", "Evaluation", "History"])
{"Deep Research": page_research, "My Papers": page_my_papers, "Evaluation": page_eval, "History": page_history}[page]()
