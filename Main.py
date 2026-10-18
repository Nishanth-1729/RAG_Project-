"""ScholarAgent command line.

    python Main.py status                      # health of LLM / Qdrant / GPU / library
    python Main.py ingest data/raw_papers/*.pdf
    python Main.py ask "How does X reduce complexity?" [--verify] [--mode hybrid_rerank]
    python Main.py research "Learned preconditioners for iterative solvers" --papers 12 --from 2020
    python Main.py serve                       # FastAPI on :8100   (UI: streamlit run frontend/app.py)
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))


def cmd_status(a):
    from scholaragent.llm.client import get_llm
    from scholaragent.utils.gpu import gpu_info
    from scholaragent.config import settings
    print("GPU:", gpu_info())
    print("LLM up:", get_llm().is_up(), settings.llm_base_url, settings.llm_model)
    try:
        from scholaragent.retrieval.library import get_library
        print("Library:", get_library().stats())
    except Exception as e:  # noqa: BLE001
        print("Library unavailable:", repr(e))


def cmd_ingest(a):
    from scholaragent.retrieval.library import get_library
    lib = get_library()
    paths = [Path(p) for p in a.paths]
    for r in lib.ingest_many(paths, on_progress=lambda i, n, r: print(f"[{i}/{n}] {'ok ' if r['ok'] else 'ERR'} {r.get('title', '')[:80]} {r.get('n_chunks', '')} {r.get('error', '')}")):
        pass
    print(lib.stats())


def cmd_ask(a):
    from scholaragent.agents.services import get_services
    s = get_services()
    ans, t = s.rag.ask(a.question, None, k=a.k, mode=a.mode, rewrite=a.rewrite)
    out = s.verifier.verify_answer(ans) if a.verify else None
    print("\nANSWER:\n", out["final_answer"] if out else ans.text)
    print("\nEVIDENCE:")
    for e in (out["evidence"] if out else [x.to_dict() for x in ans.evidence]):
        print(f"  [{e['id']}] {e['label']} - {e['title'][:70]}")
    if out:
        print("\nVERIFICATION:", json.dumps(out["summary"]))
    print("\ntiming:", t)


def cmd_research(a):
    from scholaragent.agents.services import get_services
    from scholaragent.agents.supervisor import Supervisor, summarize_state
    svc = get_services()
    sup = Supervisor(svc)
    st = sup.run(a.question, mode="deep", params={"max_papers": a.papers, "year_from": a.year_from, "year_to": a.year_to, "verify": not a.no_verify},
                 emit=lambda e: print(f"  [{e['ts']}] {e['stage']}: {e['message']}"))
    res = summarize_state(st, svc)
    out = ROOT / "results" / "last_research.json"
    out.write_text(json.dumps(res, ensure_ascii=False, default=str, indent=1))
    if res.get("report"):
        (ROOT / "results" / "last_report.md").write_text(res["report"]["markdown"])
        print("\nreport -> results/last_report.md")
    print("result ->", out)


def cmd_serve(a):
    import uvicorn
    uvicorn.run("scholaragent.api.main:app", host=a.host, port=a.port, reload=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status").set_defaults(fn=cmd_status)
    p = sub.add_parser("ingest"); p.add_argument("paths", nargs="+"); p.set_defaults(fn=cmd_ingest)
    p = sub.add_parser("ask"); p.add_argument("question"); p.add_argument("--k", type=int, default=8)
    p.add_argument("--mode", default="hybrid_rerank", choices=["dense", "bm25", "hybrid", "hybrid_rerank"])
    p.add_argument("--rewrite", action="store_true"); p.add_argument("--verify", action="store_true"); p.set_defaults(fn=cmd_ask)
    p = sub.add_parser("research"); p.add_argument("question"); p.add_argument("--papers", type=int, default=12)
    p.add_argument("--from", dest="year_from", type=int, default=None); p.add_argument("--to", dest="year_to", type=int, default=None)
    p.add_argument("--no-verify", action="store_true"); p.set_defaults(fn=cmd_research)
    p = sub.add_parser("serve"); p.add_argument("--host", default="0.0.0.0"); p.add_argument("--port", type=int, default=8100); p.set_defaults(fn=cmd_serve)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
