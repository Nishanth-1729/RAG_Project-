"""Agent evaluation: runs the Supervisor on a task suite and reports task completion, tool selection accuracy,
unnecessary calls, steps, latency, tokens/cost, failure handling.     python -m scholaragent.evaluation.run_agent_eval"""
from __future__ import annotations

import argparse
import json
import time
import traceback

from ..agents.services import get_services
from ..agents.supervisor import Supervisor, summarize_state
from ..config import settings
from ..llm.client import meter
from ..utils.logging import get_logger
from .agent_metrics import score_run
from .harness import save_json

log = get_logger("agent_eval")

TASKS = [
    dict(id="survey_deep", mode="deep", q="What are recent methods for learned preconditioners for iterative linear solvers?", params=dict(max_papers=8, year_from=2021)),
    dict(id="compare_deep", mode="deep", q="Compare graph neural network preconditioners with classical incomplete factorization preconditioners.", params=dict(max_papers=6, year_from=2021, intent="comparison")),
    dict(id="qa_deep", mode="deep", q="How does the flexible GMRES method work with a learned preconditioner?", params=dict(max_papers=5, year_from=2021, intent="qa")),
    dict(id="qa_mypapers", mode="my_papers", q="Which of these papers use graph neural networks, and for what purpose?", params=dict(intent="qa")),
    dict(id="compare_mypapers", mode="my_papers", q="Compare the learned multigrid approaches in these papers.", params=dict(intent="comparison")),
    dict(id="gaps_mypapers", mode="my_papers", q="What are the common limitations of these papers and which research directions remain?", params=dict(intent="gaps")),
    dict(id="nonsense_query", mode="deep", q="entanglement of banana sheep in quantum cafeteria topology", params=dict(max_papers=4)),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--n-mypapers", type=int, default=8)
    a = ap.parse_args()
    svc = get_services()
    sup = Supervisor(svc)
    my_ids = svc.lib.store.collection_papers("eval_corpus")[: a.n_mypapers]
    rows = []
    prev = settings.results_dir / "agent_eval.json"
    if a.only and prev.exists():            # re-running a subset: keep the other tasks' previous results
        rows = [r for r in json.loads(prev.read_text()).get("tasks", []) if r["task"] not in a.only.split(",")]
    for t in TASKS:
        if a.only and t["id"] not in a.only.split(","):
            continue
        u0, t0 = meter.snapshot(), time.perf_counter()
        status, err, score, res = "ok", None, {}, {}
        try:
            st = sup.run(t["q"], mode=t["mode"], params={"verify": True, **t["params"]}, paper_ids=my_ids if t["mode"] == "my_papers" else None,
                         run_id=f"eval_{t['id']}", emit=lambda e: log.info("   %s: %s", e["stage"], e["message"][:100]))
            res = summarize_state(st, svc)
            runs_dir = settings.results_dir / "agent_runs"
            runs_dir.mkdir(exist_ok=True)
            (runs_dir / f"{t['id']}.json").write_text(json.dumps(res, ensure_ascii=False, default=str))
            if res.get("report"):
                (runs_dir / f"{t['id']}.md").write_text(res["report"]["markdown"])
            score = score_run(st, t["mode"])
            if t["id"] == "nonsense_query":
                # success = declines (no report) OR a report that carries an explicit low-coverage caveat; never a confident report
                declined = bool(res.get("note")) and not res.get("report")
                caveat = bool(res.get("report")) and "Low coverage" in res["report"]["markdown"]
                score["deliverable"] = declined or caveat
                score["graceful"] = declined or caveat
                if declined:                      # correct behaviour = no tools needed
                    score.update(required_covered=1.0, tool_selection_accuracy=1.0)
        except Exception as e:  # noqa: BLE001
            status, err = "failed", repr(e)[:300]
            log.error(traceback.format_exc())
        u1 = meter.snapshot()
        row = {"task": t["id"], "mode": t["mode"], "status": status, "error": err, "latency_s": round(time.perf_counter() - t0, 1),
               "llm_calls": u1["calls"] - u0["calls"], "tokens": (u1["prompt_tokens"] + u1["completion_tokens"]) - (u0["prompt_tokens"] + u0["completion_tokens"]),
               "n_papers": len(res.get("papers") or []), **score}
        v = ((res.get("report") or {}).get("verification")) or (((res.get("answer") or {}).get("verification") or {}).get("summary"))
        row["claims_checked"] = (v or {}).get("claims_checked")
        rows.append(row)
        log.info("== %s", row)
        save_json({"tasks": rows}, "agent_eval")
    ok = [r for r in rows if r["status"] == "ok" and r.get("deliverable")]
    summary = {"tasks": len(rows), "task_completion_rate": round(len(ok) / max(1, len(rows)), 3),
               "failure_rate": round(sum(r["status"] != "ok" for r in rows) / max(1, len(rows)), 3),
               "avg_steps": round(sum(r.get("steps", 0) for r in rows) / max(1, len(rows)), 1),
               "avg_tool_selection_accuracy": round(sum(r.get("tool_selection_accuracy", 0) for r in rows) / max(1, len(rows)), 3),
               "avg_required_actions_covered": round(sum(r.get("required_covered", 0) for r in rows) / max(1, len(rows)), 3),
               "unnecessary_calls_total": sum(r.get("unnecessary_calls", 0) for r in rows),
               "avg_latency_s": round(sum(r["latency_s"] for r in rows) / max(1, len(rows)), 1),
               "total_tokens": sum(r["tokens"] for r in rows), "cost_usd": 0.0}
    save_json({"summary": summary, "tasks": rows}, "agent_eval")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
