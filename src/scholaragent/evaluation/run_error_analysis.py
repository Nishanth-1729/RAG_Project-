"""Error analysis of the full system.   python -m scholaragent.evaluation.run_error_analysis [--answers answer_outputs_test]

Retrieval misses (Recall@5) are attributed to: parsing/chunking loss | ranking failure | retrieval failure | wrong paper | equation.
Answer failures (correctness < 1) are attributed to: retrieval failure | abstention | hallucinated claim | citation error | generation failure.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter

from ..config import settings
from .dataset import load_jsonl
from .harness import Experiment, Variant, load_docs, save_json
from .retrieval_metrics import contains_span, grade
from .run_experiments import SEC384, split_papers


def pct(c: Counter) -> dict:
    n = sum(c.values())
    return {k: {"count": v, "percent": round(100 * v / n, 1)} for k, v in c.most_common()} if n else {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--system", default="G D+NLI-first verification")
    ap.add_argument("--answers", default="")
    a = ap.parse_args()
    qs = load_jsonl(settings.eval_dir / f"benchmark_{a.split}.jsonl")
    docs = {d.paper_id: d for d in load_docs(split_papers(a.split))}
    ex = Experiment(list(docs.values()))
    v = Variant("full", SEC384, settings.embed_model, True)
    rt = ex.retriever(v, use_rerank=True)
    by_paper: dict[str, list] = {}
    for c in rt.index.chunks.values():
        by_paper.setdefault(c.paper_id, []).append(c)

    ret_fail, ret_rows = Counter(), []
    in_top8: dict[str, bool] = {}
    for q in qs:
        evs = q["evidences"]
        top = rt.search(q["question"], k=8, mode="hybrid_rerank", n_candidates=30)
        pool = rt.search(q["question"], k=30, mode="hybrid", n_candidates=30)
        top5 = [h.chunk for h in top[:5]]
        covered = [any(contains_span(c.text, e["span"]) and c.paper_id == e["paper_id"] for c in top5) for e in evs]
        in_top8[q["qid"]] = any(contains_span(h.chunk.text, e["span"]) and h.chunk.paper_id == e["paper_id"] for h in top for e in evs)
        if all(covered):
            ret_fail["ok (all evidence in top-5)"] += 1
            continue
        miss = [e for e, c in zip(evs, covered) if not c][0]
        paper_chunks = by_paper.get(miss["paper_id"], [])
        in_paper = [c for c in paper_chunks if contains_span(c.text, miss["span"])]
        if not in_paper:
            cat = "parsing/chunking loss (evidence text not in any chunk)"
        elif any(h.chunk.chunk_id in {c.chunk_id for c in in_paper} for h in pool):
            cat = "ranking failure (candidate found, ranked below top-5)"
        elif top and top[0].chunk.paper_id != miss["paper_id"]:
            cat = "retrieval failure - wrong paper retrieved"
        else:
            cat = "retrieval failure (not in top-30 candidates)"
        if miss["paragraph"].count("=") + miss["paragraph"].count("∑") + miss["paragraph"].count("∥") > 3 and cat != "ranking failure (candidate found, ranked below top-5)":
            cat = "equation-heavy evidence (" + cat.split(" (")[0] + ")"
        ret_fail[cat] += 1
        ret_rows.append({"qid": q["qid"], "type": q["type"], "category": cat, "question": q["question"]})

    out = {"split": a.split, "n_questions": len(qs), "retrieval_outcomes": pct(ret_fail), "retrieval_misses": ret_rows}
    ans_path = settings.results_dir / f"{a.answers or 'answer_outputs_' + a.split}.json"
    if ans_path.exists():
        outs = json.loads(ans_path.read_text()).get(a.system)
        if outs:
            fail = Counter()
            rows = []
            for o in outs:
                if o["correctness"] >= 1.0:
                    fail["correct"] += 1
                    continue
                cm = o["claims"]
                if not in_top8.get(o["qid"], False):
                    cat = "retrieval failure (evidence not in context)"
                elif o["abstained"]:
                    cat = "over-cautious abstention"
                elif cm.get("hallucinated", 0) > 0:
                    cat = "hallucinated claim"
                elif cm.get("n_cited_claims", 0) > cm.get("citation_correct", 0):
                    cat = "incorrect citation"
                else:
                    cat = "generation failure (incomplete / misread evidence)"
                fail[cat] += 1
                rows.append({"qid": o["qid"], "type": o["type"], "category": cat, "correctness": o["correctness"]})
            out["system"] = a.system
            out["answer_outcomes"] = pct(fail)
            f_only = Counter({k: v for k, v in fail.items() if k != "correct"})
            out["answer_failure_breakdown"] = {k: v for k, v in f_only.items()}
            out["answer_failures"] = rows
    ag = settings.results_dir / "agent_eval.json"
    if ag.exists():
        tasks = json.loads(ag.read_text()).get("tasks", [])
        out["agent_failures"] = {"agent planning/execution failures": sum(t["status"] != "ok" for t in tasks), "tasks": len(tasks),
                                 "tasks_with_non_fatal_errors": sum(bool(t.get("errors")) for t in tasks)}
    save_json(out, f"error_analysis_{a.split}")
    print(json.dumps({k: v for k, v in out.items() if k not in ("retrieval_misses", "answer_failures")}, indent=1))


if __name__ == "__main__":
    main()
