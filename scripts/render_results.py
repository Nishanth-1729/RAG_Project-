"""Render results/*.json into docs/RESULTS.md  (every number comes from a measured run)."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"


def load(name):
    p = R / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def pct(x):
    return "–" if x is None else f"{100 * x:.1f}%"


def f3(x):
    return "–" if x is None else f"{x:.3f}"


def table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def charts(root: Path) -> None:
    """Charts that are not produced by the experiment runner itself."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # noqa: BLE001
        return
    rs = load("retrieval_systems_test")
    if rs:
        import numpy as np
        rows = {k: v for k, v in rs.items() if not k.startswith("_")}
        metrics = ["recall@1", "recall@3", "recall@5", "recall@10", "mrr"]
        fig, ax = plt.subplots(figsize=(11, 5.6))
        w = 0.8 / len(rows)
        for i, (name, row) in enumerate(rows.items()):
            ax.bar(np.arange(len(metrics)) + i * w, [row[m] for m in metrics], w, label=name)
        ax.set_xticks(np.arange(len(metrics)) + 0.4 - w / 2)
        ax.set_xticklabels(["Recall@1", "Recall@3", "Recall@5", "Recall@10", "MRR"])
        ax.set_ylim(0.4, 1.0); ax.grid(axis="y", alpha=.3); ax.set_title("Retrieval architectures (test split, n=100)")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2, fontsize=8, frameon=False)
        fig.tight_layout(); fig.savefig(R / "retrieval_systems.png", dpi=140); plt.close(fig)
    ea = load("error_analysis_test")
    if ea and ea.get("answer_outcomes"):
        d = {k: v["count"] for k, v in ea["answer_outcomes"].items() if k != "correct"}
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.barh(list(d)[::-1], list(d.values())[::-1], color="#c0504d")
        ax.set_xlabel("questions (of 100)")
        ax.set_title(f"Why answers lose points - {ea['system']}")
        fig.tight_layout(); fig.savefig(R / "error_analysis.png", dpi=140); plt.close(fig)
    an = load("answer_systems_test")
    if an:
        names = list(an)
        fig, ax = plt.subplots(figsize=(10, 4.5))
        x = range(len(names))
        ax.bar([i - 0.2 for i in x], [an[n]["correctness"] for n in names], 0.4, label="Correctness (LLM judge)")
        ax.bar([i + 0.2 for i in x], [an[n].get("faithfulness_nli") or 0 for n in names], 0.4, label="Faithfulness (NLI judge)")
        ax.set_xticks(list(x)); ax.set_xticklabels([n.split(" ")[0] for n in names]); ax.set_ylim(0, 1); ax.legend(); ax.grid(axis="y", alpha=.3)
        ax.set_title("End-to-end answer systems A-J (see docs/RESULTS.md for definitions)")
        fig.tight_layout(); fig.savefig(R / "answer_systems.png", dpi=140); plt.close(fig)


def main():
    L = ["# Experimental results", "",
         "_All numbers are measured by the scripts in `src/scholaragent/evaluation/` on the benchmark described in the README "
         "(LLM-generated questions with chunking-independent evidence spans; see 'Limitations of this evaluation')._", ""]
    for split in ("test", "dev"):
        rs = load(f"retrieval_systems_{split}")
        if rs:
            L += [f"## 1. Retrieval architectures ({split} split)", ""]
            rows = []
            for name, a in rs.items():
                if name.startswith("_"):
                    continue
                rows.append([name, f3(a["recall@1"]), f3(a["recall@3"]), f"{f3(a['recall@5'])} {a.get('recall@5_ci95', '')}", f3(a["recall@10"]),
                             f3(a["mrr"]), f3(a["ndcg@5"]), f3(a["precision@5"]), a["latency_ms_mean"]])
            L += [table(["System", "Recall@1", "Recall@3", "Recall@5 (95% CI)", "Recall@10", "MRR", "NDCG@5", "Precision@5", "latency ms"], rows), ""]
            bt = {n: a["by_type"] for n, a in rs.items() if not n.startswith("_")}
            types = sorted(next(iter(bt.values())))
            L += ["Recall@5 by question type:", "", table(["System"] + types, [[n] + [f3(b[t]["recall@5"]) for t in types] for n, b in bt.items()]), ""]
        break
    ch = load("chunking_test")
    if ch:
        L += ["## 2. Chunking strategies (test split)", ""]
        L += [table(["Strategy", "#chunks", "avg tokens", "Dense R@5", "Hybrid R@5", "Hybrid MRR", "Hybrid+Rerank R@5", "Hybrid+Rerank MRR"],
                    [[k, v["n_chunks"], v["avg_tokens"], f3(v["dense"]["recall@5"]), f3(v["hybrid"]["recall@5"]), f3(v["hybrid"]["mrr"]),
                      f3(v["hybrid_rerank"]["recall@5"]), f3(v["hybrid_rerank"]["mrr"])] for k, v in ch.items()]), "", "![chunking](../results/chunking.png)", ""]
    rr = load("rerankers_test")
    if rr:
        L += ["## 3. Rerankers (test split)", ""]
        L += [table(["Reranker", "Recall@1", "Recall@5", "MRR", "NDCG@5", "latency ms/query"],
                    [[k, f3(v.get("recall@1")), f3(v.get("recall@5")), f3(v.get("mrr")), f3(v.get("ndcg@5")), v.get("latency_ms_mean", "–")] if "error" not in v else [k, "error", "", "", "", ""]
                     for k, v in rr.items()]), ""]
    em = load("embedders_test")
    if em:
        L += ["## 4. Embedding models (test split)", ""]
        L += [table(["Model", "Dense R@5", "Dense MRR", "Hybrid R@5", "Hybrid MRR", "Hybrid+Rerank R@5", "Hybrid+Rerank MRR"],
                    [[k, f3(v["dense"]["recall@5"]), f3(v["dense"]["mrr"]), f3(v["hybrid"]["recall@5"]), f3(v["hybrid"]["mrr"]),
                      f3(v.get("hybrid_rerank", {}).get("recall@5")), f3(v.get("hybrid_rerank", {}).get("mrr"))] for k, v in em.items() if "error" not in v]), ""]
    ab = load("ablation_test")
    if ab:
        L += ["## 5. Ablation (test split)", ""]
        L += [table(["Configuration", "Recall@1", "Recall@5", "MRR", "NDCG@5"],
                    [[k, f3(v["recall@1"]), f3(v["recall@5"]), f3(v["mrr"]), f3(v["ndcg@5"])] for k, v in ab.items()]), ""]
    an = load("answer_systems_test")
    if an:
        L += ["## 6. Answer quality of the end-to-end systems (test split, n=100)", "",
              "_Faithfulness (LLM) / hallucination rate / citation metrics: claims are extracted from the final answer and judged by an LLM judge with its own rubric. "
              "Faithfulness (NLI): the same claims judged by the fine-tuned NLI model (independent architecture; for systems G-J, which use that NLI model inside the verifier, this column is NOT independent)._", ""]
        L += [table(["System", "Correctness", "Strict correct", "Abstain", "Faithfulness (LLM)", "Faithfulness (NLI)", "Citation accuracy", "Citation completeness", "Hallucination rate (LLM)", "Claims removed", "s/question", "LLM calls/q"],
                    [[k, f3(v["correctness"]), f3(v["correct_strict"]), f3(v["abstain_rate"]), f3(v.get("faithfulness")), f3(v.get("faithfulness_nli")), f3(v.get("citation_accuracy")),
                      f3(v.get("citation_completeness")), f3(v.get("hallucination_rate")), v["claims_removed"], v["latency_s_mean"], v["llm_calls_per_q"]] for k, v in an.items()]), ""]
        types = sorted({t for v in an.values() for t in v["correctness_by_type"]})
        L += ["Correctness by question type:", "", table(["System"] + types, [[k] + [f3(v["correctness_by_type"].get(t)) for t in types] for k, v in an.items()]), ""]
    ve = load("verifier_eval")
    if ve:
        L += ["## 7. Claim verifiers", "", "Held-out claims generated from the 40 test papers (628 claims, 4 balanced classes):", ""]
        L += [table(["Verifier", "Accuracy", "Macro-F1", "Bad-claim detection rate", "False-alarm rate"],
                    [[k, f3(v["accuracy"]), f3(v["macro_f1"]), f3(v["unsupported_claim_detection_rate"]), f3(v["false_alarm_rate"])] for k, v in ve.items() if isinstance(v, dict) and "unsupported_claim_detection_rate" in v]), ""]
        sf = ve.get("scifact_validation")
        if sf:
            L += ["Human-annotated SciFact validation (340 claims, 3-way; the NLI model was fine-tuned on SciFact-train):", "",
                  table(["Verifier", "Accuracy", "Macro-F1"], [[k, f3(v["accuracy"]), f3(v["macro_f1"])] for k, v in sf.items() if isinstance(v, dict) and "accuracy" in v]), ""]
    ea = load("error_analysis_test")
    if ea:
        L += ["## 8. Error analysis (test split)", "", "Retrieval (does the top-5 contain all evidence?):", "",
              table(["Outcome", "count", "%"], [[k, v["count"], v["percent"]] for k, v in ea["retrieval_outcomes"].items()]), ""]
        if ea.get("answer_outcomes"):
            L += [f"Answers of system *{ea['system']}* (correctness < 1 attributed to one cause, in priority order):", "",
                  table(["Outcome", "count", "%"], [[k, v["count"], v["percent"]] for k, v in ea["answer_outcomes"].items()]), ""]
    ag = load("agent_eval")
    if ag:
        L += ["## 9. Agent evaluation", "", table(["metric", "value"], [[k, v] for k, v in ag["summary"].items()]), "", table(
            ["task", "mode", "status", "intent", "steps", "tool-sel acc", "required covered", "unnecessary", "claims checked", "latency s", "tokens"],
            [[t["task"], t["mode"], t["status"], t.get("intent", ""), t.get("steps", ""), t.get("tool_selection_accuracy", ""), t.get("required_covered", ""),
              t.get("unnecessary_calls", ""), t.get("claims_checked", ""), t["latency_s"], t["tokens"]] for t in ag["tasks"]]), ""]
    tr = {}
    for name in ("scholar-embed", "scholar-rerank", "scholar-verifier"):
        p = ROOT / "models" / name / "train_info.json"
        if p.exists():
            tr[name] = json.loads(p.read_text())
    if tr:
        L += ["## 10. Fine-tuning", "", "```json", json.dumps(tr, indent=1)[:4000], "```", ""]
    charts(ROOT)
    (ROOT / "docs").mkdir(exist_ok=True)
    (ROOT / "docs" / "RESULTS.md").write_text("\n".join(L))
    print("wrote docs/RESULTS.md")


if __name__ == "__main__":
    main()
