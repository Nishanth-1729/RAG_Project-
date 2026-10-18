"""Run the evaluation suite.   python -m scholaragent.evaluation.run_experiments --split test --only retrieval,chunking

Experiments (all numbers are measured, never hard-coded):
  retrieval  : dense | BM25 | hybrid (RRF) | hybrid + rerank           (+ per question-type, bootstrap CIs)
  chunking   : fixed 256/512/1024, section-aware 256/384/512, semantic
  rerankers  : none / several cross-encoders (+ fine-tuned if present)  (+ latency)
  embedders  : several embedding models (+ fine-tuned if present)
  ablation   : full system minus one component at a time
  answers    : answer-level metrics for systems A-E (needs LLM server)
  errors     : error analysis of the full system (needs LLM server for answer failures)
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from ..config import settings
from ..ingestion.chunker import ChunkerConfig
from ..utils.gpu import free_gpu
from ..utils.logging import get_logger
from .dataset import load_jsonl
from .harness import Experiment, Variant, load_docs, run_retrieval, save_json
from .retrieval_metrics import bootstrap_ci, by_type

log = get_logger("experiments")
SEC384 = ChunkerConfig("section", 384, 48)
MODES = ["dense", "bm25", "hybrid", "hybrid_rerank"]


def split_papers(split: str) -> list[dict]:
    man = json.loads((settings.eval_dir / "corpus.json").read_text())
    splits = json.loads((settings.eval_dir / "splits.json").read_text())
    pool = {e["paper_id"]: e for e in man["eval"] + man["train"]}
    return [pool[p] for p in splits[split] if p in pool]


def short(agg: dict, keys=("recall@1", "recall@3", "recall@5", "recall@10", "mrr", "ndcg@5", "precision@5", "avg_ctx_tokens@5", "latency_ms_mean")) -> dict:
    return {k: agg[k] for k in keys if k in agg}


# ---------------------------------------------------------------------------- experiments
SYSTEMS = [("Dense (vector)", "dense", None), ("BM25", "bm25", None), ("Hybrid (RRF)", "hybrid", "rrf"), ("Hybrid (RSF)", "hybrid", "rsf"),
           ("Hybrid (RRF) + Reranker", "hybrid_rerank", "rrf"), ("Hybrid (RSF) + Reranker", "hybrid_rerank", "rsf")]


def exp_retrieval(ex: Experiment, qs, v: Variant, final_reranker: str | None = None) -> dict:
    out, per_q = {}, {}
    systems = list(SYSTEMS) + ([("Hybrid (RSF) + fine-tuned Reranker [final system]", "hybrid_rerank", "rsf")] if final_reranker else [])
    for label, mode, fusion in systems:
        if label.endswith("[final system]"):
            v = Variant(v.name, v.chunk_cfg, v.embed_model, v.contextual, rerank_model=final_reranker)
        rt = ex.retriever(v, use_rerank=(mode == "hybrid_rerank"))
        kw = {} if fusion is None else {"fusion": fusion, "rrf_weights": (1.0, 1.0) if fusion == "rrf" else tuple(ex.rrf_weights if ex.fusion == "rsf" else (0.4, 0.6))}
        res, agg, _ = run_retrieval(rt, qs, mode, **kw)
        r5 = [float(r.first_rank is not None and r.first_rank <= 5) for r in res]
        mrr = [1 / r.first_rank if r.first_rank else 0.0 for r in res]
        agg["recall@5_ci95"] = [round(x, 3) for x in bootstrap_ci(r5)]
        agg["mrr_ci95"] = [round(x, 3) for x in bootstrap_ci(mrr)]
        agg["by_type"] = by_type(res)
        out[label] = agg
        per_q[label] = [r.first_rank for r in res]
    out["_per_question_first_rank"] = per_q
    return out


def exp_chunking(ex: Experiment, qs, base: Variant) -> dict:
    cfgs = {"fixed-256": ChunkerConfig("fixed", 256, 32), "fixed-512": ChunkerConfig("fixed", 512, 64),
            "fixed-1024": ChunkerConfig("fixed", 1024, 100),
            "section-256": ChunkerConfig("section", 256, 32), "section-384": SEC384, "section-512": ChunkerConfig("section", 512, 64),
            "semantic-384": ChunkerConfig("semantic", 384, 0)}
    out = {}
    for name, cfg in cfgs.items():
        v = Variant(name, cfg, base.embed_model, base.contextual)
        row = {"n_chunks": len(ex.chunks(cfg, v.embed_model)),
               "avg_tokens": round(float(np.mean([c.n_tokens for c in ex.chunks(cfg, v.embed_model)])), 1)}
        for mode in ("dense", "hybrid", "hybrid_rerank"):
            rt = ex.retriever(v, use_rerank=(mode == "hybrid_rerank"))
            _, agg, _ = run_retrieval(rt, qs, mode)
            row[mode] = short(agg)
        out[name] = row
        log.info("chunking %s: hybrid R@5=%.3f", name, row["hybrid"]["recall@5"])
    return out


def exp_rerankers(ex: Experiment, qs, base: Variant, extra: list[str]) -> dict:
    names = ["BAAI/bge-reranker-v2-m3", "BAAI/bge-reranker-base", "cross-encoder/ms-marco-MiniLM-L-6-v2", *extra]
    out = {}
    rt = ex.retriever(base, use_rerank=False)
    _, agg, _ = run_retrieval(rt, qs, "hybrid")
    out["none (hybrid, RSF fusion)"] = short(agg)
    for name in names:
        try:
            v = Variant(name, base.chunk_cfg, base.embed_model, base.contextual, rerank_model=name)
            rt = ex.retriever(v, use_rerank=True)
            _, agg, _ = run_retrieval(rt, qs, "hybrid_rerank")
            out[name] = short(agg)
            ex._rerankers.pop(name, None)
            free_gpu()
        except Exception as e:  # noqa: BLE001
            log.warning("reranker %s failed: %r", name, e)
            out[name] = {"error": repr(e)[:150]}
    return out


def exp_embedders(ex: Experiment, qs, base: Variant, extra: list[str]) -> dict:
    names = ["sentence-transformers/all-MiniLM-L6-v2", "BAAI/bge-small-en-v1.5", "BAAI/bge-base-en-v1.5", "BAAI/bge-large-en-v1.5",
             "BAAI/bge-m3", "intfloat/e5-base-v2", *extra]
    out = {}
    for name in names:
        try:
            v = Variant(name, base.chunk_cfg, name, base.contextual)
            row = {}
            for mode in ("dense", "hybrid", "hybrid_rerank"):
                rt = ex.retriever(v, use_rerank=(mode == "hybrid_rerank"))
                _, agg, _ = run_retrieval(rt, qs, mode)
                row[mode] = short(agg)
            out[name] = row
            ex._embedders.pop(name, None)
            ex._caches.pop(name, None)
            free_gpu()
        except Exception as e:  # noqa: BLE001
            log.warning("embedder %s failed: %r", name, e)
            out[name] = {"error": repr(e)[:150]}
    return out


def exp_fusion(ex: Experiment, qs, base: Variant) -> dict:
    """Choose the fusion rule on DEV (pool recall@30 matters because the reranker only sees this pool)."""
    from .retrieval_metrics import evaluate_ranking
    rt = ex.retriever(base, use_rerank=False)
    cfgs = {"RRF (1:1)": ("rrf", (1, 1)), "RRF (dense 1 : bm25 2)": ("rrf", (1, 2)), "RRF (dense 1 : bm25 3)": ("rrf", (1, 3)),
            "RSF (0.5/0.5)": ("rsf", (0.5, 0.5)), "RSF (0.4 dense/0.6 bm25)": ("rsf", (0.4, 0.6)), "RSF (0.3/0.7)": ("rsf", (0.3, 0.7)),
            "RSF (0.2/0.8)": ("rsf", (0.2, 0.8))}
    out = {}
    for name, (fusion, w) in cfgs.items():
        res = []
        for q in qs:
            hits = rt.search(q["question"], k=30, mode="hybrid", n_candidates=30, fusion=fusion, rrf_weights=w)
            res.append(evaluate_ranking(q, [h.chunk for h in hits], 30))
        fr = [r.first_rank for r in res]
        out[name] = {"recall@5": round(float(np.mean([f is not None and f <= 5 for f in fr])), 4),
                     "recall@10": round(float(np.mean([f is not None and f <= 10 for f in fr])), 4),
                     "pool_recall@30": round(float(np.mean([f is not None for f in fr])), 4),
                     "mrr": round(float(np.mean([1 / f if f else 0 for f in fr])), 4)}
    return out


def exp_ablation(ex: Experiment, qs, base: Variant) -> dict:
    """Full system minus one component at a time (all rows go through the same metric code)."""
    out = {}

    def run(label, v, mode):
        rt = ex.retriever(v, use_rerank=mode.endswith("_rerank"))
        _, agg, _ = run_retrieval(rt, qs, mode)
        out[label] = short(agg)
    run("Full system (section chunks + contextual header + hybrid + rerank)", base, "hybrid_rerank")
    run("- BM25 (dense + rerank)", base, "dense_rerank")
    run("- dense retriever (BM25 + rerank)", base, "bm25_rerank")
    run("- reranker (hybrid only)", base, "hybrid")
    run("- section-aware chunking (fixed 384 tokens)", Variant("fixed", ChunkerConfig("fixed", 384, 48), base.embed_model, base.contextual, base.rerank_model), "hybrid_rerank")
    run("- contextual header (embed raw chunk text)", Variant("nocontext", base.chunk_cfg, base.embed_model, False, base.rerank_model), "hybrid_rerank")
    return out


# ---------------------------------------------------------------------------- plots
def plots(results: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = settings.results_dir
    if "retrieval_systems" in results:
        r = {k: v for k, v in results["retrieval_systems"].items() if not k.startswith("_")}
        metrics = ["recall@1", "recall@3", "recall@5", "recall@10", "mrr"]
        fig, ax = plt.subplots(figsize=(9, 4.2))
        w = 0.2
        for i, (name, row) in enumerate(r.items()):
            ax.bar(np.arange(len(metrics)) + i * w, [row[m] for m in metrics], w, label=name)
        ax.set_xticks(np.arange(len(metrics)) + 1.5 * w); ax.set_xticklabels(["Recall@1", "Recall@3", "Recall@5", "Recall@10", "MRR"])
        ax.set_ylim(0, 1); ax.legend(); ax.set_title("Retrieval architectures"); ax.grid(axis="y", alpha=.3)
        fig.tight_layout(); fig.savefig(d / "retrieval_systems.png", dpi=140); plt.close(fig)
    if "chunking" in results:
        c = results["chunking"]
        fig, ax = plt.subplots(figsize=(8, 4.2))
        for fam, mk in (("fixed", "o-"), ("section", "s-")):
            xs = [(v["avg_tokens"], v["hybrid"]["recall@5"]) for k, v in c.items() if k.startswith(fam)]
            if xs:
                ax.plot(*zip(*sorted(xs)), mk, label=f"{fam} (hybrid)")
        for k, v in c.items():
            if k.startswith("semantic"):
                ax.plot([v["avg_tokens"]], [v["hybrid"]["recall@5"]], "^", ms=9, label="semantic (hybrid)")
        ax.set_xlabel("average chunk size (tokens)"); ax.set_ylabel("Recall@5"); ax.grid(alpha=.3); ax.legend(); ax.set_title("Chunk size vs Recall@5")
        fig.tight_layout(); fig.savefig(d / "chunking.png", dpi=140); plt.close(fig)
    if "embedders" in results:
        e = {k: v for k, v in results["embedders"].items() if "error" not in v}
        fig, ax = plt.subplots(figsize=(9, 4.2))
        names = list(e)
        ax.barh(names, [e[n]["hybrid"]["recall@5"] for n in names], label="hybrid R@5")
        ax.barh(names, [e[n]["dense"]["recall@5"] for n in names], height=.4, label="dense R@5")
        ax.set_xlim(0, 1); ax.legend(); ax.set_title("Embedding models"); fig.tight_layout(); fig.savefig(d / "embedders.png", dpi=140); plt.close(fig)
    if "ablation" in results:
        a = results["ablation"]
        fig, ax = plt.subplots(figsize=(9, 4))
        names = list(a)
        ax.barh(names[::-1], [a[n]["recall@5"] for n in names][::-1])
        ax.set_xlim(0, 1); ax.set_title("Ablation: Recall@5"); fig.tight_layout(); fig.savefig(d / "ablation.png", dpi=140); plt.close(fig)
    if "error_analysis" in results and results["error_analysis"].get("answer_failure_breakdown"):
        b = results["error_analysis"]["answer_failure_breakdown"]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.pie(list(b.values()), labels=list(b), autopct="%1.0f%%"); ax.set_title("Answer failure categories (full system)")
        fig.tight_layout(); fig.savefig(d / "error_analysis.png", dpi=140); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["test", "dev"])
    ap.add_argument("--only", default="retrieval,chunking,rerankers,embedders,fusion,ablation")
    ap.add_argument("--fusion", default=settings.fusion)
    ap.add_argument("--rrf-weights", default=settings.fusion_weights)
    ap.add_argument("--extra-reranker", action="append", default=[])
    ap.add_argument("--extra-embedder", action="append", default=[])
    ap.add_argument("--embed-model", default=settings.embed_model)
    ap.add_argument("--rerank-model", default="BAAI/bge-reranker-v2-m3", help="reranker used by the generic experiments (the fine-tuned one is compared separately)")
    ap.add_argument("--tag", default="")
    ap.add_argument("--max-q", type=int, default=0)
    args = ap.parse_args()
    only = set(args.only.split(","))
    qs = load_jsonl(settings.eval_dir / f"benchmark_{args.split}.jsonl")
    if args.max_q:
        qs = qs[: args.max_q]
    docs = load_docs(split_papers(args.split))
    log.info("%s split: %d papers, %d questions", args.split, len(docs), len(qs))
    ex = Experiment(docs, fusion=args.fusion, rrf_weights=tuple(float(x) for x in args.rrf_weights.split(",")))
    base = Variant("full", SEC384, args.embed_model, True, rerank_model=args.rerank_model)
    suffix = f"_{args.split}{args.tag}"
    res: dict = {}
    steps = [("retrieval", "retrieval_systems", lambda: exp_retrieval(ex, qs, base, args.extra_reranker[0] if args.extra_reranker else None)),
             ("chunking", "chunking", lambda: exp_chunking(ex, qs, base)),
             ("rerankers", "rerankers", lambda: exp_rerankers(ex, qs, base, args.extra_reranker)),
             ("embedders", "embedders", lambda: exp_embedders(ex, qs, base, args.extra_embedder)),
             ("fusion", "fusion", lambda: exp_fusion(ex, qs, base)),
             ("ablation", "ablation", lambda: exp_ablation(ex, qs, base))]
    for key, name, fn in steps:
        if key in only:
            t0 = time.time()
            res[name] = fn()
            save_json(res[name], name + suffix)
            log.info("== %s done in %.0fs", name, time.time() - t0)
    if args.split == "test" and not args.tag:
        plots(res)
    print(json.dumps({k: ({kk: (vv.get("recall@5") if isinstance(vv, dict) else vv) for kk, vv in v.items() if not kk.startswith("_")}) for k, v in res.items()}, indent=1, default=str)[:3000])


if __name__ == "__main__":
    main()
