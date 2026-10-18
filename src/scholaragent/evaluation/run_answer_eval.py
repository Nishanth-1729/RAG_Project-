"""Answer-level evaluation of systems A-F.   python -m scholaragent.evaluation.run_answer_eval --split test

A Dense | B BM25 | C Hybrid | D Hybrid+Rerank | E D + claim verification/self-correction | F E + query rewriting
Requires the LLM server.  Per-question outputs are saved for manual auditing.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict

import numpy as np

from ..agents.verifier_agent import VerifierAgent
from ..config import settings
from ..llm.client import get_llm, meter
from ..rag.answer import AnswerGenerator
from ..rag.citations import Evidence, evidence_from_hits
from ..rag.fusion_helpers import decomposed_retrieve, multi_query_retrieve
from .answer_metrics import EvalJudge, aggregate_claims, claim_metrics
from .dataset import load_jsonl
from .harness import Experiment, Variant, load_docs, save_json
from .retrieval_metrics import evaluate_ranking
from .run_experiments import SEC384, split_papers
from ..utils.logging import get_logger

log = get_logger("answer_eval")

SYSTEMS = {
    "A Dense": dict(mode="dense", verify=False, rewrite=False),
    "B BM25": dict(mode="bm25", verify=False, rewrite=False),
    "C Hybrid": dict(mode="hybrid", verify=False, rewrite=False),
    "D Hybrid+Rerank": dict(mode="hybrid_rerank", verify=False, rewrite=False),
    "E D+Verification": dict(mode="hybrid_rerank", verify=True, rewrite=False),
    "F E+QueryRewrite (full)": dict(mode="hybrid_rerank", verify=True, rewrite=True),
    "G D+NLI-first verification": dict(mode="hybrid_rerank", verify=True, rewrite=False, nli_verify=True),
    "H G+QueryRewrite": dict(mode="hybrid_rerank", verify=True, rewrite=True, nli_verify=True),
    "I D+decomposition+reflective answering": dict(mode="hybrid_rerank", verify=False, rewrite=False, decompose=True, reflect=True),
    "J I+NLI-first verification (final system)": dict(mode="hybrid_rerank", verify=True, rewrite=False, decompose=True, reflect=True, nli_verify=True),
}


class HarnessLibrary:
    """Adapter exposing the Library interface used by agents on top of an experiment retriever."""

    def __init__(self, retriever):
        self.retriever, self.reranker = retriever, retriever.reranker

    def search(self, query, paper_ids=None, k=8, mode="hybrid_rerank", **kw):
        from ..retrieval.vectorstore import Filter
        flt = Filter(paper_ids=set(paper_ids)) if paper_ids is not None else None
        return self.retriever.search(query, k=k, mode=mode, flt=flt, **kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--systems", default="")
    ap.add_argument("--max-q", type=int, default=0)
    ap.add_argument("--embed-model", default=settings.embed_model)
    ap.add_argument("--rerank-model", default=settings.rerank_model)
    ap.add_argument("--tag", default="")
    ap.add_argument("--latency-sample", type=int, default=8)
    ap.add_argument("--nli", default="", help="path of the fine-tuned NLI verifier (independent judge)")
    a = ap.parse_args()
    llm = get_llm()
    assert llm.is_up(), "start the LLM server first (scripts/start_llm.sh)"
    qs = load_jsonl(settings.eval_dir / f"benchmark_{a.split}.jsonl")
    if a.max_q:
        qs = qs[: a.max_q]
    ex = Experiment(load_docs(split_papers(a.split)))
    v = Variant("full", SEC384, a.embed_model, True, rerank_model=a.rerank_model)
    rt = ex.retriever(v, use_rerank=True)
    from ..utils.gpu import free_gpu
    free_gpu()                      # drop indexing activations before the LLM-heavy phase
    lib = HarnessLibrary(rt)
    gen, gen_reflect, judge = AnswerGenerator(llm), AnswerGenerator(llm, reflect=True), EvalJudge(llm)
    verifier = VerifierAgent(lib, llm, nli_mode="off")                  # LLM-only verifier (systems E, F; also used for claim extraction)
    nli_verifier = None
    nli = None
    if a.nli:
        from ..training.nli import NLIVerifier
        nli = NLIVerifier(a.nli)
    names = [s for s in SYSTEMS if not a.systems or s.split()[0] in a.systems.split(",")]
    if any(SYSTEMS[n].get("nli_verify") for n in names):
        from ..training.nli import NLIVerifier
        nli_verifier = VerifierAgent(lib, llm, nli=NLIVerifier(str(settings.models_dir / "scholar-verifier")), nli_mode="primary")
    results, per_q_out = {}, {}

    def retrieve_for(q, cfg):
        if cfg.get("decompose"):
            return decomposed_retrieve(rt, q["question"], cfg["mode"], k=8, llm=llm)[0]
        return multi_query_retrieve(rt, q["question"], cfg["mode"], k=8, rewrite=cfg["rewrite"], llm=llm)[0]

    def make_process(cfg):
        def process(q):
            hits = retrieve_for(q, cfg)
            rr = evaluate_ranking(q, [h.chunk for h in hits], 8)
            ans = (gen_reflect if cfg.get("reflect") else gen).answer(q["question"], hits)
            final_text, evidence, removed, corrected = ans.text, ans.evidence, 0, 0
            if cfg["verify"]:
                rep = (nli_verifier if cfg.get("nli_verify") else verifier).verify_answer(ans)
                final_text = rep["final_answer"]
                removed, corrected = rep["summary"]["removed"], rep["summary"]["corrected"]
                evidence = [Evidence(**{k: e[k] for k in Evidence.__dataclass_fields__ if k in e}) for e in rep["evidence"]]
            c = judge.correctness(q["question"], q["reference_answer"], final_text)
            cm = claim_metrics(final_text, evidence, verifier, judge, nli)
            return {"qid": q["qid"], "type": q["type"], "answer": final_text, "draft": ans.text, "correctness": c, "claims": cm,
                    "abstained": final_text.strip().upper().startswith("INSUFFICIENT"), "removed": removed, "corrected": corrected,
                    "first_rank": rr.first_rank}
        return process

    for name in names:
        cfg = SYSTEMS[name]
        process = make_process(cfg)
        t_sys, u0 = time.perf_counter(), meter.snapshot()
        outs = [o for o in llm.map(process, qs, workers=24) if isinstance(o, dict)]
        u1 = meter.snapshot()
        if len(outs) < len(qs):
            log.warning("%s: %d/%d questions failed", name, len(qs) - len(outs), len(qs))
        # clean per-question latency: sequential, cache off, first N questions
        llm.use_cache = False
        lat = []
        for q in qs[: a.latency_sample]:
            t0 = time.perf_counter()
            hits = retrieve_for(q, cfg)
            ans = (gen_reflect if cfg.get("reflect") else gen).answer(q["question"], hits)
            if cfg["verify"]:
                (nli_verifier if cfg.get("nli_verify") else verifier).verify_answer(ans)
            lat.append(time.perf_counter() - t0)
        llm.use_cache = True
        corr = [o["correctness"] for o in outs]
        agg = {"n": len(outs), "correctness": round(float(np.mean(corr)), 4),
               "correct_strict": round(float(np.mean([c == 1.0 for c in corr])), 4),
               "abstain_rate": round(float(np.mean([o["abstained"] for o in outs])), 4),
               "recall@5": round(float(np.mean([o["first_rank"] is not None and o["first_rank"] <= 5 for o in outs])), 4),
               "latency_s_mean": round(float(np.mean(lat)), 2) if lat else None,
               "llm_calls_per_q": round((u1["calls"] - u0["calls"]) / max(1, len(outs)), 1),
               "tokens_per_q": round(((u1["prompt_tokens"] + u1["completion_tokens"]) - (u0["prompt_tokens"] + u0["completion_tokens"])) / max(1, len(outs))),
               "claims_removed": sum(o["removed"] for o in outs), "claims_corrected": sum(o["corrected"] for o in outs),
               **aggregate_claims([o["claims"] for o in outs])}
        by_t = defaultdict(list)
        for o in outs:
            by_t[o["type"]].append(o["correctness"])
        agg["correctness_by_type"] = {t: round(float(np.mean(v)), 3) for t, v in sorted(by_t.items())}
        results[name], per_q_out[name] = agg, outs
        log.info("== %s: %s (%.0fs)", name, {k: agg[k] for k in ("correctness", "faithfulness", "citation_accuracy", "hallucination_rate")}, time.perf_counter() - t_sys)
        save_json(results, f"answer_systems_{a.split}{a.tag}")
        save_json(per_q_out, f"answer_outputs_{a.split}{a.tag}")
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
