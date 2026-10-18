"""Build contrastive training data from the synthetic QA sets (papers split: train / dev / test never overlap).

For every question:  positive = chunk(s) that contain the evidence span;
hard negatives = top hybrid-retrieved chunks that are NOT relevant, filtered with a teacher cross-encoder so that
'negatives' which actually answer the question (false negatives) are removed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


from ..config import settings
from ..evaluation.dataset import load_jsonl
from ..evaluation.harness import Experiment, Variant, load_docs
from ..evaluation.retrieval_metrics import grade
from ..evaluation.run_experiments import SEC384, split_papers
from ..utils.logging import get_logger

log = get_logger("pairs")
OUT = settings.data_dir / "training"


def build(split: str, qa_path: Path, embed_model: str, teacher: str, n_neg: int = 4, max_q: int = 0, seed: int = 0) -> dict:
    qs = load_jsonl(qa_path)
    if max_q:
        qs = qs[:max_q]
    entries = split_papers(split if split in ("dev", "test") else "train")
    docs = load_docs(entries)
    ex = Experiment(docs)
    v = Variant("pairs", SEC384, embed_model, True)
    rt = ex.retriever(v, use_rerank=False)
    teacher_rr = ex.reranker(teacher)
    by_paper: dict[str, list] = {}
    for c in rt.index.chunks.values():
        by_paper.setdefault(c.paper_id, []).append(c)
    rows, skipped = [], 0
    for i, q in enumerate(qs):
        evs = q["evidences"]
        pos = [c for p in {e["paper_id"] for e in evs} for c in by_paper.get(p, []) if grade(c, evs) == 3]
        if not pos:
            skipped += 1
            continue
        hits = rt.search(q["question"], k=24, mode="hybrid", n_candidates=40)
        cands = [h.chunk for h in hits if grade(h.chunk, evs) < 2]          # drop positives and near-positives
        cands = cands[:12]
        p0 = pos[0]
        texts = [rt._rerank_text(p0)] + [rt._rerank_text(c) for c in cands]
        sc = teacher_rr.score(q["question"], texts)          # CrossEncoder.predict already returns probabilities (sigmoid)
        keep = [c for c, s in zip(cands, sc[1:]) if s < 0.3 or s < sc[0] - 0.45]
        if len(keep) < 2:
            skipped += 1
            continue
        negs = keep[:n_neg]
        unfiltered = [rt._rerank_text(h.chunk) for h in hits if grade(h.chunk, evs) < 2][:20]     # realistic candidate pool for dev evaluation
        rows.append({"dev_cands_plain": unfiltered,"qid": q["qid"], "type": q["type"], "query": q["question"], "pos_chunk": p0.chunk_id,
                     "pos": p0.embed_text(True), "pos_plain": rt._rerank_text(p0),
                     "negs": [c.embed_text(True) for c in negs], "negs_plain": [rt._rerank_text(c) for c in negs],
                     "teacher_pos": float(sc[0]), "paper_id": p0.paper_id})
        if (i + 1) % 500 == 0:
            log.info("%s: %d/%d questions processed (%d kept)", split, i + 1, len(qs), len(rows))
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"pairs_{split}.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    # dev IR-evaluation corpus (all chunks of dev papers) for early stopping / model selection
    if split == "dev":
        corpus = {c.chunk_id: c.embed_text(True) for c in rt.index.chunks.values()}
        corpus_plain = {c.chunk_id: rt._rerank_text(c) for c in rt.index.chunks.values()}
        (OUT / "dev_corpus.json").write_text(json.dumps({"corpus": corpus, "corpus_plain": corpus_plain}))
        rel = {}
        for q in qs:
            ids = {c.chunk_id for p in {e["paper_id"] for e in q["evidences"]} for c in by_paper.get(p, []) if grade(c, q["evidences"]) == 3}
            if ids:
                rel[q["qid"]] = sorted(ids)
        (OUT / "dev_queries.json").write_text(json.dumps({"queries": {q["qid"]: q["question"] for q in qs if q["qid"] in rel}, "relevant": rel}))
    stats = {"split": split, "questions": len(qs), "kept": len(rows), "skipped": skipped}
    log.info("%s", stats)
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embed-model", default=settings.embed_model)
    ap.add_argument("--teacher", default="BAAI/bge-reranker-v2-m3")
    ap.add_argument("--max-q", type=int, default=0)
    ap.add_argument("--only", default="train,dev")
    a = ap.parse_args()
    stats = []
    if "train" in a.only:
        stats.append(build("train", settings.eval_dir / "train_qa.jsonl", a.embed_model, a.teacher, max_q=a.max_q))
    if "dev" in a.only:
        stats.append(build("dev", settings.eval_dir / "benchmark_dev.jsonl", a.embed_model, a.teacher))
    (OUT / "pairs_stats.json").write_text(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
