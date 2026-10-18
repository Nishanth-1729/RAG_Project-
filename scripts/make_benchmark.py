"""Generate (1) the 100-question TEST benchmark, (2) a DEV set for model/config selection, (3) TRAIN QA pairs.

Splits are by paper:  test = eval corpus (topic papers) | dev = 15 held-out papers from the train corpus | train = rest.
Needs the LLM server (scripts/start_llm.sh) running.
"""
import argparse
import csv
import json
import random
from collections import Counter

import numpy as np

from scholaragent.config import settings
from scholaragent.evaluation.dataset import (TYPES, dedupe_questions, evidence_units, generate_comparison, generate_qa,
                                             quality_filter, sample_units, save_jsonl, ambiguity_filter)
from scholaragent.evaluation.harness import Experiment, Variant
from scholaragent.evaluation.retrieval_metrics import grade
from scholaragent.evaluation.run_experiments import SEC384
from scholaragent.evaluation.harness import load_docs
from scholaragent.llm.client import get_llm, meter
from scholaragent.retrieval.embedder import Embedder

TEST_COUNTS = {"factual": 20, "methodological": 20, "mathematical": 20, "comparison": 15, "results": 15, "limitations": 10}
DEV_COUNTS = {"factual": 30, "methodological": 30, "mathematical": 25, "comparison": 20, "results": 25, "limitations": 20}


def make_pairs(units, n, rng, embedder):
    pool = [u for u in units if u.canonical in ("method", "results", "experiments", "introduction", "background")]
    rng.shuffle(pool)
    pool = pool[: max(400, n * 6)]
    E = embedder([u.text[:600] for u in pool])
    sims = E @ E.T
    pairs, used = [], set()
    for i in range(len(pool)):
        order = np.argsort(-sims[i])
        for j in order[1:12]:
            if pool[j].paper_id != pool[i].paper_id and 0.55 < sims[i, j] < 0.93 and (i, j) not in used and (j, i) not in used:
                pairs.append((pool[i], pool[j]))
                used.add((i, j))
                break
        if len(pairs) >= n:
            break
    return pairs


def build(name, units, counts, oversample, rng, embedder, llm, docs=None):
    items = []
    rt = None
    if docs is not None:
        rt = Experiment(docs).retriever(Variant('amb', SEC384, embedder.model_name, True), use_rerank=True)

    def competitors(it):
        hits = rt.search(it['question'], k=10, mode='hybrid_rerank', n_candidates=30)
        return [h.chunk.text for h in hits if grade(h.chunk, it['evidences']) < 2][:5]

    for t in TYPES:
        if t == "comparison":
            gen = generate_comparison(make_pairs(units, int(counts[t] * oversample), rng, embedder), llm)
        else:
            gen = generate_qa(sample_units(units, t, int(counts[t] * oversample), rng), t, llm)
        n0 = len(gen)
        gen = quality_filter(gen, llm, min_spec=4)
        if rt is not None:
            n1 = len(gen)
            gen = ambiguity_filter(gen, competitors, llm)
            print(f"[{name}] {t}: ambiguity filter {n1} -> {len(gen)}", flush=True)
        gen = dedupe_questions(gen)
        rng.shuffle(gen)
        print(f"[{name}] {t}: generated {n0} -> {len(gen)} after quality filter (need {counts[t]})", flush=True)
        items.extend(gen[: counts[t]])
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--train-per-paper", type=int, default=45)
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--only-train", action="store_true")
    args = ap.parse_args()
    rng = random.Random(args.seed)
    man = json.loads((settings.eval_dir / "corpus.json").read_text())
    llm, emb = get_llm(), Embedder()
    assert llm.is_up(), "LLM server is not running (scripts/start_llm.sh)"

    test_docs = load_docs(man["eval"])
    train_all = man["train"]
    rng.shuffle(train_all)
    dev_entries, train_entries = train_all[:15], train_all[15:]
    dev_docs, train_docs = load_docs(dev_entries), load_docs(train_entries)
    (settings.eval_dir / "splits.json").write_text(json.dumps({
        "test": [d.paper_id for d in test_docs], "dev": [d.paper_id for d in dev_docs], "train": [d.paper_id for d in train_docs]}, indent=1))
    print("papers: test", len(test_docs), "dev", len(dev_docs), "train", len(train_docs), flush=True)

    units = lambda docs: [u for d in docs for u in evidence_units(d)]  # noqa: E731
    if not args.only_train:
        test = build("test", units(test_docs), TEST_COUNTS, 5.0, rng, emb, llm, docs=test_docs)
        for i, q in enumerate(test):
            q["qid"] = f"t{i:03d}"
        save_jsonl(test, settings.eval_dir / "benchmark_test.jsonl")
        with open(settings.eval_dir / "benchmark_test_audit.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["qid", "type", "question", "reference_answer", "evidence_title", "page", "section", "evidence_span", "human_ok(y/n)"])
            for q in test:
                for e in q["evidences"]:
                    w.writerow([q["qid"], q["type"], q["question"], q["reference_answer"], e["title"], e["page"], e["section"], e["span"], ""])
        print("test benchmark:", Counter(q["type"] for q in test), flush=True)

        dev = build("dev", units(dev_docs), DEV_COUNTS, 5.0, rng, emb, llm, docs=dev_docs)
        for i, q in enumerate(dev):
            q["qid"] = f"d{i:03d}"
        save_jsonl(dev, settings.eval_dir / "benchmark_dev.jsonl")
        print("dev:", Counter(q["type"] for q in dev), flush=True)


    if not args.skip_train:
        tu = units(train_docs)
        by_paper = {}
        for u in tu:
            by_paper.setdefault(u.paper_id, []).append(u)
        train = []
        for t in ("factual", "methodological", "mathematical", "results", "limitations"):
            pool = []
            for pid, us in by_paper.items():
                cand = sample_units(us, t, max(2, args.train_per_paper // 5), rng, per_paper_cap=args.train_per_paper)
                pool.extend(cand)
            gen = generate_qa(pool, t, llm, temperature=0.5)
            print(f"[train] {t}: {len(gen)} from {len(pool)} units", flush=True)
            train.extend(gen)
        n0 = len(train)
        train = dedupe_questions(quality_filter(train, llm, min_spec=3))
        print(f"[train] quality filter: {n0} -> {len(train)}", flush=True)
        for i, q in enumerate(train):
            q["qid"] = f"r{i:05d}"
        save_jsonl(train, settings.eval_dir / "train_qa.jsonl")
        print("train QA:", len(train), Counter(q["type"] for q in train), flush=True)
    print("LLM usage:", meter.snapshot())


if __name__ == "__main__":
    main()
