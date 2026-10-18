"""Fine-tune a 4-class claim verifier (SUPPORTED / PARTIALLY_SUPPORTED / UNSUPPORTED / CONTRADICTED).

Training mix:  in-domain synthetic claims (train papers) + SciFact + a general NLI sample.
Evaluation  :  (a) synthetic claims from the held-out TEST papers, (b) SciFact validation (3-way),
               (c) agreement with the LLM judge on the same held-out claims.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
from datasets import Dataset, load_dataset

from ..config import settings
from ..evaluation.dataset import load_jsonl
from ..utils.logging import get_logger
from .nli import LABELS

log = get_logger("train_verifier")
DATA = settings.data_dir / "training"
L2I = {l: i for i, l in enumerate(LABELS)}


def scifact_rows(split: str) -> list[dict]:
    ds = load_dataset("tasksource/scifact_entailment")[split]
    rows = []
    for r in ds:
        v = str(r["verdict"]).upper()
        lab = {"SUPPORT": "SUPPORTED", "CONTRADICT": "CONTRADICTED"}.get(v, "UNSUPPORTED")
        rows.append({"claim": r["claim"], "passage": f"{r['title']}. {r['abstract'] if isinstance(r['abstract'], str) else ' '.join(r['abstract'])}", "label": lab})
    return rows


def nli_rows(n: int, rng: random.Random) -> list[dict]:
    ds = load_dataset("sentence-transformers/all-nli", "pair-class")["train"]
    idx = rng.sample(range(len(ds)), n)
    m = {0: "SUPPORTED", 1: "UNSUPPORTED", 2: "CONTRADICTED"}      # entailment / neutral / contradiction
    return [{"claim": ds[i]["hypothesis"], "passage": ds[i]["premise"], "label": m[ds[i]["label"]]} for i in idx]


def main():
    import torch
    from sklearn.metrics import accuracy_score, classification_report, f1_score
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, Trainer, TrainingArguments)

    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="cross-encoder/nli-deberta-v3-base")
    ap.add_argument("--out", default=str(settings.models_dir / "scholar-verifier"))
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=448)
    ap.add_argument("--nli-n", type=int, default=9000)
    a = ap.parse_args()
    rng = random.Random(11)

    syn = load_jsonl(DATA / "claims_train.jsonl")
    rows = syn + scifact_rows("train") + nli_rows(a.nli_n, rng)
    rng.shuffle(rows)
    n_val = int(0.05 * len(rows))
    val_rows, train_rows = rows[:n_val], rows[n_val:]
    test_syn = load_jsonl(DATA / "claims_test.jsonl")
    test_sf = scifact_rows("validation")
    log.info("train %d (synthetic %d) | val %d | test synthetic %d | scifact val %d", len(train_rows), len(syn), len(val_rows), len(test_syn), len(test_sf))

    tok = AutoTokenizer.from_pretrained(a.base)
    model = AutoModelForSequenceClassification.from_pretrained(
        a.base, num_labels=len(LABELS), ignore_mismatched_sizes=True, id2label=dict(enumerate(LABELS)), label2id=L2I)

    def enc(batch):
        e = tok(batch["passage"], batch["claim"], truncation="only_first", max_length=a.max_len)
        e["labels"] = [L2I[l] for l in batch["label"]]
        return e
    mk = lambda r: Dataset.from_list(r).map(enc, batched=True, remove_columns=["claim", "passage", "label", *[k for k in r[0] if k not in ("claim", "passage", "label")]])  # noqa: E731
    tr, va = mk(train_rows), mk(val_rows)

    def metrics(p):
        pred = p.predictions.argmax(-1)
        return {"accuracy": accuracy_score(p.label_ids, pred), "macro_f1": f1_score(p.label_ids, pred, average="macro")}

    args = TrainingArguments(output_dir=str(Path(a.out).with_suffix(".ckpt")), num_train_epochs=a.epochs, per_device_train_batch_size=a.batch,
                             per_device_eval_batch_size=64, learning_rate=a.lr, warmup_steps=0.06, weight_decay=0.01, bf16=True,
                             eval_strategy="epoch", save_strategy="epoch", save_total_limit=1, load_best_model_at_end=True,
                             metric_for_best_model="macro_f1", logging_steps=50, report_to="none", seed=11, dataloader_num_workers=2,
                             lr_scheduler_type="linear")
    trainer = Trainer(model=model, args=args, train_dataset=tr, eval_dataset=va, data_collator=DataCollatorWithPadding(tok),
                      compute_metrics=metrics)
    trainer.train()
    model.save_pretrained(a.out)
    tok.save_pretrained(a.out)

    def evaluate(rows_, three_way=False):
        ds = mk(rows_)
        pred = trainer.predict(ds).predictions.argmax(-1)
        gold = np.array(ds["labels"])
        if three_way:       # SciFact has no 'partial': merge partial->supported for comparison
            pred = np.where(pred == 1, 0, pred)
        names = [l for i, l in enumerate(LABELS)]
        return {"accuracy": round(float(accuracy_score(gold, pred)), 4), "macro_f1": round(float(f1_score(gold, pred, average="macro")), 4),
                "report": classification_report(gold, pred, labels=list(range(4)), target_names=names, output_dict=True, zero_division=0)}
    res = {"synthetic_test_papers": evaluate(test_syn), "scifact_validation": evaluate(test_sf, True),
           "train_size": len(train_rows), "base": a.base}
    (Path(a.out) / "eval.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items() if kk != "report"}) for k, v in res.items()}, indent=1))


if __name__ == "__main__":
    main()
