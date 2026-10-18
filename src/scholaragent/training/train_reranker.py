"""Fine-tune the cross-encoder reranker on (query, passage, label) with mined hard negatives."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from datasets import Dataset

from ..config import settings
from ..evaluation.dataset import load_jsonl
from ..utils.logging import get_logger

log = get_logger("train_rerank")
DATA = settings.data_dir / "training"


def main():
    import torch
    from sentence_transformers.cross_encoder import CrossEncoder, CrossEncoderTrainer, CrossEncoderTrainingArguments
    from sentence_transformers.cross_encoder.evaluation import CrossEncoderRerankingEvaluator
    from sentence_transformers.cross_encoder.losses import BinaryCrossEntropyLoss

    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=settings.rerank_model)
    ap.add_argument("--out", default=str(settings.models_dir / "scholar-rerank"))
    ap.add_argument("--epochs", type=float, default=1)
    ap.add_argument("--lr", type=float, default=1.5e-5)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=320)
    ap.add_argument("--n-neg", type=int, default=3)
    ap.add_argument("--max-pairs", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(3)

    rows = load_jsonl(DATA / "pairs_train.jsonl")
    q, p, y = [], [], []
    for r in rows:
        q.append(r["query"]); p.append(r["pos_plain"]); y.append(1.0)
        for n in r["negs_plain"][: a.n_neg]:
            q.append(r["query"]); p.append(n); y.append(0.0)
    idx = list(range(len(q)))
    rng.shuffle(idx)
    if a.max_pairs:
        idx = idx[: a.max_pairs]
    ds = Dataset.from_dict({"query": [q[i] for i in idx], "passage": [p[i] for i in idx], "label": [y[i] for i in idx]})
    log.info("train examples: %d (pos %.0f%%)", len(ds), 100 * sum(ds["label"]) / len(ds))

    dev = load_jsonl(DATA / "pairs_dev.jsonl")
    # realistic dev task: rerank the positive among the (unfiltered) top-20 hybrid candidates
    samples = [{"query": r["query"], "positive": [r["pos_plain"]], "negative": r.get("dev_cands_plain") or r["negs_plain"]} for r in dev]
    ev = CrossEncoderRerankingEvaluator(samples, at_k=10, name="dev", batch_size=16, show_progress_bar=False, mrr_at_k=10)

    model = CrossEncoder(a.base, num_labels=1, max_length=a.max_len, device="cuda", model_kwargs={"torch_dtype": torch.float32}, trust_remote_code=True)
    for n, prm in model.model.named_parameters():      # the 250k-token embedding matrix is ~45% of XLM-R-large: freezing it saves ~3 GB of optimizer state
        if "word_embeddings" in n:
            prm.requires_grad = False
    base_m = ev(model)
    log.info("baseline dev: %s", base_m)
    loss = BinaryCrossEntropyLoss(model, pos_weight=torch.tensor(float(a.n_neg)))
    args = CrossEncoderTrainingArguments(
        output_dir=str(Path(a.out).with_suffix(".ckpt")), num_train_epochs=a.epochs, per_device_train_batch_size=a.batch,
        gradient_accumulation_steps=a.grad_accum, learning_rate=a.lr, warmup_steps=0.1, bf16=True,
        eval_strategy="steps", eval_steps=200, save_strategy="steps", save_steps=200, save_total_limit=2,
        load_best_model_at_end=True, metric_for_best_model="eval_dev_ndcg@10", logging_steps=20, report_to="none",
        seed=3, dataloader_num_workers=2, weight_decay=0.01)
    trainer = CrossEncoderTrainer(model=model, args=args, train_dataset=ds, loss=loss, evaluator=ev)
    trainer.train()
    final = ev(model)
    log.info("fine-tuned dev: %s", final)
    model.save_pretrained(a.out)
    (Path(a.out) / "train_info.json").write_text(json.dumps({"base": a.base, "examples": len(ds), "epochs": a.epochs,
                                                              "baseline_dev": {k: float(v) for k, v in base_m.items()},
                                                              "dev_metrics": {k: float(v) for k, v in final.items()}}, indent=1))
    print("saved", a.out)


if __name__ == "__main__":
    main()
