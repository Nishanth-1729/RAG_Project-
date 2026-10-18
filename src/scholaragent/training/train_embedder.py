"""Fine-tune the dense retriever (bi-encoder) on synthetic scientific QA with hard negatives.

MultipleNegativesRankingLoss (in-batch + mined hard negatives), cached gradients for a large effective batch,
early model selection on the DEV papers (disjoint from train and test papers).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import Dataset

from ..config import settings
from ..evaluation.dataset import load_jsonl
from ..retrieval.embedder import _prefix, _QUERY_PREFIX, _DOC_PREFIX
from ..utils.logging import get_logger

log = get_logger("train_embed")
DATA = settings.data_dir / "training"


def main():
    import torch
    from sentence_transformers import (SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments, losses)
    from sentence_transformers.evaluation import InformationRetrievalEvaluator
    from sentence_transformers.training_args import BatchSamplers

    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=settings.embed_model)
    ap.add_argument("--out", default=str(settings.models_dir / "scholar-embed"))
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch", type=int, default=96)
    ap.add_argument("--mini-batch", type=int, default=24)
    ap.add_argument("--max-len", type=int, default=384)
    ap.add_argument("--n-neg", type=int, default=1, help="hard negatives per query (1 keeps memory low)")
    ap.add_argument("--freeze-embeddings", action="store_true", help="freeze the token-embedding matrix (saves memory for large-vocab models like bge-m3)")
    a = ap.parse_args()

    qp, dp = _prefix(a.base, _QUERY_PREFIX), _prefix(a.base, _DOC_PREFIX)
    rows = load_jsonl(DATA / "pairs_train.jsonl")
    cols = {"anchor": [], "positive": []}
    for k in range(a.n_neg):
        cols[f"negative_{k + 1}"] = []
    for r in rows:
        cols["anchor"].append(qp + r["query"])
        cols["positive"].append(dp + r["pos"])
        for k in range(a.n_neg):
            cols[f"negative_{k + 1}"].append(dp + r["negs"][k % len(r["negs"])])
    ds = Dataset.from_dict(cols).shuffle(seed=1)
    log.info("train pairs: %d", len(ds))

    dq = json.loads((DATA / "dev_queries.json").read_text())
    dc = json.loads((DATA / "dev_corpus.json").read_text())["corpus"]
    ev = InformationRetrievalEvaluator(
        queries={k: qp + v for k, v in dq["queries"].items()}, corpus={k: dp + v for k, v in dc.items()},
        relevant_docs={k: set(v) for k, v in dq["relevant"].items()}, name="dev", mrr_at_k=[10], ndcg_at_k=[10],
        accuracy_at_k=[1, 5, 10], precision_recall_at_k=[5], show_progress_bar=False, batch_size=64)

    model = SentenceTransformer(a.base, device="cuda", model_kwargs={"torch_dtype": torch.bfloat16}, trust_remote_code=True)
    model.max_seq_length = a.max_len
    if a.freeze_embeddings:
        for n, prm in model.named_parameters():
            if "word_embeddings" in n:
                prm.requires_grad = False
    log.info("baseline dev: %s", ev(model))
    loss = losses.CachedMultipleNegativesRankingLoss(model, mini_batch_size=a.mini_batch)
    args = SentenceTransformerTrainingArguments(
        output_dir=str(Path(a.out).with_suffix(".ckpt")), num_train_epochs=a.epochs, per_device_train_batch_size=a.batch,
        learning_rate=a.lr, warmup_steps=0.1, bf16=True, batch_sampler=BatchSamplers.NO_DUPLICATES,
        eval_strategy="steps", eval_steps=max(10, len(ds) // a.batch // 2), save_strategy="steps",
        save_steps=max(10, len(ds) // a.batch // 2), save_total_limit=2, load_best_model_at_end=True,
        metric_for_best_model="eval_dev_cosine_ndcg@10", logging_steps=10, report_to="none", seed=1,
        dataloader_num_workers=2, lr_scheduler_type="linear", weight_decay=0.01)
    trainer = SentenceTransformerTrainer(model=model, args=args, train_dataset=ds, loss=loss, evaluator=ev)
    trainer.train()
    final = ev(model)
    log.info("fine-tuned dev: %s", final)
    model.save_pretrained(a.out)
    (Path(a.out) / "train_info.json").write_text(json.dumps({"base": a.base, "train_pairs": len(ds), "epochs": a.epochs,
                                                              "dev_metrics": {k: float(v) for k, v in final.items()}}, indent=1))
    print("saved", a.out)


if __name__ == "__main__":
    main()
