# Training the ScholarAgent models (all on the GPU)

Three models were fine-tuned. Everything is reproducible with the commands in the README; the data are generated from the
paper corpus by ScholarAgent itself.

## Data (split by paper – no passage of a test paper is ever seen in training)

| Split | Papers | Use |
|---|---|---|
| train | 125 (arXiv, scientific-ML / numerical analysis) | training data |
| dev | 15 | early stopping, **model/configuration selection** |
| test | 40 (ML for numerical linear algebra – the headline topic) | final reporting only |

* **Retrieval pairs** – 4,864 LLM-generated questions (Qwen2.5-7B, guided JSON) from paragraphs of the train papers, quality-audited
  (specificity, answer correctness, self-containment, no “this paper” references). For each question: *positive* = the chunk that contains the
  verbatim evidence span; *hard negatives* = top hybrid-retrieved chunks that are not relevant, **filtered with a teacher cross-encoder** so that
  passages which actually answer the question are not used as negatives (4,445 pairs survive).
* **Claim-verification data** – 3,587 train paragraphs → 1,333 usable → 5,332 labelled rows: a faithful claim (SUPPORTED), the same claim plus an
  unsupported detail (PARTIALLY_SUPPORTED), a minimally edited false claim (CONTRADICTED), and a faithful claim paired with an unrelated paragraph
  (UNSUPPORTED); mixed with SciFact (919 human-annotated claims) and 9,000 general NLI pairs (AllNLI).

## Models

| Model | Base | Method | Hyper-parameters | Time |
|---|---|---|---|---|
| `models/scholar-embed` | `BAAI/bge-base-en-v1.5` | bi-encoder, CachedMultipleNegativesRankingLoss with 1 mined hard negative, in-batch negatives | 3 epochs, batch 64 (cached gradients), lr 2e-5, bf16 | ~6 min |
| `models/scholar-embed-m3` | `BAAI/bge-m3` | same; word-embedding matrix frozen (memory) | 3 epochs, batch 64, lr 1.5e-5, bf16 | ~17 min |
| `models/scholar-rerank` | `BAAI/bge-reranker-v2-m3` | cross-encoder, binary cross-entropy, 1 positive : 3 hard negatives (17,656 examples); word embeddings frozen | 1 epoch, effective batch 32, lr 1.5e-5, bf16 | ~7 min |
| `models/scholar-verifier` | `cross-encoder/nli-deberta-v3-base` | 4-way sequence classification (SUPPORTED / PARTIALLY_SUPPORTED / UNSUPPORTED / CONTRADICTED) on 14.5k examples | 3 epochs, batch 16, lr 2e-5, bf16 | ~15 min |

All training metrics (baseline vs fine-tuned on the dev split, and the held-out evaluations) are in `models/*/train_info.json`,
`models/scholar-verifier/eval.json` and in [RESULTS.md](RESULTS.md). The honest summary:

* **Embedders**: clear gains for dense-only retrieval (e.g. bge-base: Recall@5 0.62 → 0.77 on the test papers, dev MRR@10 0.563 → 0.658; bge-m3: dev MRR@10 0.676 → 0.717).
  In the full hybrid + reranker pipeline the gain disappears (BM25 and the reranker already cover most of it), so the **default embedder is zero-shot bge-m3**, which won the dev comparison.
* **Reranker**: dev (pair-level) MRR@10 0.864 → 0.889; end-to-end the dev split shows parity and the test split shows a gain in top-1 accuracy, so the fine-tuned reranker is the default
  (note: chosen after seeing both splits – see the evaluation notes in the README).
* **Verifier**: the largest effect. On held-out claims from the test papers the fine-tuned NLI model reaches ~93 % accuracy versus ~67 % for the zero-shot 7B LLM judge;
  on human-annotated SciFact it reaches 83 % (3-way) versus ~69 % for the LLM judge. It is used as an independent judge in the evaluation and (optionally) as a veto in the verifier agent.

## Why training runs with the LLM stopped
The 7B model served by vLLM occupies ~11–12 GB of the 16 GB card; the trainers need 6–12 GB, so `scripts/stop_stack.sh` is run first.
