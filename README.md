# ScholarAgent: An Evidence-Grounded Agentic RAG System for Scientific Literature Discovery, Analysis and Research-Gap Exploration

> Give it a research question. It finds papers, reads them, compares them, answers with exact page/section citations,
> verifies its own claims against the sources, proposes novelty-checked research directions, and writes a report.
> Everything (LLM, embeddings, reranking, training) runs **on your GPU**.

```
Hybrid retrieval (dense + BM25 + RRF)  +  cross-encoder reranking  +  scientific PDF understanding
+  multi-agent research (LangGraph)  +  claim-level verification & self-correction  +  research-gap exploration
```

## What you get

| | |
|---|---|
| **Two modes** | **My Papers** – upload 5–50 PDFs and ask/compare/analyse; **Deep Research** – only a question: the agent searches arXiv / Semantic Scholar / Crossref, downloads open-access PDFs, analyses them and writes a report |
| **Evidence everywhere** | every answer sentence carries `[n]` → paper · section · page · exact passage; unsupported or invalid citations are removed |
| **Agents** | Supervisor (plans, decides next action, may search again) · Search · Paper Analyst · Comparison · Gap · Verifier · Graph · Report |
| **Verifier** | claims → evidence → `SUPPORTED / PARTIALLY_SUPPORTED / UNSUPPORTED / CONTRADICTED` → re-retrieve → correct or delete → claim–evidence matrix |
| **Research gaps** | limitations → clusters → recurring problems → candidate direction → *targeted novelty search* → reject or report as a **potential research direction** with confidence and closest work |
| **Knowledge base** | papers, chunks, structured analyses, citation graph (+ Neo4j Cypher export) |
| **Trained models** | fine-tuned embedder, reranker and claim-verifier (NLI) trained on synthetic scientific QA with mined hard negatives |
| **Evaluation** | 100-question benchmark, Recall@K / Precision@K / MRR / NDCG, chunking + reranker + embedder experiments, ablation, answer faithfulness / citation accuracy / hallucination rate, agent metrics, error analysis – all measured, see [docs/RESULTS.md](docs/RESULTS.md) |
| **Interfaces** | Streamlit UI · FastAPI (`/docs`) · CLI (`Main.py`) · Docker Compose |

## Quick start (this machine is already set up)

```bash
cd /home/user31/AgriBrain
./scripts/run_stack.sh                    # Qdrant -> local LLM on the GPU -> API :8100 -> UI :8501
# open http://localhost:8501      (API docs: http://localhost:8100/docs)
./scripts/stop_stack.sh                   # stop everything / free the GPU
```

CLI examples

```bash
.venv/bin/python Main.py status
.venv/bin/python Main.py ingest data/raw_papers/*.pdf
.venv/bin/python Main.py ask "How do learned preconditioners reduce GMRES iterations?" --verify
.venv/bin/python Main.py research "What are recent ML methods for accelerating numerical linear algebra solvers?" --papers 12 --from 2021
```

### From scratch on another Linux + NVIDIA machine
```bash
git clone <this repo> && cd AgriBrain && ./scripts/setup.sh      # creates 2 Python-3.11 envs, installs torch(cu128)/vLLM/Qdrant, downloads models
cp .env.example .env                                              # optional: API keys, hosted LLM instead of local
./scripts/run_stack.sh
```
To use a hosted model instead of the local GPU LLM set `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL` in `.env` (any OpenAI-compatible endpoint) and start with `NO_LLM=1 ./scripts/run_stack.sh`.

## What was installed here (decisions are documented in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md))

| Component | Choice |
|---|---|
| GPU | NVIDIA RTX 5060 Ti 16 GB (Blackwell, sm_120) → PyTorch 2.11 + CUDA 12.8 wheels |
| Languages | Python 3.11 (two isolated conda envs: `.venv` app/training, `.venv-llm` vLLM) |
| LLM | `Qwen2.5-7B-Instruct`, FP8 weights on the fly, served by **vLLM 0.31** (OpenAI API, guided-JSON decoding) |
| Embeddings | `BAAI/bge-base-en-v1.5` (fp16, GPU) → fine-tuned `models/scholar-embed` |
| Reranker | `BAAI/bge-reranker-v2-m3` (fp16, GPU) → fine-tuned `models/scholar-rerank` |
| Claim verifier | `cross-encoder/nli-deberta-v3-base` → fine-tuned 4-class `models/scholar-verifier` |
| Vector DB | **Qdrant 1.19** native binary (no Docker available here) |
| Metadata DB | SQLAlchemy + SQLite (PostgreSQL via `DATABASE_URL`; compose file included) |
| Agents | LangGraph |
| PDF | PyMuPDF |
| UI / API | Streamlit / FastAPI |

GPU memory budget (16 GB): LLM ≈ 12.4 GB (weights 8.2 + KV cache 2.5) · embedder ≈ 0.4 GB · reranker ≈ 1.2 GB. Training jobs run with the LLM server stopped.

## Repository layout

```
Main.py                      CLI
src/scholaragent/
  ingestion/   pdf_parser, section_parser, metadata, chunker, pipeline
  retrieval/   embedder, sparse(BM25), vectorstore(Qdrant|GPU), fusion(RRF), reranker, hybrid, library
  search/      arxiv, semantic_scholar, crossref, openalex, dedup, ranking, fetcher, service
  rag/         context_builder, answer, citations, query, pipeline
  agents/      supervisor, search_agent, paper_agent, comparison_agent, gap_agent, verifier_agent, graph_agent, report_agent
  evaluation/  dataset, retrieval_metrics, answer_metrics, agent_metrics, harness, run_experiments, run_answer_eval, run_agent_eval, run_verifier_eval, run_error_analysis
  training/    build_pairs, train_embedder, train_reranker, train_verifier, nli
  llm/         OpenAI-compatible client (cache, JSON schema, tracing)
  database/    SQLAlchemy models + store
  api/         FastAPI app + background jobs
frontend/app.py             Streamlit UI
scripts/                    setup, run/stop stack, corpus + benchmark + claim-data generation, results renderer
data/                       raw_papers/, parsed/, evaluation/ (corpus, benchmark_test/dev.jsonl, train_qa.jsonl, audit CSV), training/
models/                     fine-tuned checkpoints
results/                    measured metrics (JSON) + charts
docker/                     Dockerfile + docker-compose.yml
tests/                      unit + integration tests (pytest)
```

## Using it

### Deep Research (UI)
Enter a question, choose the period, the maximum number of papers and the sources, and press **Start research**.
The progress panel shows the agents' work (search → ingest → analyse → compare → graph → gaps → report → verify);
the dashboard then offers **Overview · Papers · Methods · Comparison · Citation graph · Research directions · Ask · Report · Agent trace**.
Reports export to Markdown, HTML, LaTeX, BibTeX and JSON.

### My Papers
Upload PDFs → **Ask** (choose retrieval mode, rewriting, verification) or **Analyse & compare**. Every answer has an *Evidence* panel
(paper, section, pages, passage) and, when verification is on, a claim–evidence matrix.

### REST API (selected)
| Endpoint | |
|---|---|
| `POST /papers/upload` | index PDFs (corrupt/scanned files are reported, never fatal) |
| `POST /ask` | `{question, collection_id\|paper_ids, mode, k, rewrite, verify}` → answer + evidence + verification |
| `POST /search/papers` | literature search across sources (dedup + relevance score) |
| `POST /research` → `GET /runs/{id}` | start a background research run, poll progress events |
| `GET /runs/{id}/report.{md,html,tex,bib,json}` · `GET /runs/{id}/graph?fmt=cypher` | exports |
| `GET /evaluation/results` | measured metrics for the dashboard |
| `GET /health` | LLM / GPU / Qdrant / library status |

Sample questions to try on the bundled corpus (40 ML-for-numerical-linear-algebra papers are indexed):
*“How do learned preconditioners reduce the number of GMRES iterations, and what do they cost to train?” ·
“Which papers use graph neural networks and for what purpose?” · “What are the common limitations of neural multigrid methods?”*

## Reproducing the data, training and evaluation

```bash
# 0) services
./scripts/start_qdrant.sh && ./scripts/start_llm.sh
# 1) corpora (180 arXiv papers: 40 for testing, 140 disjoint for training/dev) – uses ScholarAgent's own search layer
.venv/bin/python scripts/build_corpus.py
# 2) benchmark (100 test questions, dev set, training QA) and synthetic claim data  – LLM generated, quality + ambiguity filtered
.venv/bin/python scripts/make_benchmark.py --train-per-paper 130
.venv/bin/python scripts/make_claim_data.py
# 3) training (stop the LLM first: ./scripts/stop_stack.sh)
.venv/bin/python -m scholaragent.training.build_pairs
.venv/bin/python -m scholaragent.training.train_embedder
.venv/bin/python -m scholaragent.training.train_reranker
.venv/bin/python -m scholaragent.training.train_verifier
# 4) experiments
.venv/bin/python -m scholaragent.evaluation.run_experiments --split test
./scripts/start_llm.sh
.venv/bin/python -m scholaragent.evaluation.run_answer_eval
.venv/bin/python -m scholaragent.evaluation.run_verifier_eval
.venv/bin/python -m scholaragent.evaluation.run_agent_eval
.venv/bin/python -m scholaragent.evaluation.run_error_analysis
.venv/bin/python scripts/render_results.py        # -> docs/RESULTS.md
```

## Evaluation design (and its honest limits)

* **Corpus split by paper**: 40 *test* papers (the headline topic), 15 *dev* papers (model/config selection), 125 *train* papers. No test passage is ever seen in training.
* **Relevance labels are chunking-independent**: questions are generated from paragraphs with a verbatim *evidence span*; a retrieved chunk is relevant iff it contains that span. This makes fixed / section-aware / semantic chunking directly comparable.
* **Quality control of the benchmark**: deterministic filters (no “this paper / the passage” references, span must exist verbatim), an LLM specificity/answer-correctness audit, and an **ambiguity filter** that drops questions for which another retrieved passage also answers them. `data/evaluation/benchmark_test_audit.csv` lists every question with its evidence for manual verification (column `human_ok`).
* **Limits**: questions and the answer-level judge are produced by LLMs (Qwen2.5-7B) – not human annotated; self-preference bias is reduced by using a different judging prompt and by the independent fine-tuned NLI model, but not eliminated. 100 test questions give wide confidence intervals (reported). The corpus is one scientific subfield.

## Key results (all measured; test split = 40 papers, 100 questions; full tables in [docs/RESULTS.md](docs/RESULTS.md))

**Retrieval architectures** (Recall@5 [95% bootstrap CI] · MRR · latency per query on the RTX 5060 Ti)

| System | R@1 | R@5 | MRR | latency |
|---|---|---|---|---|
| Dense (bge-m3) | 0.49 | 0.74 [0.65, 0.82] | 0.60 | 8 ms |
| BM25 (own implementation) | 0.62 | 0.82 [0.74, 0.89] | 0.70 | 0.2 ms |
| Hybrid, Reciprocal Rank Fusion | 0.62 | 0.79 | 0.70 | 8 ms |
| Hybrid, relative-score fusion (chosen on dev) | 0.68 | 0.87 [0.80, 0.93] | 0.75 | 8 ms |
| Hybrid RSF + reranker (bge-reranker-v2-m3) | 0.81 | 0.92 | 0.86 | 259 ms |
| **Hybrid RSF + fine-tuned reranker (default)** | **0.87** | **0.93 [0.87, 0.98]** | **0.90** | 259 ms |

* **Hybrid beats either retriever alone, a cross-encoder adds the largest jump** (+6 pts R@5, +15 pts MRR over hybrid). Rank-only fusion (RRF) lost to score-aware fusion (RSF) – on the dev split too.
* **Ablation** (remove one component): BM25 −8 pts R@5 · reranker −5 pts R@5 / −10 pts MRR · dense −1 pt · section-aware chunking −7 pts R@1 / −4 pts MRR · contextual header −2 pts R@5.
* **Chunking**: section-aware chunks beat fixed windows at every size (hybrid+rerank R@5 0.92–0.94 vs 0.88–0.89); 1000-token chunks collapse once reranked (0.69) because the reranker window is 512 tokens; semantic chunking was worst (0.87).
* **Embedders**: zero-shot bge-m3 best on dev; with BM25 + reranker all embedders land within 0.89–0.93 R@5, i.e. the full pipeline is robust to the embedder choice.
* **Training** (details in [docs/TRAINING.md](docs/TRAINING.md)): fine-tuning lifts *dense-only* retrieval a lot (bge-base R@5 0.62→0.77, bge-m3 0.74→0.79) but not the full pipeline; the fine-tuned reranker lifts top-1 (R@1 0.81→0.87, MRR 0.855→0.900 on test; parity on dev); the fine-tuned claim verifier reaches **93 %** accuracy on held-out claims (zero-shot 7B LLM judge: 67 %) and 82 % on human-annotated SciFact (LLM: 69 %).

**Answers** (LLM-judged correctness; n = 100, so differences below ≈ ±0.09 are noise): retrieval quality is what moves correctness – Dense 0.51 → BM25 0.61 → Hybrid+Rerank 0.63. The verification / decomposition / rewriting add-ons did **not** produce a measurable correctness gain
(0.58–0.62) – they trade speed (2 s → 9–13 s per question) for claim-level filtering: with the NLI-first verifier 51–52 unsupported claims are removed and NLI-judged faithfulness rises 0.78 → 0.87 (partly circular: the same NLI model judges and filters), while the independent LLM judge shows no significant change.
Recommended modes: hybrid+rerank (fast) for exploration, +verification where every sentence must be trustworthy.

**Error analysis**: retrieval puts all evidence in the top-5 for 86 % of questions (7 % wrong paper, 6 % ranked below 5, 1 % parsing loss). For answers, 22 % of the failures are flagged as hallucinated claims by the (strict, noisy) judge and 12 % are over-cautious abstentions although the right passage was retrieved.
The weakest category is **comparison** questions (correctness 0.07–0.27): single-shot RAG retrieves both papers' evidence for only 40 % of them (60 % with per-entity query decomposition). The agentic *compare* workflow (per-paper analysis → comparison table) exists for exactly this case.

**Agents** (7 tasks: deep survey / comparison / QA, three My-Papers tasks, one nonsense query): 7/7 completed, tool-selection accuracy 1.0, required actions covered 1.0, 0 unnecessary calls, mean 91 s per task, 223k tokens ($0 – local model).
The nonsense query is now declined in 1 s; before the coherence check it produced a confident report on unrelated quantum-topology papers (found and fixed during evaluation). Planner intent accuracy on 28 hand-written queries: 28/28 (the first prompt scored 21/24 = 87.5 %).

**Honest notes**: (1) questions and the answer judge are LLM-generated (Qwen2.5-7B), the benchmark favours lexical matching (BM25 is unusually strong), and 100 questions give wide intervals; `data/evaluation/benchmark_test_audit.csv` is provided for human verification.
(2) The fine-tuned reranker became the default after seeing parity on dev and a gain on test. (3) Latency numbers are single-query, warmed-up, with the LLM server stopped.

Full measured results: **[docs/RESULTS.md](docs/RESULTS.md)** (generated by `scripts/render_results.py`).

## Failure handling

Corrupt/scanned PDFs, missing abstracts/DOIs, duplicate papers (DOI / arXiv id / title / content hash), API 429/timeouts (cached + rate-limited + circuit breakers), unavailable full text (abstract-only fallback),
empty retrieval, malformed LLM JSON (schema-constrained decoding + repair + retry), LLM timeouts – none aborts a run; problems are listed in the run's `errors`.

## Limitations & roadmap
* Scanned PDFs need OCR (not bundled; detected and reported). Equations are extracted as text, not LaTeX.
* Semantic Scholar without an API key is heavily rate-limited (set `SEMANTIC_SCHOLAR_API_KEY`); only open-access full text can be downloaded.
* Research gaps are *potential directions* checked against the searched literature, never claims of absolute novelty.
* Roadmap: React front-end, Neo4j-backed graph queries, OCR/Nougat for scanned papers and equations, multi-GPU/hosted-LLM profiles, human-verified benchmark release.

## Tests
`.venv/bin/python -m pytest` – parsing, chunking, BM25, RRF, dedup, citation validation, metrics, supervisor policy; integration tests for GPU retrieval, Qdrant and the LLM (skipped when a service is down).
