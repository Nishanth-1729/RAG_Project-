# ScholarAgent – architecture & design decisions

## 1. System map

```
                              USER  (Streamlit UI :8501  |  REST :8100  |  CLI Main.py)
                                │
                     ┌──────────▼───────────┐
                     │   SUPERVISOR (LangGraph state machine)   │  plans, then repeatedly chooses the next action
                     └──┬────┬────┬────┬────┬────┬────┬────┬───┘  from what the state still lacks (search again if too few papers)
                        │    │    │    │    │    │    │    │
        search ─────────┘    │    │    │    │    │    │    └── report
        (Search Agent)       │    │    │    │    │    └─────── answer (+ Verifier)
                  ingest ────┘    │    │    │    └──────────── find_gaps (Gap Agent)
        (download → parse →       │    │    └───────────────── build_graph (citation graph)
         chunk → embed → index)   │    └────────────────────── compare (Comparison Agent)
                       analyze ───┘   (Paper Analyst Agent, evidence-linked JSON per paper)

 ┌────────────────────────────── Hybrid RAG engine (src/scholaragent/retrieval) ─────────────────────────────┐
 │ question ─┬─► dense: bge-base embeddings ─► Qdrant (cosine, payload filters) ─ top-30 ─┐                  │
 │           └─► sparse: BM25 (own implementation, scientific tokenizer)           ─ top-30 ─┤                  │
 │                                            Reciprocal Rank Fusion (k=60) ◄──────────────┘                  │
 │                                                    │ top-30                                                │
 │                                      cross-encoder reranker (bge-reranker-v2-m3 / fine-tuned) ─► top-8      │
 └────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
        ▲ chunks with paper / section / page / DOI metadata            │
        │                                                              ▼
 PDF → PyMuPDF blocks → sections (abstract … references) → chunker   grounded answer  ──►  VERIFIER
 (page-aware, 2-column aware, header/footer removal, equations,      (numbered evidence,     claim extraction → evidence → SUPPORTED / PARTIAL /
  tables, run-in headings, ligature/math-glyph cleaning)              [n] citations)         UNSUPPORTED / CONTRADICTED → re-retrieve → fix / delete
```

## 2. Modules

| Package | Responsibility |
|---|---|
| `ingestion/` | `pdf_parser` (PyMuPDF, fonts, columns, equations, tables), `section_parser` (heading detection, canonical sections), `metadata` (title/authors/DOI/arXiv/year), `chunker` (fixed / section-aware / semantic), `pipeline` (cached parse) |
| `retrieval/` | `embedder`, `sparse` (BM25), `vectorstore` (Qdrant + GPU in-memory store), `fusion` (RRF), `reranker`, `hybrid` (retriever), `library` (index orchestration, dedup, incremental updates) |
| `search/` | arXiv, Semantic Scholar, Crossref, OpenAlex clients; cached/rate-limited HTTP with circuit breakers; deduplication (DOI → arXiv id → title → fuzzy); relevance scoring; open-access PDF fetcher |
| `rag/` | context builder, grounded answer generation, citation parsing/validation, query rewriting |
| `agents/` | supervisor, search, paper analyst, comparison, gap, verifier, graph, report agents |
| `database/` | SQLAlchemy models + store (papers, chunks, collections, analyses, runs, graph edges) – SQLite by default, PostgreSQL via `DATABASE_URL` |
| `evaluation/` | benchmark generation, retrieval/answer/agent metrics, experiment runners, error analysis |
| `training/` | hard-negative pair mining, fine-tuning of embedder, reranker and claim verifier |
| `api/` | FastAPI app + background research jobs |

## 3. Key design decisions

* **Evidence never gets lost.** A chunk is never "just text": it carries paper, authors, year, DOI, section, page range, and flags for equations/tables. Answers cite `[n]`; each `n` resolves to `(paper, §section, page, passage)`. Citations that do not point at supplied evidence are stripped.
* **Section-aware chunking + contextual headers.** Chunks never cross section boundaries; the text that is *embedded* is prefixed with `title | section`. Both choices are ablated in the evaluation.
* **Hybrid retrieval is written from scratch** (BM25, RRF) before any framework is used, so retrieval failures can be diagnosed. LangGraph is used only for agent orchestration.
* **One global index.** Papers are parsed/embedded once; collections and research runs are sets of paper ids with payload filters, so a paper never costs compute twice.
* **Local GPU LLM.** vLLM serves Qwen2.5-7B-Instruct (FP8) through the OpenAI API, so the same code runs against any hosted OpenAI-compatible model by changing `.env`. All LLM calls are schema-constrained (guided JSON), disk-cached at temperature 0, traced (tokens, latency) and retried.
* **Verifier-in-the-loop.** Claims are extracted from drafts, judged against retrieved evidence, re-retrieved with rewritten queries when unsupported, corrected when contradicted, and deleted when no evidence exists; the final text is re-composed only from verified claims.
* **Gap finding is conservative.** Limitations/future-work (evidence-linked) → embedding clusters → recurring across ≥2 papers → candidate direction → *targeted novelty search* in the literature APIs → LLM judge "already addressed?" → reject or report as **potential research direction** with confidence and closest work. The system never claims "nobody has done X".
* **Failure isolation.** Corrupt/scanned PDFs, API rate limits (circuit breakers), missing abstracts/DOIs, malformed LLM JSON, empty retrieval, duplicate papers, unavailable full text (falls back to abstract-only) never abort a run.
* **Substitutions forced by this machine** (no Docker, no sudo): native Qdrant binary instead of the container; SQLite instead of PostgreSQL (switch with `DATABASE_URL`); `docker/docker-compose.yml` contains the container deployment.

## 4. Data flow of a Deep Research run

1. **plan** – LLM classifies intent (survey / comparison / qa / gaps) and decides whether to search.
2. **search** – query expansion (LLM) → arXiv / Semantic Scholar / Crossref (/ OpenAlex) → dedup → relevance score `S = .50 semantic + .15 title + .15 abstract + .10 recency + .10 citations/year`.
3. **ingest** – download open-access PDFs in rank order until the target is met, parse, chunk, embed, index; abstract-only fallback.
4. **analyze** – per paper: retrieval *inside* the paper per field group → schema-constrained extraction (problem, method, equations, complexity, datasets, baselines, metrics, results, limitations, future work, claims) → every quote verified against the source chunk, unverifiable items dropped.
5. **compare / build_graph / find_gaps** – comparison table + landscape, citation graph (reference lists matched by arXiv id/DOI/title) + entity nodes, novelty-checked research directions.
6. **report** – sections written from evidence-linked facts, claim-verified and self-corrected, with one global evidence index, search methodology, verification statistics, BibTeX.

## 5. Decisions log (what the experiments changed)

| Decision | Evidence (dev split, then confirmed on test) |
|---|---|
| **Relative-score fusion** instead of Reciprocal Rank Fusion as the hybrid default (weights 0.4 dense / 0.6 BM25) | dev R@5 0.875 vs 0.850, MRR 0.741 vs 0.730; test R@5 0.87 vs 0.79 (RRF discards score magnitudes). RRF stays available (`FUSION=rrf`) |
| **bge-m3** as dense model | best zero-shot dev result (hybrid+rerank R@5 0.917); bge-base / MiniLM / e5 are within noise once BM25 + reranker are present |
| **Section-aware chunks, 384 tokens, contextual header** | beats fixed windows at equal size; fixed-1024 collapses under the 512-token reranker window; semantic chunking worst |
| **Fine-tuned reranker** as default | dev parity (R@5 0.933 vs 0.917, MRR 0.874 vs 0.872), test gain in top-1 (R@1 0.87 vs 0.81) |
| **NLI-first verifier** (fine-tuned DeBERTa decides when confident, LLM breaks ties) | on held-out claims the zero-shot 7B judge reaches 67 % accuracy vs 93 % for the fine-tuned model; bad-claim detection 84.5 % vs 97 % |
| **No query rewriting by default** | rewriting lowered answer correctness (0.605→0.575 with verification on); kept as an option |
| **Planner with contrastive examples + coherence check** | intent accuracy 87.5 % → 100 % on 24 queries; nonsense queries are declined instead of producing a confident report |
| **One GPU thread for all CUDA work** | with 24 worker threads each thread allocated its own cuBLAS workspace (~1.5 GB of hidden VRAM) – OOM next to the LLM server |
| **vLLM compact-JSON structured outputs** (`disable_any_whitespace`) | the default grammar sometimes emitted endless whitespace until `max_tokens`, silently dropping 3-45 % of structured calls |
