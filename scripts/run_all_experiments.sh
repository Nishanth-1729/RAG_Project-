#!/usr/bin/env bash
# Reproduce every number in docs/RESULTS.md (needs the corpus + benchmark; see README "Reproducing").
# Phase A runs with the LLM server stopped (clean GPU timings); phase B starts the LLM for answer/agent evaluation.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
PY=.venv/bin/python
./scripts/start_qdrant.sh
EMB_FT="models/scholar-embed"; RR_FT="models/scholar-rerank"
EXTRA_E=(); EXTRA_R=()
[ -d "$EMB_FT" ] && EXTRA_E+=(--extra-embedder "$EMB_FT")
[ -d "$RR_FT" ] && EXTRA_R+=(--extra-reranker "$RR_FT")

echo "== A. retrieval experiments (dev for model selection, test for reporting)"
$PY -m scholaragent.evaluation.run_experiments --split dev  --only retrieval,rerankers,embedders "${EXTRA_E[@]}" "${EXTRA_R[@]}"
$PY -m scholaragent.evaluation.run_experiments --split test --only retrieval,chunking,rerankers,embedders,ablation "${EXTRA_E[@]}" "${EXTRA_R[@]}"

echo "== B. LLM-based evaluation"
./scripts/start_llm.sh
$PY -m scholaragent.evaluation.run_answer_eval --split test
$PY -m scholaragent.evaluation.run_verifier_eval
$PY -m scholaragent.evaluation.run_agent_eval
$PY -m scholaragent.evaluation.run_error_analysis
$PY scripts/render_results.py
