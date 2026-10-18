#!/usr/bin/env bash
# One-shot setup of everything ScholarAgent needs (Linux, NVIDIA GPU).  Re-runnable.
#   * Python 3.11 app env  (.venv)        - FastAPI, Streamlit, PyMuPDF, Qdrant client, SentenceTransformers, LangGraph, PyTorch (CUDA 12.8)
#   * Python 3.11 LLM env  (.venv-llm)    - vLLM (OpenAI-compatible server for the local LLM)
#   * Qdrant vector DB binary (bin/qdrant)
#   * Model weights (HuggingFace cache)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
CONDA="${CONDA_EXE:-$HOME/miniconda3/bin/conda}"
QDRANT_VERSION="${QDRANT_VERSION:-v1.19.2}"

echo "== 1/5 python environments"
[ -x .venv/bin/python ] || "$CONDA" create -y -q -p ./.venv -c conda-forge --override-channels python=3.11
[ -x .venv-llm/bin/python ] || "$CONDA" create -y -q -p ./.venv-llm -c conda-forge --override-channels python=3.11

echo "== 2/5 app dependencies (PyTorch cu128 supports Blackwell sm_120 as well as older GPUs)"
./.venv/bin/pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu128
./.venv/bin/pip install -q -r requirements.txt
./.venv/bin/pip install -q -e . --no-deps

echo "== 3/5 vLLM"
./.venv-llm/bin/pip install -q vllm

echo "== 4/5 Qdrant"
mkdir -p bin storage logs
if [ ! -x bin/qdrant ]; then
  curl -sL "https://github.com/qdrant/qdrant/releases/download/${QDRANT_VERSION}/qdrant-x86_64-unknown-linux-musl.tar.gz" | tar xz -C bin
fi

echo "== 5/5 models"
[ -f .env ] || cp .env.example .env
./.venv/bin/python scripts/download_models.py
echo "done.  Next: scripts/run_stack.sh"
