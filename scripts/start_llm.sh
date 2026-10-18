#!/usr/bin/env bash
# Serve the local LLM on the GPU through an OpenAI-compatible API (vLLM, FP8 weights so it shares the 16 GB card
# with the embedding model + reranker).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODEL="${LLM_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
PORT="${LLM_PORT:-8000}"
UTIL="${LLM_GPU_UTIL:-0.70}"
CTX="${LLM_MAX_CONTEXT:-12288}"
if curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then echo "LLM server already running"; exit 0; fi
mkdir -p "$ROOT/logs"
source "$ROOT/.venv-llm/bin/activate" 2>/dev/null || export PATH="$ROOT/.venv-llm/bin:$PATH"
export VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 VLLM_USE_FLASHINFER_SAMPLER=0
nohup vllm serve "$MODEL" \
  --served-model-name "$MODEL" \
  --quantization fp8 \
  --max-model-len "$CTX" \
  --gpu-memory-utilization "$UTIL" \
  --max-num-seqs 64 --max-num-batched-tokens 4096 \
  --structured-outputs-config '{"backend": "xgrammar", "disable_any_whitespace": true}' \
  --port "$PORT" --host 127.0.0.1 \
  > "$ROOT/logs/vllm.log" 2>&1 &
echo "starting vLLM ($MODEL) - first start takes 1-3 minutes; logs/vllm.log"
for i in $(seq 1 240); do
  curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1 && { echo "LLM ready"; exit 0; }
  sleep 2
done
echo "LLM did not become ready, see logs/vllm.log"; exit 1
