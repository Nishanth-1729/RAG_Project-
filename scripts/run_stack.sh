#!/usr/bin/env bash
# Start the whole stack: Qdrant -> local LLM (GPU) -> API (:8100) -> Streamlit UI (:8501)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
mkdir -p logs
./scripts/start_qdrant.sh
if [ "${NO_LLM:-0}" != "1" ]; then ./scripts/start_llm.sh; fi
if ! curl -sf http://127.0.0.1:8100/health >/dev/null 2>&1; then
  nohup ./.venv/bin/python Main.py serve --host 0.0.0.0 --port 8100 > logs/api.log 2>&1 &
  for i in $(seq 1 90); do curl -sf http://127.0.0.1:8100/health >/dev/null 2>&1 && break; sleep 2; done
fi
echo "API   : http://localhost:8100/docs"
if ! curl -sf http://127.0.0.1:8501/_stcore/health >/dev/null 2>&1; then
  API_URL=http://127.0.0.1:8100 nohup ./.venv/bin/streamlit run frontend/app.py --server.port 8501 --server.headless true \
     --browser.gatherUsageStats false > logs/ui.log 2>&1 &
fi
echo "UI    : http://localhost:8501"
