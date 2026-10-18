#!/usr/bin/env bash
# Stop API, UI, LLM server and Qdrant (frees the GPU).
for pat in "streamlit run" "Main.py serve" "vllm serve" "bin/qdrant"; do
  for pid in $(ps -eo pid,args | grep -F "$pat" | grep -v grep | grep -v "stop_stack" | awk '{print $1}'); do kill "$pid" 2>/dev/null || true; done
done
echo "stopped"
