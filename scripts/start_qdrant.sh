#!/usr/bin/env bash
# Start the local Qdrant vector DB (native binary, no Docker needed).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$ROOT/storage/qdrant" "$ROOT/logs"
if curl -sf http://127.0.0.1:6333/healthz >/dev/null 2>&1; then echo "qdrant already running"; exit 0; fi
export QDRANT__STORAGE__STORAGE_PATH="$ROOT/storage/qdrant/storage"
export QDRANT__STORAGE__SNAPSHOTS_PATH="$ROOT/storage/qdrant/snapshots"
export QDRANT__SERVICE__HTTP_PORT=6333
export QDRANT__SERVICE__GRPC_PORT=6334
export QDRANT__TELEMETRY_DISABLED=true
cd "$ROOT/storage/qdrant"
nohup "$ROOT/bin/qdrant" > "$ROOT/logs/qdrant.log" 2>&1 &
for i in $(seq 1 30); do curl -sf http://127.0.0.1:6333/healthz >/dev/null 2>&1 && { echo "qdrant up"; exit 0; }; sleep 1; done
echo "qdrant failed to start, see logs/qdrant.log"; exit 1
