#!/bin/bash
# Start a vllm-metal server (env passed through), run a command against it, stop it.
# Usage: serve_run.sh <tag> <model> <cmd...>
set -euo pipefail
TAG=$1; MODEL=$2; shift 2
REPO=VLLM_METAL_CHECKOUT
OUT=./quality
PORT=${PORT:-8011}
mkdir -p "$OUT"
source "$REPO/.venv-vllm-metal/bin/activate"
cd "$REPO"
VLLM_ENABLE_V1_MULTIPROCESSING=1 vllm serve "$MODEL" --host 127.0.0.1 --port "$PORT" \
  --served-model-name bench --max-model-len 4096 --max-num-seqs 8 \
  --gpu-memory-utilization 0.5 --generation-config vllm \
  --enable-auto-tool-choice --tool-call-parser hermes > "$OUT/$TAG.server.log" 2>&1 &
SPID=$!
trap 'kill $SPID 2>/dev/null; sleep 2; pkill -P $SPID 2>/dev/null || true' EXIT
for _ in $(seq 1 300); do
  if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null; then break; fi
  if ! kill -0 $SPID 2>/dev/null; then echo "server died"; tail -40 "$OUT/$TAG.server.log"; exit 1; fi
  sleep 1
done
"$@"
