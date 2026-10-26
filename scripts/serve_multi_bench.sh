#!/bin/bash
# Start one vllm-metal server, run several `vllm bench serve` configs against it, stop it.
# Usage: serve_multi_bench.sh <tag> <model_path> <in> <out> <max_conc> <cfgfile> [extra serve args...]
# cfgfile lines: <label> <concurrency> <num_prompts> <sampling flags...>
# Env passed through to the server (e.g. VLLM_METAL_NATIVE_SAMPLING=1).
set -euo pipefail
TAG=$1; MODEL=$2; IN=$3; OUTLEN=$4; MAXC=$5; CFG=$6; shift 6
REPO=VLLM_METAL_CHECKOUT
OUT=${OUT:-./scripts/bench}
PORT=${PORT:-8011}
mkdir -p "$OUT"
source "$REPO/.venv-vllm-metal/bin/activate"
cd "$REPO"
MAXLEN=$(( IN + OUTLEN + 256 ))
VLLM_ENABLE_V1_MULTIPROCESSING=1 vllm serve "$MODEL" --host 127.0.0.1 --port "$PORT" \
  --served-model-name bench --max-model-len "$MAXLEN" --max-num-seqs "$MAXC" \
  --max-num-batched-tokens "${MNBT:-2048}" --no-enable-prefix-caching --generation-config vllm "$@" \
  > "$OUT/$TAG.server.log" 2>&1 &
SPID=$!
trap 'kill $SPID 2>/dev/null; sleep 2; pkill -P $SPID 2>/dev/null || true' EXIT
for _ in $(seq 1 300); do
  if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null; then break; fi
  if ! kill -0 $SPID 2>/dev/null; then echo "server died"; tail -40 "$OUT/$TAG.server.log"; exit 1; fi
  sleep 1
done
while read -r LABEL CONC NP SAMPLING; do
  [ -z "$LABEL" ] && continue
  case "$LABEL" in \#*) continue;; esac
  # shellcheck disable=SC2086
  vllm bench serve --model "$MODEL" --served-model-name bench \
    --backend openai --base-url "http://127.0.0.1:$PORT" --endpoint /v1/completions \
    --dataset-name random --random-input-len "$IN" --random-output-len "$OUTLEN" \
    --random-range-ratio 0 --num-prompts "$NP" --num-warmups "${WARMUPS:-2}" \
    --request-rate inf --max-concurrency "$CONC" --ignore-eos $SAMPLING --seed 713 \
    --percentile-metrics ttft,tpot,e2el --metric-percentiles 50,95,99 \
    --save-result --result-dir "$OUT" --result-filename "${TAG}_${LABEL}.json" > "$OUT/${TAG}_${LABEL}.client.log" 2>&1
  printf '%s %s ' "$TAG" "$LABEL"; grep -E "Median TPOT|Output token throughput" "$OUT/${TAG}_${LABEL}.client.log" | tr -s ' ' | tr '\n' ' '; echo
done < "$CFG"
