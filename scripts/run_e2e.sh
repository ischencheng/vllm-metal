#!/bin/zsh
# before = S1-a head (162c24c), after = this branch; ABBA per model. Native sampling default-on in both.
set -u
D=EVIDENCE_DIR
S=EVIDENCE_DIR/scripts/serve_multi_bench.sh
M=HF_CACHE/mlx-community/Qwen3-4B-4bit
H=HF_CACHE/Qwen/Qwen3.5-0.8B
R=VLLM_METAL_CHECKOUT
A=perf/native-topk-chunked
B=perf/native-sampling-no-sort
export OUT=$D/bench PORT=8047
sw() { git -C $R switch -q $1 && echo "checkout $(git -C $R branch --show-current) $(git -C $R rev-parse --short HEAD) $(date '+%H:%M:%S')"; }
echo "start $(date '+%H:%M:%S') $(pmset -g batt | head -1) swap: $(sysctl -n vm.swapusage)"
sw $A; $S r1_before $M 1024 256 16 $D/cfg_4b.txt --kv-cache-memory-bytes 5368709120
sw $B; $S r1_after $M 1024 256 16 $D/cfg_4b.txt --kv-cache-memory-bytes 5368709120
       $S r2_after $M 1024 256 16 $D/cfg_4b.txt --kv-cache-memory-bytes 5368709120
sw $A; $S r2_before $M 1024 256 16 $D/cfg_4b.txt --kv-cache-memory-bytes 5368709120
       $S h1_before $H 1024 256 32 $D/cfg_hybrid.txt --kv-cache-memory-bytes 3221225472
sw $B; $S h1_after $H 1024 256 32 $D/cfg_hybrid.txt --kv-cache-memory-bytes 3221225472
       $S h2_after $H 1024 256 32 $D/cfg_hybrid.txt --kv-cache-memory-bytes 3221225472
sw $A; $S h2_before $H 1024 256 32 $D/cfg_hybrid.txt --kv-cache-memory-bytes 3221225472
sw $B
echo "end $(date '+%H:%M:%S') $(pmset -g batt | head -1) swap: $(sysctl -n vm.swapusage)"
touch $D/e2e.done
