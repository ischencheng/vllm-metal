#!/bin/zsh
# main arm on upstream main, PR arm on the branch; ABBA per model. Native sampling is default-on on both.
set -u
D=EVIDENCE_DIR
M=HF_CACHE/mlx-community/Qwen3-4B-4bit
H=HF_CACHE/Qwen/Qwen3.5-0.8B
R=VLLM_METAL_CHECKOUT
B=perf/native-topk-chunked
export OUT=$D/bench
sw() { git -C $R switch -q $1 && echo "checkout $(git -C $R branch --show-current) $(git -C $R rev-parse --short HEAD) $(date '+%H:%M:%S')"; }
echo "start $(date '+%H:%M:%S') $(pmset -g batt | head -1) swap: $(sysctl -n vm.swapusage)"
sw main; $D/serve_multi_bench.sh r1_main $M 1024 256 16 $D/cfg_4b.txt --gpu-memory-utilization 0.5
sw $B;   $D/serve_multi_bench.sh r1_pr $M 1024 256 16 $D/cfg_4b.txt --gpu-memory-utilization 0.5
         $D/serve_multi_bench.sh r2_pr $M 1024 256 16 $D/cfg_4b.txt --gpu-memory-utilization 0.5
sw main; $D/serve_multi_bench.sh r2_main $M 1024 256 16 $D/cfg_4b.txt --gpu-memory-utilization 0.5
         $D/serve_multi_bench.sh h1_main $H 1024 256 32 $D/cfg_hybrid.txt --gpu-memory-utilization 0.4
sw $B;   $D/serve_multi_bench.sh h1_pr $H 1024 256 32 $D/cfg_hybrid.txt --gpu-memory-utilization 0.4
         $D/serve_multi_bench.sh h2_pr $H 1024 256 32 $D/cfg_hybrid.txt --gpu-memory-utilization 0.4
sw main; $D/serve_multi_bench.sh h2_main $H 1024 256 32 $D/cfg_hybrid.txt --gpu-memory-utilization 0.4
sw $B
echo "end $(date '+%H:%M:%S') $(pmset -g batt | head -1) swap: $(sysctl -n vm.swapusage)"
touch $D/e2e.done
