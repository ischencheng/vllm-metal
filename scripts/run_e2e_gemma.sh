#!/bin/zsh
set -u
D=EVIDENCE_DIR
G=HF_CACHE/mlx-community/gemma-4-e2b-it-4bit
R=VLLM_METAL_CHECKOUT
B=perf/native-topk-chunked
export OUT=$D/bench
sw() { git -C $R switch -q $1 && echo "checkout $(git -C $R branch --show-current) $(git -C $R rev-parse --short HEAD) $(date '+%H:%M:%S')"; }
echo "start $(date '+%H:%M:%S') $(pmset -g batt | head -1) swap: $(sysctl -n vm.swapusage)"
sw main; $D/serve_multi_bench.sh g1_main $G 1024 256 16 $D/cfg_gemma.txt --gpu-memory-utilization 0.5
sw $B;  $D/serve_multi_bench.sh g1_pr $G 1024 256 16 $D/cfg_gemma.txt --gpu-memory-utilization 0.5
         $D/serve_multi_bench.sh g2_pr $G 1024 256 16 $D/cfg_gemma.txt --gpu-memory-utilization 0.5
sw main; $D/serve_multi_bench.sh g2_main $G 1024 256 16 $D/cfg_gemma.txt --gpu-memory-utilization 0.5
sw $B
echo "end $(date '+%H:%M:%S') $(pmset -g batt | head -1) swap: $(sysctl -n vm.swapusage)"
touch $D/e2e_gemma.done
