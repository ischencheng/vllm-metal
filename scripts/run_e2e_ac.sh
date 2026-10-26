#!/bin/zsh
# AC-power rerun: main arm on the main checkout, PR arm on the branch with default env.
set -u
D=./scripts
M=mlx-community/Qwen3-4B-4bit
H=Qwen/Qwen3.5-0.8B
R=VLLM_METAL_CHECKOUT
export OUT=$D/bench_ac
sw() { git -C $R switch -q $1 && echo "checkout $(git -C $R branch --show-current) $(git -C $R rev-parse --short HEAD)"; }
echo "start $(date '+%H:%M:%S') $(pmset -g batt | head -1) swap: $(sysctl -n vm.swapusage)"
sw main;                          $D/serve_multi_bench.sh r1_main $M 1024 256 8 $D/cfg_main.txt --gpu-memory-utilization 0.5
sw perf/native-sampling-default;  $D/serve_multi_bench.sh r1_pr $M 1024 256 8 $D/cfg_main.txt --gpu-memory-utilization 0.5
                                  $D/serve_multi_bench.sh r2_pr $M 1024 256 8 $D/cfg_main.txt --gpu-memory-utilization 0.5
sw main;                          $D/serve_multi_bench.sh r2_main $M 1024 256 8 $D/cfg_main.txt --gpu-memory-utilization 0.5
                                  $D/serve_multi_bench.sh h1_main $H 1024 256 4 $D/cfg_hybrid.txt --gpu-memory-utilization 0.3
sw perf/native-sampling-default;  $D/serve_multi_bench.sh h1_pr $H 1024 256 4 $D/cfg_hybrid.txt --gpu-memory-utilization 0.3
                                  $D/serve_multi_bench.sh h2_pr $H 1024 256 4 $D/cfg_hybrid.txt --gpu-memory-utilization 0.3
sw main;                          $D/serve_multi_bench.sh h2_main $H 1024 256 4 $D/cfg_hybrid.txt --gpu-memory-utilization 0.3
sw perf/native-sampling-default
echo "end $(date '+%H:%M:%S') $(pmset -g batt | head -1) swap: $(sysctl -n vm.swapusage)"
