#!/bin/zsh
# Sampling step A/B, then ABBA serving per model (main arm upstream main, PR arm the branch),
# then the distribution check on the branch. Python only, so native artifacts are shared.
set -u
D=EVIDENCE_DIR
M=HF_CACHE/mlx-community/Qwen3-4B-4bit
H=HF_CACHE/Qwen/Qwen3.5-0.8B
G=HF_CACHE/mlx-community/gemma-4-e2b-it-4bit
R=VLLM_METAL_CHECKOUT
B=perf/native-topk-candidates
export OUT=$D/bench
sw() { git -C $R switch -q $1 && echo "checkout $(git -C $R branch --show-current) $(git -C $R rev-parse --short HEAD) $(date '+%H:%M:%S')"; }
echo "start $(date '+%H:%M:%S') $(pmset -g batt | head -1) swap: $(sysctl -n vm.swapusage)"
sw $B
( cd $R && source .venv-vllm-metal/bin/activate && python $D/sampling_ab_microbench.py $D/sampling_ab_microbench.json > $D/sampling_ab_microbench.log 2>&1; tail -3 $D/sampling_ab_microbench.log )
echo "microbench done $(date '+%H:%M:%S')"
sw main; $D/serve_multi_bench.sh r1_main $M 1024 256 16 $D/cfg_4b.txt --gpu-memory-utilization 0.5
sw $B;   $D/serve_multi_bench.sh r1_pr $M 1024 256 16 $D/cfg_4b.txt --gpu-memory-utilization 0.5
         $D/serve_multi_bench.sh r2_pr $M 1024 256 16 $D/cfg_4b.txt --gpu-memory-utilization 0.5
sw main; $D/serve_multi_bench.sh r2_main $M 1024 256 16 $D/cfg_4b.txt --gpu-memory-utilization 0.5
         $D/serve_multi_bench.sh h1_main $H 1024 256 32 $D/cfg_hybrid.txt --gpu-memory-utilization 0.4
sw $B;   $D/serve_multi_bench.sh h1_pr $H 1024 256 32 $D/cfg_hybrid.txt --gpu-memory-utilization 0.4
         $D/serve_multi_bench.sh h2_pr $H 1024 256 32 $D/cfg_hybrid.txt --gpu-memory-utilization 0.4
sw main; $D/serve_multi_bench.sh h2_main $H 1024 256 32 $D/cfg_hybrid.txt --gpu-memory-utilization 0.4
         $D/serve_multi_bench.sh g1_main $G 1024 256 16 $D/cfg_gemma.txt --gpu-memory-utilization 0.5
sw $B;   $D/serve_multi_bench.sh g1_pr $G 1024 256 16 $D/cfg_gemma.txt --gpu-memory-utilization 0.5
         $D/serve_multi_bench.sh g2_pr $G 1024 256 16 $D/cfg_gemma.txt --gpu-memory-utilization 0.5
sw main; $D/serve_multi_bench.sh g2_main $G 1024 256 16 $D/cfg_gemma.txt --gpu-memory-utilization 0.5
sw $B
echo "e2e done $(date '+%H:%M:%S') $(pmset -g batt | head -1) swap: $(sysctl -n vm.swapusage)"
( cd $R && source .venv-vllm-metal/bin/activate && python $D/distribution_parity.py $M 10000 > $D/distribution_parity.log 2>&1; tail -5 $D/distribution_parity.log )
echo "end $(date '+%H:%M:%S')"
touch $D/all.done
