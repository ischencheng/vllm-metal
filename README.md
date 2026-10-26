# Native sampling by default: benchmark records

Records for the vllm-metal PR that turns `VLLM_METAL_NATIVE_SAMPLING` on by default (branch `perf/native-sampling-default`, commit `3a09336`, base `cac610b`).

Machine: Apple M5 Pro, 20-core GPU, 48 GB, macOS 26.4.1, AC power, power mode automatic. A VM process used about 85% of one CPU core during all runs, both arms. MLX 0.32.4, mlx-lm 0.32.0, vLLM 0.32.0, torch 2.13.0. Native extension built from source on `cac610b` (the PR changes no native code).

- `microbench_ac.log`: one sampling step with evaluated logits, 151,936 vocab, median of 40 (`scripts/sampling_microbench.py`).
- `serving/`: `vllm bench serve` results and server logs. `r*` is `mlx-community/Qwen3-4B-4bit` (max-num-seqs 8, gpu-memory-utilization 0.5), `h*` is `Qwen/Qwen3.5-0.8B` (max-num-seqs 4, 0.3). Order: r1_main, r1_pr, r2_pr, r2_main, h1_main, h1_pr, h2_pr, h2_main. `main` runs on the `cac610b` checkout, `pr` on the branch with no env vars. Client configs are in `scripts/cfg_main.txt` and `scripts/cfg_hybrid.txt`, driver `scripts/run_e2e_ac.sh`. Random dataset, 1024 input / 256 output tokens, `--ignore-eos`, 2 warmups, seed 713. `summary.md` has the medians, P99s, TTFTs and counts.
- `route/`: the branch with `VLLM_METAL_DECODE_PIPELINE=0` (`pr_nopipe`), with that plus `VLLM_METAL_NATIVE_SAMPLING=0` (`torch_nopipe`), and the default (`pr_default`).
- `distribution_parity.log`: native and torch samplers against the exact masked distribution on real Qwen3-4B logits (`scripts/distribution_parity.py`).
- `quality/`: sampled GSM8K (first 400 test questions, `scripts/gsm8k_sampled.py`), default vs `VLLM_METAL_NATIVE_SAMPLING=0`, plus `tools/native_sampling_probe.py` results.
- `fast_suite_final.log`: lint, mypy and `pytest -m "not slow"` on the branch.

Paths are shortened: `VLLM_METAL_CHECKOUT` is the vllm-metal checkout, `GSM8K_DIR` holds `test.jsonl` from openai/grade-school-math.
