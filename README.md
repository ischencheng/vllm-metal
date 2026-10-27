# Native top-k without a vocab sort: benchmark records

Records for the vllm-metal PR that picks native top-k candidates from the top chunks instead of argsorting the whole vocab (branch `perf/native-topk-chunked`, commit `162c24c`, base `28cc4c7`).

Machine: Apple M5 Pro, 20-core GPU, 48 GB, macOS 26.4.1, AC power. MLX 0.32.4, mlx-lm 0.32.0, vLLM 0.32.0. The PR changes no native code; both arms use the same native build.

- `sampling_ab_microbench.log` / `.json`: one sampling step with evaluated bf16 logits. Both paths run in one process on the same logits (the full-sort arm forces `_uses_top_k_candidates` to return False), 3 interleaved rounds of 50 per arm, after an untimed warm-up of both paths. `_r1` is the first run, without that warm-up. Script: `scripts/sampling_ab_microbench.py`.
- `serving/`: `vllm bench serve` results, client and server logs. `r*` is `mlx-community/Qwen3-4B-4bit` (max-num-seqs 16, gpu-memory-utilization 0.5), `h*` is `Qwen/Qwen3.5-0.8B` (32, 0.4), `g*` is `mlx-community/gemma-4-e2b-it-4bit` (16, 0.5). Order per model: round 1 main, round 1 PR, round 2 PR, round 2 main. `main` runs on the `28cc4c7` checkout, `pr` on the branch, no env vars on either. Drivers `scripts/run_e2e.sh` and `scripts/run_e2e_gemma.sh`, client configs `scripts/cfg_*.txt`. Random dataset, 1024 input / 256 output tokens, `--ignore-eos`, 2 warmups, seed 713. `e2e_summary.md` has the medians, P99s, TTFTs and counts (`scripts/summarize.py`).
- `distribution_parity.log`: native and torch samplers against the exact masked distribution on real Qwen3-4B logits, 10,000 draws per case (`scripts/distribution_parity.py`). Statistics are per class of equal logits. `scripts/debug_p5.py` is the check for the one case where the native path kept a different member of a two-token tie at the top-p cut than numpy's order (same kept count).
- `fast_suite.log`: ruff, ruff format, mypy and `pytest -m "not slow" tests/ .github/scripts/test_parity.py` on `162c24c`.

Paths are shortened: `VLLM_METAL_CHECKOUT` is the vllm-metal checkout, `HF_CACHE/<org>/<name>` a Hugging Face cache snapshot, `EVIDENCE_DIR` the directory these files came from.
