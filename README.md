# Native top-k without a vocab sort: benchmark records

Records for the vllm-metal PR that picks native top-k candidates from the top chunks instead of argsorting the whole vocab (branch `perf/native-topk-candidates`, commit `955b1e2`, base `fc9c6ee`).

Machine: Apple M5 Pro, 20-core GPU, 48 GB, macOS 26.4.1, AC power. MLX 0.32.4, vLLM 0.33.0, torch 2.14.0. The PR changes no native code; both arms use the same native build.

- `sampling_ab_microbench.log` / `.json`: one sampling step with evaluated bf16 logits. Both paths run in one process on the same logits (the full-sort arm forces `_uses_top_k_candidates` to return False), 3 interleaved rounds of 50 per arm, after an untimed warm-up of both paths. Script: `scripts/sampling_ab_microbench.py`.
- `serving/`: `vllm bench serve` results, client and server logs. `r*` is `mlx-community/Qwen3-4B-4bit` (max-num-seqs 16, gpu-memory-utilization 0.5), `h*` is `Qwen/Qwen3.5-0.8B` (32, 0.4), `g*` is `mlx-community/gemma-4-e2b-it-4bit` (16, 0.5). Order per model: round 1 main, round 1 PR, round 2 PR, round 2 main. `main` runs on the `fc9c6ee` checkout, `pr` on the branch, no env vars on either. Driver `scripts/run_all.sh`, client configs `scripts/cfg_*.txt`. Random dataset, 1024 input / 256 output tokens, `--ignore-eos`, 2 warmups, seed 713. `e2e_summary.md` has the medians, P99s, TTFTs and counts (`scripts/summarize.py`).
- `distribution_parity.log`: native and torch samplers against the exact masked distribution on real Qwen3-4B logits, 10,000 draws per case (`scripts/distribution_parity.py`). Statistics are per class of equal logits.
- `fast_suite.log`: `pytest -m "not slow" tests/ .github/scripts/test_parity.py` on `955b1e2`.
- `threshold_bench.log`, `threshold_bench_small_vocab.log`: the two paths near the top-k bound and for small vocabs and batches, which set `TOP_K_MIN_BATCH_LOGITS`. These ran with the guard off, so both arms are real.
- `first_run_f600c1d/`: the same runs on the previous commit of the branch, before the small-batch guard and the tie mask reuse.
- `stack_on_1234.patch`, `stack_on_1234_microbench.log` / `.json`: the same change ported onto PR #1234 at `95da4bb`, and one sampling step there, same method. `sort` is #1234 as is, `chunked` adds this change. The temp_only, min_p and top_p_only rows don't take the chunked path, so their two arms should match.

Paths are shortened: `VLLM_METAL_CHECKOUT` is the vllm-metal checkout, `HF_CACHE/<org>/<name>` a Hugging Face cache snapshot, `EVIDENCE_DIR` the directory these files came from.
