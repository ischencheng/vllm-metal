# Native sampling without a vocab sort for temperature / min-p rows: benchmark records

Records for the vllm-metal PR that skips the vocab sort when no row sets top-k or top-p, and draws full-width rows with a two-level categorical (branch `perf/native-sampling-no-sort`, commit `05d9092`, on top of the chunked top-k branch `162c24c`, base `28cc4c7`).

Machine: Apple M5 Pro, 20-core GPU, 48 GB, macOS 26.4.1, AC power. MLX 0.32.4, vLLM 0.32.0. The PR changes no native code; both arms use the same native build.

- `sampling_ab_microbench.log` / `.json`: one sampling step with evaluated bf16 logits. Both arms are the real `_native_random_tokens`: "before" is loaded from a copy of `sampling_batch.py` at `162c24c` (`scripts/sampling_batch_162c24c.py`), "after" is the branch. Same process, same logits, 3 interleaved rounds of 40 per arm after an untimed warm-up of every path. Script: `scripts/sampling_ab_microbench.py`.
- `probe_rng.log`, `probe_twolevel.log`, `probe_ops.log`: MLX op timings behind the change (cost of `uniform`, `gumbel` and `categorical` on a full row; two-level draw chunk sizes; scatter-add histograms and masked sums). Scripts in `scripts/`.
- `serving/`: `vllm bench serve` results, client and server logs. `r*` is `mlx-community/Qwen3-4B-4bit` (max-num-seqs 16, KV 5 GiB), `h*` is `Qwen/Qwen3.5-0.8B` (32, KV 3 GiB). Order per model: round 1 before, round 1 after, round 2 after, round 2 before. Driver `scripts/run_e2e.sh` (server wrapper `scripts/serve_multi_bench.sh`), client configs `scripts/cfg_*.txt`. Random dataset, 1024 input / 256 output tokens, `--ignore-eos`, 2 warmups, seed 713, `--generation-config vllm`. `e2e_summary.md` has the medians, P99s, TTFTs and counts (`scripts/summarize.py`).
- `distribution_parity.log`: native and torch samplers against the exact masked distribution on real Qwen3-4B logits, 20,000 draws per case (`scripts/distribution_parity.py`). TV, chi-square and masked hits are per class of equal logits; the last lines also give per-token TV. `recheck_control.log` reruns the two cases with chi-square p < 0.01 with fresh seeds and 100,000 draws, and checks the chunked top-k candidate probabilities exactly (`scripts/recheck_control.py`).
- `test_margin_mutant.log`: TV of the new distribution tests on the branch and with the chunk weight replaced by the chunk max.
- `fast_suite.log`: ruff, ruff format, mypy and `pytest -m "not slow" tests/ .github/scripts/test_parity.py` on `05d9092`.

Paths are shortened: `VLLM_METAL_CHECKOUT` is the vllm-metal checkout, `HF_CACHE/<org>/<name>` a Hugging Face cache snapshot, `EVIDENCE_DIR` the directory these files came from.
