"""Sampling-step A/B: full-vocab sort vs chunked top-k candidates, same process.

Both arms call SamplingBatch._native_random_tokens on the same logits; the sort
arm forces _uses_top_k_candidates to False. Times one evaluated step (includes
one MLX eval round trip, which greedy argmax also pays). 3 interleaved rounds.
"""
import json, platform, statistics, sys, time
import mlx.core as mx
from vllm.sampling_params import SamplingParams
from vllm_metal.v1.sampling_batch import SamplingBatch

ITERS, WARMUP, ROUNDS = 50, 8, 3
CONFIGS = {  # name: (vocab, SamplingParams kwargs)
    "qwen3_default": (151936, dict(temperature=0.6, top_k=20, top_p=0.95)),
    "gemma_default": (262144, dict(temperature=1.0, top_k=64, top_p=0.95)),
    "qwen3_minp": (151936, dict(temperature=0.7, top_k=20, top_p=0.8, min_p=0.05)),
}
BATCHES = (1, 2, 4, 8, 16, 32, 64)
orig = SamplingBatch._uses_top_k_candidates

def time_arm(logits, params, chunked):
    SamplingBatch._uses_top_k_candidates = staticmethod(orig if chunked else (lambda *a: False))
    key = mx.random.key(1)
    ts = []
    for i in range(WARMUP + ITERS):
        key, sub = mx.random.split(key)
        mx.synchronize(); t = time.perf_counter()
        mx.eval(SamplingBatch._native_random_tokens(logits, params, sub))
        mx.synchronize()
        if i >= WARMUP: ts.append((time.perf_counter() - t) * 1e3)
    SamplingBatch._uses_top_k_candidates = orig
    return ts

def time_greedy(logits):
    ts = []
    for i in range(WARMUP + ITERS):
        mx.synchronize(); t = time.perf_counter(); mx.eval(mx.argmax(logits, axis=-1)); mx.synchronize()
        if i >= WARMUP: ts.append((time.perf_counter() - t) * 1e3)
    return ts

def q(ts, p): return sorted(ts)[int(p * (len(ts) - 1))]
out = {"mlx": mx.__version__, "device": mx.device_info().get("architecture"), "macos": platform.mac_ver()[0], "rows": []}
print(json.dumps({k: out[k] for k in ("mlx", "device", "macos")}))
# Untimed warm-up of both paths so the first timed config is not a compile/allocation run.
for vocab, kw in CONFIGS.values():
    w = (mx.random.normal((4, vocab)) * 3).astype(mx.bfloat16)
    for chunked in (False, True):
        time_arm(w, [SamplingParams(**kw)] * 4, chunked)
for name, (vocab, kw) in CONFIGS.items():
    for B in BATCHES:
        logits = (mx.random.normal((B, vocab), key=mx.random.key(B)) * 3).astype(mx.bfloat16)
        mx.eval(logits)
        params = [SamplingParams(**kw)] * B
        arms = {"sort": [], "chunked": [], "greedy": []}
        for r in range(ROUNDS):
            order = ("sort", "chunked") if r % 2 == 0 else ("chunked", "sort")
            for arm in order:
                arms[arm] += time_arm(logits, params, arm == "chunked")
            arms["greedy"] += time_greedy(logits)
        row = {"config": name, "vocab": vocab, "batch": B}
        for arm, ts in arms.items():
            row[arm] = {"median": statistics.median(ts), "p10": q(ts, .1), "p90": q(ts, .9), "n": len(ts)}
        out["rows"].append(row)
        print(f"{name:14s} B={B:3d} sort {row['sort']['median']:.3f} [{row['sort']['p10']:.3f},{row['sort']['p90']:.3f}]  "
              f"chunked {row['chunked']['median']:.3f} [{row['chunked']['p10']:.3f},{row['chunked']['p90']:.3f}]  "
              f"greedy {row['greedy']['median']:.3f}", flush=True)
json.dump(out, open(sys.argv[1], "w"), indent=1)
