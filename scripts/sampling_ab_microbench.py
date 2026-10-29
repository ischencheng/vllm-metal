"""Sampling-step A/B, same process: S1-a head (162c24c, full-vocab Gumbel draw,
sort for every batch without top-k on all rows) vs this branch.

Both arms are the real _native_random_tokens of each revision; the S1-a module is
loaded from a copy of its file. Times one evaluated step (includes one MLX eval
round trip, which greedy argmax also pays). 3 interleaved rounds x 40 iters.
"""
import importlib.util, json, platform, sys, time
from pathlib import Path
import mlx.core as mx
from vllm.sampling_params import SamplingParams
from vllm_metal.v1.sampling_batch import SamplingBatch as After

spec = importlib.util.spec_from_file_location("s1a", Path(__file__).with_name("sampling_batch_162c24c.py"))
s1a = importlib.util.module_from_spec(spec); spec.loader.exec_module(s1a)
Before = s1a.SamplingBatch

ITERS, WARMUP, ROUNDS = 40, 8, 3
CONFIGS = {
    "temp_only_qwen": (151936, dict(temperature=1.0, top_k=0, top_p=1.0)),
    "temp_only_llama": (128256, dict(temperature=0.7, top_k=0, top_p=1.0)),
    "min_p_qwen": (151936, dict(temperature=0.7, top_k=0, top_p=1.0, min_p=0.05)),
    "top_p_llama": (128256, dict(temperature=0.6, top_k=0, top_p=0.9)),
    "top_p_deepseek": (129280, dict(temperature=0.6, top_k=0, top_p=0.95)),
    "qwen_default_control": (151936, dict(temperature=0.6, top_k=20, top_p=0.95)),
}
BATCHES = (1, 4, 8, 16, 32, 64)

def time_fn(fn):
    ts, key = [], mx.random.key(1)
    for i in range(WARMUP + ITERS):
        key, sub = mx.random.split(key)
        mx.synchronize(); t = time.perf_counter(); mx.eval(fn(sub)); mx.synchronize()
        if i >= WARMUP: ts.append((time.perf_counter() - t) * 1e3)
    return ts

def q(ts, p): return sorted(ts)[int(p * (len(ts) - 1))]
out = {"mlx": mx.__version__, "device": mx.device_info().get("architecture"), "macos": platform.mac_ver()[0], "rows": []}
print(json.dumps({k: out[k] for k in ("mlx", "device", "macos")}), flush=True)
for vocab, kw in CONFIGS.values():  # untimed warm-up of every path
    w = (mx.random.normal((4, vocab)) * 3).astype(mx.bfloat16)
    for cls in (Before, After):
        time_fn(lambda k: cls._native_random_tokens(w, [SamplingParams(**kw)] * 4, k))
for name, (vocab, kw) in CONFIGS.items():
    for B in BATCHES:
        logits = (mx.random.normal((B, vocab), key=mx.random.key(B)) * 3).astype(mx.bfloat16)
        mx.eval(logits)
        params = [SamplingParams(**kw)] * B
        arms = {"before": [], "after": [], "greedy": []}
        for r in range(ROUNDS):
            for arm in (("before", "after") if r % 2 == 0 else ("after", "before")):
                cls = Before if arm == "before" else After
                arms[arm] += time_fn(lambda k: cls._native_random_tokens(logits, params, k))
            arms["greedy"] += time_fn(lambda k: mx.argmax(logits, axis=-1))
        row = {"config": name, "vocab": vocab, "batch": B}
        for arm, ts in arms.items():
            row[arm] = {"p50": round(q(ts, 0.5), 3), "p10": round(q(ts, 0.1), 3), "p90": round(q(ts, 0.9), 3)}
        out["rows"].append(row)
        print(f"{name:22s} B={B:<3d} greedy={row['greedy']['p50']:.3f} before={row['before']['p50']:.3f} "
              f"[{row['before']['p10']:.3f}-{row['before']['p90']:.3f}] after={row['after']['p50']:.3f} "
              f"[{row['after']['p10']:.3f}-{row['after']['p90']:.3f}]", flush=True)
json.dump(out, open(Path(__file__).with_name("sampling_ab_microbench.json"), "w"), indent=1)
