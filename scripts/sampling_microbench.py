"""Per-step sampling cost on the Metal runner, logits already evaluated.

Arms: native greedy argmax, torch CPU sampler (default path for non-greedy),
MLX native categorical (VLLM_METAL_NATIVE_SAMPLING=1 path).
"""
import statistics
import sys
import time

import mlx.core as mx
from vllm.sampling_params import SamplingParams
from vllm.v1.sample.sampler import Sampler

from vllm_metal.v1.sampling_batch import (
    SamplingBatch,
    mlx_greedy_tokens,
    sample_from_logits,
)

VOCAB = 151936
REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 50
CONFIGS = {
    "qwen3_default(t0.6,k20,p0.95)": dict(temperature=0.6, top_k=20, top_p=0.95),
    "t0.7,p0.9": dict(temperature=0.7, top_p=0.9),
    "t1.0_only": dict(temperature=1.0),
}


def timeit(fn, reps):
    for _ in range(5):
        fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(ts), min(ts), max(ts)


sampler = Sampler()
key = [mx.random.key(0)]


def next_key():
    key[0], sub = mx.random.split(key[0])
    return sub


print(f"vocab={VOCAB} reps={REPS} times in ms: median (min-max)")
for b in (1, 4, 8, 16):
    logits = mx.random.normal((b, VOCAB)).astype(mx.bfloat16) * 4
    mx.eval(logits)

    def greedy():
        t = mlx_greedy_tokens(logits)
        mx.eval(t)
        t.tolist()

    print(f"B={b} greedy_native {timeit(greedy, REPS)[0]:.3f}")
    for name, kw in CONFIGS.items():
        sps = [SamplingParams(**kw) for _ in range(b)]

        def torch_path():
            batch = SamplingBatch(
                sps, [[1, 2, 3]] * b, [[4]] * b, vocab_size=VOCAB
            )
            sample_from_logits(logits, batch, sampler)

        def native_path():
            t = SamplingBatch.native_decode_tokens(
                logits, sps, vocab_size=VOCAB, next_key=next_key
            )
            mx.eval(t)
            t.tolist()

        tm = timeit(torch_path, REPS)
        nm = timeit(native_path, REPS)
        print(
            f"B={b} {name:30s} torch_cpu {tm[0]:7.3f} ({tm[1]:.3f}-{tm[2]:.3f})"
            f"  mlx_native {nm[0]:7.3f} ({nm[1]:.3f}-{nm[2]:.3f})"
        )
