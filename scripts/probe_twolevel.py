"""Prototype: skip the vocab sort for temperature-only rows, and draw with a
two-level categorical (chunk by logsumexp, then token) instead of full-vocab
Gumbel noise. Same-process interleaved timing, median of 30 evals."""
import time, mlx.core as mx

def bench(fn, iters=30, warm=5):
    for _ in range(warm): mx.eval(fn())
    ts=[]
    for _ in range(iters):
        t=time.perf_counter(); mx.eval(fn()); ts.append((time.perf_counter()-t)*1e3)
    ts.sort(); return ts[len(ts)//2]

def two_level(logits, key, chunk):
    B, V = logits.shape
    nc = -(-V // chunk); pad = nc * chunk - V
    if pad: logits = mx.pad(logits, [(0, 0), (0, pad)], constant_values=-mx.inf)
    ch = logits.reshape(B, nc, chunk)
    k1, k2 = mx.random.split(key)
    c = mx.random.categorical(mx.logsumexp(ch, axis=-1), axis=-1, key=k1)
    sel = mx.take_along_axis(ch, c[:, None, None], axis=1)[:, 0, :]
    return c * chunk + mx.random.categorical(sel, axis=-1, key=k2)

def main_sorted(x, key, top_p=None):
    idx = mx.argsort(x, axis=-1)[..., ::-1]; s = mx.take_along_axis(x, idx, axis=-1)
    if top_p is not None:
        p = mx.softmax(s, axis=-1); lead = mx.cumsum(p, axis=-1) - p
        s = mx.where(lead < top_p, s, -mx.inf)
    pos = mx.random.categorical(s, axis=-1, key=key)
    return mx.take_along_axis(idx, pos[:, None], axis=-1)

def sorted_two_level(x, key, top_p, chunk):
    idx = mx.argsort(x, axis=-1)[..., ::-1]; s = mx.take_along_axis(x, idx, axis=-1)
    p = mx.softmax(s, axis=-1); lead = mx.cumsum(p, axis=-1) - p
    s = mx.where(lead < top_p, s, -mx.inf)
    pos = two_level(s, key, chunk)
    return mx.take_along_axis(idx, pos[:, None].astype(mx.int32), axis=-1)

key = mx.random.key(0)
for V in (151936, 128256):
    for B in (1, 8, 16, 32, 64):
        x = (mx.random.normal((B, V)) * 2.5).astype(mx.float32); mx.eval(x)
        r = {}
        r["greedy"] = bench(lambda: mx.argmax(x, axis=-1))
        r["T:main"] = bench(lambda: main_sorted(x, key))
        r["T:cat"] = bench(lambda: mx.random.categorical(x, axis=-1, key=key))
        for c in (64, 256, 512):
            r[f"T:2lvl{c}"] = bench(lambda: two_level(x, key, c))
        r["P:main"] = bench(lambda: main_sorted(x, key, 0.95))
        r["P:sort+2lvl256"] = bench(lambda: sorted_two_level(x, key, 0.95, 256))
        print(f"V={V} B={B} " + " ".join(f"{k}={v:.3f}" for k, v in r.items()), flush=True)
