"""Inspect the t1.0_k64_p0.95 / prompt 5 case: where do draws fall outside the float64 target?"""
import sys
import mlx.core as mx
import numpy as np
from mlx_lm import load
from vllm.sampling_params import SamplingParams
sys.path.insert(0, sys.argv[2]); sys.argv = sys.argv[:2]
from distribution_parity import exact_target, PROMPTS
from vllm_metal.v1.sampling_batch import SamplingBatch

model, tok = load(sys.argv[1])
vocab = model.args.vocab_size
ids = mx.array(tok.encode(PROMPTS[5]))[None]
row = model(ids)[0, -1, :vocab].astype(mx.bfloat16)
mx.eval(row)
sp = SamplingParams(temperature=1.0, top_k=64, top_p=0.95)
x = np.array(row.astype(mx.float32))
target = exact_target(x, sp)
sup = np.flatnonzero(target > 0)
print("support", sup.size, "target sum", target.sum())
xs = x / 1.0
kth = np.sort(xs)[-64]
print("kth", kth, "ties at kth over vocab", int((xs == kth).sum()), "support min value", xs[sup].min(), "ties at that value", int((xs == xs[sup].min()).sum()), "kept at that value", int(((xs == xs[sup].min()) & (target > 0)).sum()))
B = 500
logits = mx.contiguous(mx.tile(row[None], (B, 1)))
key = mx.random.key(1234)
draws = []
for _ in range(20):
    key, sub = mx.random.split(key)
    draws.append(np.array(SamplingBatch._native_random_tokens(logits, [sp] * B, sub)))
draws = np.concatenate(draws)
out = draws[target[draws] == 0]
print("native draws outside target:", out.size, "unique", np.unique(out)[:20], "their logits", np.unique(xs[out])[:10])
# chunked candidates: implied kept count vs target
cl, tid, tv, tc = SamplingBatch._top_k_candidates(mx.array(xs)[None], [64], [0.95], [0.0])
mx.eval(cl, tid, tv, tc)
kept_above = int((np.array(cl)[0, :-1] > -np.inf).sum())
print("chunked: kept above kth", kept_above, "tie value", float(np.array(tv)[0, 0]), "tie_count kept", float(np.array(tc)[0, 0]), "total", kept_above + float(np.array(tc)[0, 0]))
sd, si = SamplingBatch._sorted_candidate_logits(mx.array(xs)[None], [64], [0.95], [0.0])
mx.eval(sd)
print("full sort: kept", int((np.array(sd)[0] > -np.inf).sum()))
