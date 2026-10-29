"""Recheck the two chi2 p<0.01 cases from distribution_parity.log with fresh keys
and more draws, and check the chunked top-k candidate probabilities exactly."""
import sys
import mlx.core as mx
import numpy as np
from mlx_lm import load
from scipy import stats
from vllm.sampling_params import SamplingParams
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from distribution_parity import PROMPTS, exact_target
from vllm_metal.v1.sampling_batch import SamplingBatch

model, tok = load(sys.argv[1])
vocab = model.args.vocab_size
cases = [("t0.6_k20_p0.95", 0, dict(temperature=0.6, top_k=20, top_p=0.95)),
         ("t0.7", 4, dict(temperature=0.7))]
for name, pi, kw in cases:
    sp = SamplingParams(**kw)
    ids = mx.array(tok.encode(PROMPTS[pi]))[None]
    row = model(ids)[0, -1, :vocab].astype(mx.bfloat16); mx.eval(row)
    x = np.array(row.astype(mx.float32))
    target = exact_target(x, sp)
    _, cls = np.unique(x, return_inverse=True)
    tgt = np.bincount(cls, weights=target)
    if kw.get("top_k"):
        scaled = (row.astype(mx.float32) / sp.temperature)[None]
        lg, tid, tv, tc = SamplingBatch._top_k_candidates(scaled, [sp.top_k], [sp.top_p], [sp.min_p])
        p = np.array(mx.softmax(lg, axis=-1))[0]; ids_ = np.array(tid)[0]
        cand = np.zeros(vocab); cand[ids_] = p[: len(ids_)]
        print(name, "candidate probs vs exact: max abs diff", float(np.abs(cand - target).max()),
              "top target", sorted(((round(float(target[i]), 5), int(i)) for i in np.nonzero(target)[0]), reverse=True)[:4])
    for seed in (1, 2, 3):
        key = mx.random.key(1000 + seed)
        batch = mx.contiguous(mx.tile(row[None], (1000, 1))); mx.eval(batch)
        draws = []
        for _ in range(100):
            key, sub = mx.random.split(key)
            t = SamplingBatch.native_decode_tokens(batch, [sp] * 1000, vocab_size=vocab, next_key=lambda s=sub: s)
            draws.append(np.array(t))
        d = np.concatenate(draws)
        c = np.bincount(cls[d], minlength=tgt.size).astype(float)
        n = c.sum(); sup = tgt > 0; e = tgt[sup] * n; o = c[sup]; big = e >= 5
        eb = np.append(e[big], e[~big].sum()); ob = np.append(o[big], o[~big].sum())
        if eb[-1] == 0: eb, ob = eb[:-1], ob[:-1]
        print(f"{name} p{pi} seed{seed} N={int(n)} TV={0.5*np.abs(c/n - tgt).sum():.4f} chi2 p={stats.chisquare(ob, eb).pvalue:.3f} masked={int(c[~sup].sum())}", flush=True)
