"""Compare the native MLX sampler and the torch sampler against the exact
masked distribution on real Qwen3-4B logits.

For each (prompt row, sampling config): compute the exact target in float64
with vLLM's semantics (temperature -> top-k by value -> min-p -> top-p), draw
N tokens through each production path, and report the total-variation
distance to the target, a chi-square p-value, and draws that land on masked
tokens. A numpy draw from the exact target gives the noise floor for N.

Usage: python distribution_parity.py <model_path> [N]
"""

import sys

import mlx.core as mx
import numpy as np
from mlx_lm import load
from scipy import stats
from vllm.sampling_params import SamplingParams
from vllm.v1.sample.sampler import Sampler

from vllm_metal.v1.sampling_batch import SamplingBatch, sample_from_logits

MODEL = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 20000
ROWS_PER_CALL = 500

PROMPTS = [
    "The capital of France is",
    "Write a haiku about the ocean:\n",
    "def fibonacci(n):\n    ",
    "Q: What is 17 * 23?\nA:",
    "Once upon a time, in a small village,",
    "List three primary colors:",
]
CONFIGS = {
    "t0.6_k20_p0.95": dict(temperature=0.6, top_k=20, top_p=0.95),
    "t1.0": dict(temperature=1.0),
    "t0.7_p0.9": dict(temperature=0.7, top_p=0.9),
    "t1.0_k50": dict(temperature=1.0, top_k=50),
    "t0.8_minp0.05": dict(temperature=0.8, min_p=0.05),
}


def exact_target(logits: np.ndarray, sp: SamplingParams) -> np.ndarray:
    x = logits.astype(np.float64) / sp.temperature
    if 0 < sp.top_k < x.size:
        kth = np.sort(x)[-sp.top_k]
        x = np.where(x < kth, -np.inf, x)
    if sp.min_p > 0:
        p = np.exp(x - x.max())
        p /= p.sum()
        x = np.where(p < sp.min_p * p.max(), -np.inf, x)
    if sp.top_p < 1.0:
        order = np.argsort(x)  # ascending, like vLLM
        p = np.exp(x[order] - x.max())
        p /= p.sum()
        cum = np.cumsum(p)
        drop = cum <= 1 - sp.top_p
        drop[-1] = False
        x[order[drop]] = -np.inf
    p = np.exp(x - x.max())
    return p / p.sum()


def tv_and_chi2(counts: np.ndarray, target: np.ndarray):
    n = counts.sum()
    tv = 0.5 * np.abs(counts / n - target).sum()
    support = target > 0
    exp = target[support] * n
    obs = counts[support]
    big = exp >= 5
    exp_b = np.append(exp[big], exp[~big].sum())
    obs_b = np.append(obs[big], obs[~big].sum())
    if exp_b[-1] == 0:
        exp_b, obs_b = exp_b[:-1], obs_b[:-1]
    if len(exp_b) < 2:
        return tv, float("nan"), int(counts[~support].sum())
    p = stats.chisquare(obs_b, exp_b).pvalue
    return tv, p, int(counts[~support].sum())


def main():
    model, tok = load(MODEL)
    vocab = model.args.vocab_size
    rows = []
    for prompt in PROMPTS:
        ids = mx.array(tok.encode(prompt))[None]
        logits = model(ids)[0, -1, :vocab].astype(mx.bfloat16)
        mx.eval(logits)
        rows.append(logits)

    sampler = Sampler()
    key = [mx.random.key(1234)]

    def next_key():
        key[0], sub = mx.random.split(key[0])
        return sub

    rng = np.random.default_rng(0)
    print(f"N={N} draws per arm; vocab={vocab}")
    print("cfg prompt support | TV native / torch / ideal | chi2 p native / torch / ideal | masked hits native / torch")
    worst = {"native": 0.0, "torch": 0.0, "ideal": 0.0}
    masked_total = {"native": 0, "torch": 0}
    pvals = {"native": [], "torch": [], "ideal": []}
    for cname, kw in CONFIGS.items():
        sp = SamplingParams(**kw)
        for pi, row in enumerate(rows):
            target = exact_target(np.array(row.astype(mx.float32)), sp)
            batch_logits = mx.contiguous(mx.tile(row[None], (ROWS_PER_CALL, 1)))
            mx.eval(batch_logits)
            sps = [sp] * ROWS_PER_CALL
            native, torch_draws = [], []
            for _ in range(N // ROWS_PER_CALL):
                t = SamplingBatch.native_decode_tokens(
                    batch_logits, sps, vocab_size=vocab, next_key=next_key
                )
                native.append(np.array(t))
                b = SamplingBatch(
                    sps, [[1]] * ROWS_PER_CALL, [[]] * ROWS_PER_CALL, vocab_size=vocab
                )
                # sample_from_logits may rewrite its input in place; pass a copy.
                fresh = mx.contiguous(batch_logits + 0)
                mx.eval(fresh)
                res = sample_from_logits(fresh, b, sampler)
                torch_draws.append(np.array(res.token_ids))
            arms = {
                "native": np.concatenate(native),
                "torch": np.concatenate(torch_draws),
                "ideal": rng.choice(target.size, size=N, p=target),
            }
            res = {}
            for arm, draws in arms.items():
                counts = np.bincount(draws, minlength=target.size).astype(np.float64)
                res[arm] = tv_and_chi2(counts, target)
                worst[arm] = max(worst[arm], res[arm][0])
                pvals[arm].append(res[arm][1])
            for arm in ("native", "torch"):
                masked_total[arm] += res[arm][2]
            print(
                f"{cname:15s} p{pi} {int((target > 0).sum()):6d} | "
                f"{res['native'][0]:.4f} / {res['torch'][0]:.4f} / {res['ideal'][0]:.4f} | "
                f"{res['native'][1]:.3f} / {res['torch'][1]:.3f} / {res['ideal'][1]:.3f} | "
                f"{res['native'][2]} / {res['torch'][2]}"
            )
    print("worst TV:", {k: round(v, 4) for k, v in worst.items()})
    print("masked hits:", masked_total)
    for arm, ps in pvals.items():
        ps = [p for p in ps if p == p]
        print(f"{arm}: chi2 p<0.01 in {sum(p < 0.01 for p in ps)}/{len(ps)}, median p {np.median(ps):.3f}")


if __name__ == "__main__":
    main()
