"""Summarize vllm bench serve JSONs in bench/ into a markdown table."""
import glob, json, os, re, sys
from collections import defaultdict
D = os.path.dirname(os.path.abspath(__file__))
rows = defaultdict(dict)
for f in sorted(glob.glob(os.path.join(D, "bench", "*.json"))):
    m = re.match(r"([a-z])(\d)_(before|after)_(.+)\.json", os.path.basename(f))
    if not m: continue
    model, rnd, arm, label = m.groups()
    r = json.load(open(f))
    rows[(model, label)][(arm, rnd)] = r
names = {"r": "Qwen3-4B-4bit", "h": "Qwen3.5-0.8B", "g": "gemma-4-e2b-it-4bit"}
print("| model | config | arm | median TPOT r1 / r2 (ms) | P99 TPOT r1 / r2 | median TTFT r1 / r2 | output tok/s r1 / r2 | completed, output tokens |")
print("|---|---|---|---|---|---|---|---|")
fails = 0
for (model, label), arms in sorted(rows.items(), key=lambda kv: ("rhg".index(kv[0][0]), kv[0][1])):
    for arm in ("before", "after"):
        rs = [arms.get((arm, k)) for k in ("1", "2")]
        if not all(rs): continue
        f = lambda key, fmt="{:.2f}": " / ".join(fmt.format(x[key]) for x in rs)
        fails += sum(x.get("failed", 0) for x in rs)
        print(f"| {names[model]} | {label} | {arm} | {f('median_tpot_ms')} | {f('p99_tpot_ms')} | {f('median_ttft_ms')} | {f('output_throughput')} | {rs[0]['completed']}, {rs[0]['total_output_tokens']} |")
print(f"\nfailed requests across all runs: {fails}")
print("\nchange in median TPOT (mean of two rounds):")
for (model, label), arms in sorted(rows.items(), key=lambda kv: ("rhg".index(kv[0][0]), kv[0][1])):
    try:
        mm = sum(arms[("before", k)]["median_tpot_ms"] for k in "12") / 2
        pp = sum(arms[("after", k)]["median_tpot_ms"] for k in "12") / 2
    except KeyError: continue
    print(f"- {names[model]} {label}: {mm:.2f} -> {pp:.2f} ms ({(pp/mm-1)*100:+.1f}%)")
