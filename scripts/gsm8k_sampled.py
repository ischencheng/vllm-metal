"""Sampled GSM8K accuracy against a running vllm-metal server.

Chat completions with the model's recommended sampling (temperature 0.6,
top_k 20, top_p 0.95), thinking off, concurrency C. Prints accuracy and wall
time; writes per-question records to <out>.jsonl.

Usage: python gsm8k_sampled.py <test.jsonl> <num_questions> <concurrency> <out_prefix> [port]
"""

import concurrent.futures as cf
import json
import re
import sys
import time
import urllib.request

PATH, NQ, CONC, OUT = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
PORT = int(sys.argv[5]) if len(sys.argv) > 5 else 8011
URL = f"http://127.0.0.1:{PORT}/v1/chat/completions"


def gold(answer: str) -> str:
    return answer.split("####")[-1].strip().replace(",", "")


def pred(text: str) -> str | None:
    m = re.findall(r"answer is\s*\$?(-?[\d,]*\.?\d+)", text)
    nums = m or re.findall(r"-?[\d,]*\.?\d+", text)
    return nums[-1].replace(",", "").rstrip(".") if nums else None


def ask(q: str) -> str:
    body = {
        "model": "bench",
        "messages": [
            {
                "role": "user",
                "content": q
                + "\nSolve step by step. End with 'The answer is N.' where N is a number.",
            }
        ],
        "max_tokens": 512,
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        URL, json.dumps(body).encode(), {"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)["choices"][0]["message"]["content"]


def main():
    items = [json.loads(line) for line in open(PATH)][:NQ]
    t0 = time.perf_counter()
    with cf.ThreadPoolExecutor(CONC) as ex:
        outs = list(ex.map(lambda it: ask(it["question"]), items))
    wall = time.perf_counter() - t0
    correct = 0
    with open(OUT + ".jsonl", "w") as f:
        for it, out in zip(items, outs, strict=True):
            p, g = pred(out), gold(it["answer"])
            ok = p is not None and abs(float(p) - float(g)) < 1e-6
            correct += ok
            f.write(json.dumps({"gold": g, "pred": p, "ok": ok, "out": out}) + "\n")
    print(f"{OUT}: accuracy {correct}/{len(items)} = {correct / len(items):.3f}, wall {wall:.1f} s")


if __name__ == "__main__":
    main()
