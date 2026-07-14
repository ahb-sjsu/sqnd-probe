#!/usr/bin/env python3
"""Full CHSH/CbD contextuality sweep across the NRP model capability ladder.

Emergence design: for each model, probe K real (Social-Chem-seeded) scenarios, N samples
per CHSH cell, concurrently. Per scenario compute the CbD contextuality CNT (signaling
already subtracted), a bootstrap p-value (is it robustly contextual?), and a
permutation-null CNT (false-positive baseline). Per model report the FDR-significant
fraction contextual and mean CNT vs the null. Emergence = that rising with capability.

Checkpointed: writes results_<model>.json per model, skips finished ones. Run detached.

    K=150 N=20 CONC=24 MODELS=gemma-small,gpt-oss,qwen3,kimi python run_ladder.py
"""
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import chsh_probe as C

BASE = os.environ.get("NRP_BASE", "https://ellm.nrp-nautilus.io/v1")
TOKEN = open(os.path.expanduser("~/.llmtoken")).read().strip()
HDR = {"Authorization": "Bearer " + TOKEN}
CONC = int(os.environ.get("CONC", "24"))
N = int(os.environ.get("N", "20"))
K = int(os.environ.get("K", "150"))
# reasoning models need room to emit the FINAL line; non-reasoning answer directly (fast).
REASONING = {"gpt-oss", "glm-5", "qwen3", "minimax-m2", "kimi"}
def maxtok_for(m):
    return int(os.environ.get("MAXTOK_REASON", "320")) if m in REASONING else int(os.environ.get("MAXTOK_FAST", "28"))
CAP_RANK = {"gemma-small-e4b": 1, "gemma-small": 2, "qwen3-small": 3, "qwen3-4bit": 4,
            "gemma": 5, "gpt-oss": 6, "glm-5": 7, "qwen3": 8, "minimax-m2": 9, "kimi": 10}
MODELS = os.environ.get("MODELS", "gemma-small,qwen3-small,gemma,gpt-oss,glm-5,qwen3,minimax-m2,kimi").split(",")


def call(model_name, prompt, temperature):
    base = {"model": model_name, "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature, "max_tokens": maxtok_for(model_name)}
    for body in ({**base, "chat_template_kwargs": {"enable_thinking": False}}, base):
        for attempt in range(4):
            try:
                r = requests.post(BASE + "/chat/completions", json=body, headers=HDR, timeout=120)
                if r.status_code == 200:
                    m = r.json()["choices"][0]["message"]
                    return (m.get("content") or m.get("reasoning") or "")
                if r.status_code in (429, 500, 502, 503):
                    time.sleep(2 * (attempt + 1)); continue
                if r.status_code == 400:
                    break
            except Exception:
                time.sleep(2)
    return ""


def load_scenarios():
    scen = [dict(name=s.name, setup=s.setup, agent_a=s.agent_a, agent_b=s.agent_b,
                 a_settings=list(s.a_settings), b_settings=list(s.b_settings)) for s in C.SCENARIOS]
    if os.path.exists("seeded_scenarios.json"):
        scen += json.load(open("seeded_scenarios.json", encoding="utf-8"))
    return scen[:K]


def probe_all(model_name, scenarios):
    counts = [{(ai, bj): {} for ai in (0, 1) for bj in (0, 1)} for _ in scenarios]
    bad = [0] * len(scenarios)
    lock = threading.Lock()
    tasks = [(si, ai, bj) for si in range(len(scenarios)) for ai in (0, 1) for bj in (0, 1) for _ in range(N)]

    def work(t):
        si, ai, bj = t
        sc = scenarios[si]
        prompt = C.QUESTION.format(setup=sc["setup"], a=sc["agent_a"], b=sc["agent_b"],
                                   a_frame=sc["a_settings"][ai], b_frame=sc["b_settings"][bj])
        out = C.parse_outcomes(call(model_name, prompt, 0.9))
        with lock:
            if out is None:
                bad[si] += 1
            else:
                counts[si][(ai, bj)][out] = counts[si][(ai, bj)].get(out, 0) + 1

    with ThreadPoolExecutor(max_workers=CONC) as ex:
        list(ex.map(work, tasks))
    return counts, bad


def analyze(model_name, scenarios, counts, bad):
    rows, pvals = [], []
    for si, sc in enumerate(scenarios):
        r = C.cbd_chsh(counts[si])
        p, cnt_boot = C.bootstrap_pvalue(counts[si], B=300, seed=si)
        null = C.permutation_null(counts[si], P=300, seed=si)
        rows.append({"name": sc["name"], "S": r["S"], "signaling": r["signaling"],
                     "CNT": r["CNT"], "p": p, "cnt_boot": cnt_boot,
                     "null_cnt": sum(null) / len(null), "bad": bad[si]})
        pvals.append(p)
    sig = C.bh_fdr(pvals, q=0.05)
    summary = {"model": model_name, "rank": CAP_RANK.get(model_name, 0), "n": len(rows),
               "frac_contextual_fdr": sum(sig) / len(rows),
               "mean_CNT": sum(x["CNT"] for x in rows) / len(rows),
               "null_mean_CNT": sum(x["null_cnt"] for x in rows) / len(rows),
               "mean_bad": sum(bad) / len(rows) / (4 * N)}
    json.dump({**summary, "rows": rows}, open(f"results_{model_name}.json", "w"), indent=1)
    return summary


def main():
    scen = load_scenarios()
    print(f"ladder sweep: {len(MODELS)} models x {len(scen)} scenarios x {N} samples x 4 cells; conc={CONC}", flush=True)
    for m in MODELS:
        if os.path.exists(f"results_{m}.json"):
            print(f"skip {m} (done)", flush=True); continue
        t0 = time.time()
        counts, bad = probe_all(m, scen)
        s = analyze(m, scen, counts, bad)
        print(f"{m:16} rank={s['rank']:2} n={s['n']} frac_ctx_FDR={s['frac_contextual_fdr']:.3f} "
              f"meanCNT={s['mean_CNT']:.4f} nullCNT={s['null_mean_CNT']:.4f} "
              f"bad={s['mean_bad']:.2f} t={time.time()-t0:.0f}s", flush=True)
    print("=== ladder sweep complete ===", flush=True)


if __name__ == "__main__":
    main()
