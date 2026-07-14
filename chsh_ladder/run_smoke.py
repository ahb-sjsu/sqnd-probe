#!/usr/bin/env python3
"""Smoke run: probe the CHSH battery on one NRP model, end-to-end, to validate the
probe -> parse -> CbD pipeline before scaling to the full ladder.

    MODEL=gpt-oss N=20 python run_smoke.py
"""
import os
import sys
import time

import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import chsh_probe as C

BASE = os.environ.get("NRP_BASE", "https://ellm.nrp-nautilus.io/v1")
TOKEN = open(os.path.expanduser("~/.llmtoken")).read().strip()
MODEL = os.environ.get("MODEL", "gpt-oss")
N = int(os.environ.get("N", "20"))
HDR = {"Authorization": "Bearer " + TOKEN}
MAXTOK = int(os.environ.get("MAXTOK", "700"))


def model(prompt, temperature):
    base = {"model": MODEL, "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature, "max_tokens": MAXTOK}
    for body in ({**base, "chat_template_kwargs": {"enable_thinking": False}}, base):
        for _ in range(3):
            try:
                r = requests.post(BASE + "/chat/completions", json=body, headers=HDR, timeout=90)
                if r.status_code == 200:
                    m = r.json()["choices"][0]["message"]
                    return (m.get("content") or m.get("reasoning") or "")
                if r.status_code == 400:
                    break  # drop the extra field, try plain body
            except Exception:
                time.sleep(2)
    return ""


def main():
    # one warm call to confirm auth + format
    probe = model("Reply with exactly: YES NO", 0.0)
    print(f"model={MODEL}  warmup reply={probe!r}")
    print(f"N={N} samples/cell; {len(C.SCENARIOS)} scenarios (reconstructed v4.1 battery)\n")
    ctx = 0
    for sc in C.SCENARIOS:
        counts, bad = C.probe_scenario(sc, model, n=N, temperature=0.9)
        r = C.cbd_chsh(counts)
        ctx += r["contextual"]
        print(f"  {sc.name:22} S={r['S']:.2f}  signaling={r['signaling']:.2f}  "
              f"CNT={r['CNT']:.3f}  contextual={r['contextual']}  bad={bad}/{4*N}")
    print(f"\nSUMMARY: {ctx}/{len(C.SCENARIOS)} scenarios CbD-contextual on {MODEL} "
          f"(smoke; expect ~0 at classical baseline).")


if __name__ == "__main__":
    main()
