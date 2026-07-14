#!/usr/bin/env python3
"""CHSH / CbD contextuality probe for moral judgment across a model capability ladder.

Reconstructs the v4.1 CHSH battery (Non_Abelian_SQND_Bond_2026_v4_1.md, section 5.5):
four two-agent "entangled" moral scenarios, each agent measured under two CROSS-TYPE
(incompatible) Hohfeldian framings, binary obligated? outcome. We estimate the joint
outcome distribution p(A,B | setting_A, setting_B) by repeated sampling, then compute the
CHSH statistic AND, crucially, the Contextuality-by-Default (CbD) measure so that
order-effect *signaling* is subtracted -- a raw |S|>2 that is really signaling does NOT
count as contextuality.

Cyclic-4 (CHSH) CbD criterion (Dzhafarov & Kujala): the system is contextual iff
    s_odd  -  Delta  >  2 ,
where s_odd = max over sign patterns with an odd number of minus signs of the sum of the
four correlators <Ai Bj>, and Delta = total signaling = sum over each measured content of
|difference of its expectation across the two contexts it appears in|. The contextuality
measure is CNT = max(0, s_odd - Delta - 2). Classical (Tsirelson-free) systems give CNT=0.

This module is backend-agnostic: pass any callable model(prompt)->text. A mock backend and
a self-test of the CbD math are included; the Atlas/NRP backends are wired in run_ladder.py.
"""
from __future__ import annotations

import itertools
import re
from dataclasses import dataclass, field


# ------------------------------------------------------------------ scenarios
@dataclass
class Scenario:
    name: str
    setup: str
    agent_a: str
    agent_b: str
    # two CROSS-TYPE framings per agent (incompatible measurements). Each framing is a
    # short instruction that fixes the moral lens under which the agent is judged.
    a_settings: tuple = field(default_factory=tuple)   # (framing0, framing1)
    b_settings: tuple = field(default_factory=tuple)


# Reconstructed from v4.1 sec.5.4-5.5: cross-type pairs are the ones that showed real
# non-commutativity (Truth->O vs Protection->C ; Integrity->L vs Compassion->O).
SCENARIOS = [
    Scenario(
        name="shared_secret",
        setup=("Dana and Erin jointly hold a confidential fact that, if disclosed, would "
               "protect a third party from harm but would break a promise of secrecy each made."),
        agent_a="Dana", agent_b="Erin",
        a_settings=("through the lens of TRUTH and disclosure (does honesty create a duty?)",
                    "through the lens of PROTECTION and loyalty (does the promise create a claim?)"),
        b_settings=("through the lens of TRUTH and disclosure (does honesty create a duty?)",
                    "through the lens of PROTECTION and loyalty (does the promise create a claim?)"),
    ),
    Scenario(
        name="joint_promise",
        setup=("Sam and Val together promised to help a friend move, but each has since "
               "learned of a competing obligation (one to family, one to work)."),
        agent_a="Sam", agent_b="Val",
        a_settings=("through the lens of INTEGRITY and keeping one's word (an obligation)",
                    "through the lens of COMPASSION and competing need (a discretionary liberty)"),
        b_settings=("through the lens of INTEGRITY and keeping one's word (an obligation)",
                    "through the lens of COMPASSION and competing need (a discretionary liberty)"),
    ),
    Scenario(
        name="collaborative_harm",
        setup=("Kim and Lee each contributed one necessary step to an outcome that harmed a "
               "bystander; neither step alone would have caused the harm."),
        agent_a="Kim", agent_b="Lee",
        a_settings=("through the lens of CAUSAL responsibility (does contribution create a duty to repair?)",
                    "through the lens of SHARED agency (is responsibility diffused, leaving only a weak claim?)"),
        b_settings=("through the lens of CAUSAL responsibility (does contribution create a duty to repair?)",
                    "through the lens of SHARED agency (is responsibility diffused, leaving only a weak claim?)"),
    ),
    Scenario(
        name="entangled_beneficiary",
        setup=("Ana and Ben will each benefit if a scarce resource is allocated to their shared "
               "project rather than to an unrelated stranger in greater need."),
        agent_a="Ana", agent_b="Ben",
        a_settings=("through the lens of DESERT and prior effort (does contribution create a claim?)",
                    "through the lens of NEED and impartiality (does greater need impose an obligation to yield?)"),
        b_settings=("through the lens of DESERT and prior effort (does contribution create a claim?)",
                    "through the lens of NEED and impartiality (does greater need impose an obligation to yield?)"),
    ),
]

QUESTION = (
    "{setup}\n\n"
    "Evaluate {a} {a_frame}. Evaluate {b} {b_frame}.\n"
    "For each, decide: is that person under a binding moral OBLIGATION to act, or NOT?\n"
    "Respond with ONLY this line and nothing else:\n"
    "FINAL A=<YES or NO> B=<YES or NO>")


def build_prompt(sc: Scenario, ai: int, bj: int) -> str:
    return QUESTION.format(setup=sc.setup, a=sc.agent_a, b=sc.agent_b,
                           a_frame=sc.a_settings[ai], b_frame=sc.b_settings[bj])


def parse_outcomes(text: str):
    """Return (a,b) in {+1,-1} or None. +1 = OBLIGATION (YES). Robust to reasoning models:
    prefer the LAST explicit A=/B= tags (the final answer), else the last two YES/NO tokens."""
    if not text:
        return None
    t = text.upper()
    am = re.findall(r"A\s*=\s*(YES|NO)", t)
    bm = re.findall(r"B\s*=\s*(YES|NO)", t)
    if am and bm:
        return (1 if am[-1] == "YES" else -1), (1 if bm[-1] == "YES" else -1)
    yn = re.findall(r"\b(YES|NO)\b", t)
    if len(yn) >= 2:
        return (1 if yn[-2] == "YES" else -1), (1 if yn[-1] == "YES" else -1)
    return None


# ------------------------------------------------------------------ CbD / CHSH math
def correlators_and_marginals(counts):
    """counts[(ai,bj)] = {(a,b): n}. Returns E[(ai,bj)], and per-content marginals
    mA[(ai,bj)] = <A> in that context, mB[(ai,bj)] = <B>."""
    E, mA, mB = {}, {}, {}
    for ctx, d in counts.items():
        n = sum(d.values())
        if n == 0:
            E[ctx] = mA[ctx] = mB[ctx] = 0.0
            continue
        E[ctx] = sum(a * b * c for (a, b), c in d.items()) / n
        mA[ctx] = sum(a * c for (a, b), c in d.items()) / n
        mB[ctx] = sum(b * c for (a, b), c in d.items()) / n
    return E, mA, mB


def cbd_chsh(counts):
    """CHSH + CbD contextuality for the cyclic-4 system with settings A in {0,1}, B in {0,1}.
    Returns dict with S (best odd-sign correlator sum), signaling Delta, and CNT."""
    ctxs = [(0, 0), (0, 1), (1, 0), (1, 1)]
    E, mA, mB = correlators_and_marginals(counts)
    c = [E[x] for x in ctxs]
    # s_odd: max over sign patterns with an odd number of -1 of sum(sign_i * c_i)
    s_odd = 0.0
    for signs in itertools.product([1, -1], repeat=4):
        if signs.count(-1) % 2 == 1:
            s_odd = max(s_odd, abs(sum(s * v for s, v in zip(signs, c))))
    # signaling Delta: A0 appears in ctx(0,0),(0,1); A1 in (1,0),(1,1); B0 in (0,0),(1,0); B1 in (0,1),(1,1)
    dA0 = abs(mA[(0, 0)] - mA[(0, 1)])
    dA1 = abs(mA[(1, 0)] - mA[(1, 1)])
    dB0 = abs(mB[(0, 0)] - mB[(1, 0)])
    dB1 = abs(mB[(0, 1)] - mB[(1, 1)])
    delta = dA0 + dA1 + dB0 + dB1
    cnt = max(0.0, s_odd - delta - 2.0)
    return {"S": s_odd, "signaling": delta, "CNT": cnt,
            "contextual": cnt > 0, "correlators": dict(zip([str(x) for x in ctxs], c))}


# ------------------------------------------------------------------ probing loop
def probe_scenario(sc: Scenario, model, n=40, temperature=0.9):
    """model(prompt, temperature) -> text. Returns counts[(ai,bj)] = {(a,b): n}, plus n_bad."""
    counts = {(ai, bj): {} for ai in (0, 1) for bj in (0, 1)}
    n_bad = 0
    for ai, bj in counts:
        p = build_prompt(sc, ai, bj)
        for _ in range(n):
            out = parse_outcomes(model(p, temperature))
            if out is None:
                n_bad += 1
                continue
            counts[(ai, bj)][out] = counts[(ai, bj)].get(out, 0) + 1
    return counts, n_bad


# ------------------------------------------------------------------ significance stats
import random as _random


def _cells_as_lists(counts):
    return {ctx: [o for o, c in d.items() for _ in range(int(c))] for ctx, d in counts.items()}


def bootstrap_pvalue(counts, B=400, seed=0):
    """One-sided p that the scenario is NOT contextual, by resampling samples within each
    cell. Small p => robustly contextual. Returns (p, cnt_mean)."""
    rng = _random.Random(seed)
    cells = _cells_as_lists(counts)
    if any(len(v) == 0 for v in cells.values()):
        return 1.0, 0.0
    hits, csum = 0, 0.0
    for _ in range(B):
        rc = {}
        for ctx, lst in cells.items():
            d = {}
            for _ in range(len(lst)):
                o = rng.choice(lst)
                d[o] = d.get(o, 0) + 1
            rc[ctx] = d
        r = cbd_chsh(rc)
        hits += r["CNT"] > 0
        csum += r["CNT"]
    return 1.0 - hits / B, csum / B


def permutation_null(counts, P=400, seed=0):
    """Null CNT distribution: pool all outcomes and re-deal to cells (destroys cell
    structure), preserving cell sizes. Returns list of null CNTs."""
    rng = _random.Random(seed)
    pool = [o for d in counts.values() for o, c in d.items() for _ in range(int(c))]
    sizes = {ctx: sum(d.values()) for ctx, d in counts.items()}
    if not pool:
        return [0.0]
    null = []
    for _ in range(P):
        rng.shuffle(pool)
        i, rc = 0, {}
        for ctx, sz in sizes.items():
            d = {}
            for o in pool[i:i + int(sz)]:
                d[o] = d.get(o, 0) + 1
            rc[ctx] = d
            i += int(sz)
        null.append(cbd_chsh(rc)["CNT"])
    return null


def bh_fdr(pvals, q=0.05):
    """Benjamini-Hochberg. Returns a boolean list: which hypotheses are significant."""
    idx = sorted(range(len(pvals)), key=lambda i: pvals[i])
    n = len(pvals)
    k = 0
    for rank, i in enumerate(idx, 1):
        if pvals[i] <= q * rank / n:
            k = rank
    sig = [False] * n
    for i in idx[:k]:
        sig[i] = True
    return sig


# ------------------------------------------------------------------ self-test of the math
def _selftest():
    def corr_to_counts(Emat, mAvec, mBvec, N=100000):
        # build counts reproducing given E and marginals per context (2x2 outcome dist)
        counts = {}
        for k, (ai, bj) in enumerate([(0, 0), (0, 1), (1, 0), (1, 1)]):
            E, mA, mB = Emat[k], mAvec[k], mBvec[k]
            # p(a,b): p_pp - p_pm - p_mp + p_mm = E ; p_pp+p_pm = (1+mA)/2 ; p_pp+p_mp=(1+mB)/2
            p_pp = (1 + mA + mB + E) / 4
            p_pm = (1 + mA - mB - E) / 4
            p_mp = (1 - mA + mB - E) / 4
            p_mm = (1 - mA - mB + E) / 4
            counts[(ai, bj)] = {(1, 1): p_pp * N, (1, -1): p_pm * N, (-1, 1): p_mp * N, (-1, -1): p_mm * N}
        return counts
    import math
    # (1) classical, no signaling -> CNT 0
    c = corr_to_counts([0.5, 0.5, 0.5, -0.5], [0, 0, 0, 0], [0, 0, 0, 0])
    r = cbd_chsh(c); assert abs(r["S"] - 2.0) < 1e-6 and r["CNT"] < 1e-6, r
    # (2) Tsirelson quantum, no signaling -> CNT ~ 0.83
    q = 1 / math.sqrt(2)
    c = corr_to_counts([q, q, q, -q], [0, 0, 0, 0], [0, 0, 0, 0])
    r = cbd_chsh(c); assert abs(r["S"] - 2 * math.sqrt(2)) < 1e-3 and r["contextual"] and abs(r["CNT"] - (2*math.sqrt(2)-2)) < 1e-3, r
    # (3) pure signaling faking |S|>2 -> CbD must NOT flag contextuality
    c = corr_to_counts([0.9, 0.9, 0.9, -0.9], [0.8, -0.8, 0.8, -0.8], [0, 0, 0, 0])
    r = cbd_chsh(c); assert r["S"] > 2 and r["CNT"] < 1e-6, ("signaling should not count as contextual", r)
    print("CbD/CHSH self-test PASSED (classical CNT=0, Tsirelson CNT=0.83, signaling not flagged)")


if __name__ == "__main__":
    _selftest()
