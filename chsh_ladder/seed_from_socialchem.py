#!/usr/bin/env python3
"""Seed CHSH scenarios from Social Chemistry 101 real situations.

A CHSH scenario needs: two parties, and two INCOMPATIBLE (cross-type) moral framings each.
Social Chem gives both for free: a situation with >=2 characters and >=2 distinct moral
foundations across its rules-of-thumb. The two characters become the parties; the two most
prevalent, distinct foundations become the two measurement settings (framings). This yields
a large, ecologically-valid battery instead of a few hand-designed scenarios -- which blunts
the "scenarios were rigged" objection.

    SOCIALCHEM=/tmp/social-chem-101/social-chem-101.v1.0.tsv python seed_from_socialchem.py
"""
from __future__ import annotations

import csv
import json
import os
from collections import defaultdict

TSV = os.environ.get("SOCIALCHEM", "/tmp/social-chem-101/social-chem-101.v1.0.tsv")

# foundation -> a short moral-lens framing (the measurement setting)
_FRAMING = {
    "care-harm": "through the lens of care and harm (does preventing harm create a duty?)",
    "fairness-cheating": "through the lens of fairness (does desert or equal treatment create a claim?)",
    "loyalty-betrayal": "through the lens of loyalty (does the relationship create a binding tie?)",
    "authority-subversion": "through the lens of authority and respect (does role or rule create an obligation?)",
    "sanctity-degradation": "through the lens of dignity and purity (does respect for the person create a duty?)",
}
# pairs we prefer as most likely to be genuinely incompatible (pull to different deontic types)
_TENSION = {frozenset(p) for p in [
    ("care-harm", "fairness-cheating"), ("loyalty-betrayal", "fairness-cheating"),
    ("care-harm", "authority-subversion"), ("loyalty-betrayal", "care-harm"),
    ("authority-subversion", "fairness-cheating"), ("sanctity-degradation", "care-harm"),
]}

_CHAR = {  # normalize the char label to a readable name
    "narrator": "the narrator", "my friend": "the friend", "a friend": "the friend",
    "my mom": "the mother", "my dad": "the father", "my boss": "the boss",
    "my girlfriend": "the partner", "my boyfriend": "the partner", "someone": "the other person",
}


def _name(raw):
    raw = (raw or "").strip().lower()
    return _CHAR.get(raw, raw.replace("my ", "the ").replace("a ", "the ") or "the other person")


def load_situations(tsv, limit=None):
    sits = defaultdict(lambda: {"text": "", "chars": [], "founds": defaultdict(int)})
    with open(tsv, encoding="utf-8") as f:
        rd = csv.DictReader(f, delimiter="\t")
        for i, row in enumerate(rd):
            if limit and i >= limit:
                break
            sid = row.get("situation-short-id") or row.get("situation")
            s = sits[sid]
            if not s["text"]:
                s["text"] = (row.get("situation") or "").strip()
                s["chars"] = [c for c in (row.get("characters") or "").split("|") if c]
            for fnd in (row.get("rot-moral-foundations") or "").split("|"):
                if fnd in _FRAMING:
                    s["founds"][fnd] += 1
    return sits


def seed(tsv, max_scenarios=40):
    out = []
    for sid, s in load_situations(tsv).items():
        founds = sorted(s["founds"], key=lambda k: -s["founds"][k])
        chars = s["chars"]
        if len(chars) < 2 or len(founds) < 2 or not s["text"]:
            continue
        f0, f1 = founds[0], founds[1]
        tension = frozenset((f0, f1)) in _TENSION
        a, b = _name(chars[0]), _name(chars[1])
        if a == b:
            continue
        out.append({
            "name": f"sc_{sid}", "setup": s["text"][:400], "agent_a": a, "agent_b": b,
            "a_settings": [_FRAMING[f0], _FRAMING[f1]],
            "b_settings": [_FRAMING[f0], _FRAMING[f1]],
            "foundations": [f0, f1], "tension_pair": tension,
        })
    # prefer genuine-tension pairs first (more likely incompatible = higher chance of a signal)
    out.sort(key=lambda x: (not x["tension_pair"], -len(x["setup"])))
    return out[:max_scenarios]


def main():
    if not os.path.exists(TSV):
        print(f"Social Chemistry TSV not found at {TSV}."); return
    scen = seed(TSV)
    with open("seeded_scenarios.json", "w", encoding="utf-8") as f:
        json.dump(scen, f, indent=1)
    n_tension = sum(x["tension_pair"] for x in scen)
    print(f"seeded {len(scen)} CHSH scenarios ({n_tension} genuine-tension foundation pairs) "
          f"-> seeded_scenarios.json")
    for x in scen[:3]:
        print(f"\n[{x['name']}] {x['foundations']} tension={x['tension_pair']}")
        print(f"  parties: {x['agent_a']} / {x['agent_b']}")
        print(f"  setup: {x['setup'][:140]}...")
        print(f"  setting0: {x['a_settings'][0][:60]}")
        print(f"  setting1: {x['a_settings'][1][:60]}")


if __name__ == "__main__":
    main()
