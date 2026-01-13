#@title 11. Comprehensive Final Analysis { display-mode: "form" }
#@markdown **Honest evaluation reconciling probe results with transfer performance**
#@markdown 
#@markdown This cell fixes the metric confusion and provides rigorous assessment.

import json
import numpy as np
from collections import defaultdict

print("="*70)
print("BIP v10.2 - COMPREHENSIVE FINAL ANALYSIS")
print("="*70)

# ============================================================================
# SECTION 1: Reconcile all results
# ============================================================================

print("\n" + "─"*70)
print("1. RESULTS RECONCILIATION")
print("─"*70)

# Merge probe results with training results
final_analysis = {}

for split_name, train_result in all_results.items():
    analysis = {
        'split': split_name,
        'bond_f1': train_result['bond_f1_macro'],
        'bond_f1_vs_chance': train_result['bond_f1_macro'] / 0.1,  # 10 classes = 10% chance
        'training_lang_acc': train_result['language_acc'],  # From adversarial head (NOT the invariance test)
        'per_language_f1': train_result.get('per_language_f1', {}),
    }
    
    # Add probe results if available (THIS is the real invariance test)
    if 'probe_results' in dir() and split_name in probe_results:
        pr = probe_results[split_name]
        analysis['probe_lang_acc'] = pr['language_acc']
        analysis['probe_lang_chance'] = pr['language_chance']
        analysis['probe_period_acc'] = pr['period_acc']
        analysis['probe_period_chance'] = pr['period_chance']
        analysis['probe_tested'] = True
    else:
        analysis['probe_tested'] = False
    
    final_analysis[split_name] = analysis

# ============================================================================
# SECTION 2: Apply CORRECT evaluation criteria
# ============================================================================

print("\n" + "─"*70)
print("2. EVALUATION CRITERIA")
print("─"*70)

print("""
BIP Success requires BOTH:
  (A) Transfer: Bond F1 > 1.5× chance (>0.15) on held-out domain
  (B) Invariance: Probe can't recover language from z
      - Probe accuracy < chance + 15 percentage points
      
Failure modes:
  - High transfer + High probe = Model uses language features (LEAK)
  - Low transfer + Low probe = Model learned nothing useful
  - High transfer + Low probe = TRUE INVARIANT TRANSFER ✓
""")

TRANSFER_THRESHOLD = 1.5  # Must beat chance by 1.5x
INVARIANCE_MARGIN = 0.15   # Probe must be within 15pp of chance

# ============================================================================
# SECTION 3: Evaluate each split with CORRECT metrics
# ============================================================================

print("\n" + "─"*70)
print("3. PER-SPLIT EVALUATION")
print("─"*70)

verdicts = {}

for split_name, a in final_analysis.items():
    print(f"\n{'='*50}")
    print(f"SPLIT: {split_name}")
    print('='*50)
    
    # Transfer assessment
    transfer_ratio = a['bond_f1_vs_chance']
    transfer_ok = transfer_ratio >= TRANSFER_THRESHOLD
    
    print(f"\n  TRANSFER:")
    print(f"    Bond F1:        {a['bond_f1']:.3f}")
    print(f"    vs Chance:      {transfer_ratio:.1f}× (threshold: {TRANSFER_THRESHOLD}×)")
    print(f"    Status:         {'✓ PASS' if transfer_ok else '✗ FAIL'}")
    
    # Per-language breakdown
    if a['per_language_f1']:
        print(f"\n    Per-language F1:")
        for lang, m in sorted(a['per_language_f1'].items(), key=lambda x: -x[1]['n']):
            marker = "←train" if 'hebrew' in split_name and lang == 'hebrew' else ""
            print(f"      {lang:18s}: {m['f1']:.3f} (n={m['n']:,}) {marker}")
    
    # Invariance assessment - USE PROBE RESULTS, NOT TRAINING ACCURACY
    print(f"\n  INVARIANCE:")
    
    if a['probe_tested']:
        probe_acc = a['probe_lang_acc']
        probe_chance = a['probe_lang_chance']
        invariance_threshold = probe_chance + INVARIANCE_MARGIN
        invariance_ok = probe_acc < invariance_threshold
        
        print(f"    Probe accuracy: {probe_acc:.1%}")
        print(f"    Chance level:   {probe_chance:.1%}")
        print(f"    Threshold:      {invariance_threshold:.1%} (chance + {INVARIANCE_MARGIN:.0%})")
        print(f"    Status:         {'✓ INVARIANT' if invariance_ok else '✗ LANGUAGE LEAKING'}")
        
        # Period invariance too
        period_ok = a['probe_period_acc'] < a['probe_period_chance'] + INVARIANCE_MARGIN
        print(f"\n    Period probe:   {a['probe_period_acc']:.1%} (chance: {a['probe_period_chance']:.1%})")
        print(f"    Period status:  {'✓ INVARIANT' if period_ok else '✗ PERIOD LEAKING'}")
    else:
        print(f"    ⚠️  NOT PROBE-TESTED")
        print(f"    Training adversarial acc: {a['training_lang_acc']:.1%}")
        print(f"    (This is NOT an invariance test - adversarial head SHOULD be high)")
        invariance_ok = None  # Unknown
    
    # Overall verdict for this split
    if a['probe_tested']:
        if transfer_ok and invariance_ok:
            verdict = "SUCCESS"
            explanation = "Transfer works AND representation is language-invariant"
        elif transfer_ok and not invariance_ok:
            verdict = "LEAK"
            explanation = "Transfer works BUT model encodes language (not invariant)"
        elif not transfer_ok and invariance_ok:
            verdict = "WEAK"
            explanation = "Representation is invariant BUT transfer is weak"
        else:
            verdict = "FAIL"
            explanation = "Neither transfer nor invariance achieved"
    else:
        if transfer_ok:
            verdict = "PARTIAL"
            explanation = "Transfer works but invariance not tested"
        else:
            verdict = "FAIL"
            explanation = "Transfer too weak"
    
    verdicts[split_name] = {
        'verdict': verdict,
        'explanation': explanation,
        'transfer_ok': transfer_ok,
        'invariance_ok': invariance_ok,
        'probe_tested': a['probe_tested'],
    }
    
    print(f"\n  VERDICT: {verdict}")
    print(f"    {explanation}")

# ============================================================================
# SECTION 4: Overall BIP Assessment
# ============================================================================

print("\n" + "─"*70)
print("4. OVERALL BIP ASSESSMENT")
print("─"*70)

successes = [k for k, v in verdicts.items() if v['verdict'] == 'SUCCESS']
leaks = [k for k, v in verdicts.items() if v['verdict'] == 'LEAK']
partials = [k for k, v in verdicts.items() if v['verdict'] == 'PARTIAL']
fails = [k for k, v in verdicts.items() if v['verdict'] in ['FAIL', 'WEAK']]

print(f"\n  SUCCESS (transfer + invariant):  {len(successes)}")
for s in successes:
    print(f"    • {s}")

print(f"\n  LEAK (transfer but NOT invariant): {len(leaks)}")
for s in leaks:
    print(f"    • {s}")

print(f"\n  PARTIAL (transfer, invariance unknown): {len(partials)}")
for s in partials:
    print(f"    • {s}")

print(f"\n  FAIL/WEAK: {len(fails)}")
for s in fails:
    print(f"    • {s}")

# Final verdict
print("\n" + "="*70)
print("FINAL VERDICT")
print("="*70)

if len(successes) >= 2:
    FINAL = "STRONGLY_SUPPORTED"
    MSG = "Multiple independent invariant transfer paths demonstrated"
elif len(successes) >= 1:
    FINAL = "SUPPORTED"
    MSG = "At least one invariant transfer path demonstrated"
elif len(leaks) > 0 and len(successes) == 0:
    FINAL = "CHALLENGED"
    MSG = "Transfer occurs but representations encode language - not truly invariant"
elif len(partials) >= 2:
    FINAL = "PROVISIONAL"
    MSG = "Transfer works but invariance not rigorously tested"
else:
    FINAL = "NOT_SUPPORTED"
    MSG = "Insufficient evidence for Bond Invariance Principle"

print(f"\n  {FINAL}")
print(f"\n  {MSG}")

# ============================================================================
# SECTION 5: Falsification Status
# ============================================================================

print("\n" + "─"*70)
print("5. FALSIFICATION STATUS")
print("─"*70)

falsified_claims = []
supported_claims = []
untested_claims = []

# Claim 1: Cross-linguistic transfer
cross_ling_splits = ['hebrew_to_others', 'semitic_to_non_semitic']
cross_ling_transfer = any(verdicts.get(s, {}).get('transfer_ok', False) for s in cross_ling_splits)
if cross_ling_transfer:
    supported_claims.append("Cross-linguistic moral transfer (>1.5× chance)")
else:
    falsified_claims.append("Cross-linguistic moral transfer")

# Claim 2: Language invariance
probe_tested_splits = [k for k, v in verdicts.items() if v['probe_tested']]
if probe_tested_splits:
    any_invariant = any(verdicts[s]['invariance_ok'] for s in probe_tested_splits)
    all_invariant = all(verdicts[s]['invariance_ok'] for s in probe_tested_splits)
    
    if all_invariant:
        supported_claims.append("Language-invariant representations (all tested splits)")
    elif any_invariant:
        supported_claims.append("Language-invariant representations (some splits)")
        falsified_claims.append("Universal language invariance (some splits leak)")
    else:
        falsified_claims.append("Language-invariant representations (all tested splits leak)")
else:
    untested_claims.append("Language-invariant representations (no probe tests run)")

# Claim 3: Temporal transfer
if 'ancient_to_modern' in verdicts:
    if verdicts['ancient_to_modern']['transfer_ok']:
        supported_claims.append("Ancient→Modern temporal transfer")
    else:
        falsified_claims.append("Ancient→Modern temporal transfer")

print("\n  ✓ SUPPORTED:")
for c in supported_claims:
    print(f"    • {c}")

print("\n  ✗ FALSIFIED:")
for c in falsified_claims:
    print(f"    • {c}")

print("\n  ? UNTESTED:")
for c in untested_claims:
    print(f"    • {c}")

# ============================================================================
# SECTION 6: Recommendations
# ============================================================================

print("\n" + "─"*70)
print("6. RECOMMENDATIONS FOR NEXT STEPS")
print("─"*70)

recommendations = []

if len(leaks) > 0:
    recommendations.append(
        "INCREASE ADVERSARIAL WEIGHT: Current model leaks language info. "
        "Try LANG_WEIGHT=0.1 or 0.5 instead of 0.01"
    )

if len([k for k, v in verdicts.items() if not v['probe_tested']]) > 0:
    recommendations.append(
        "RUN PROBES ON ALL SPLITS: Add ancient_to_modern and mixed_baseline to Cell 8 probe tests"
    )

untested_splits = [s for s in all_splits.keys() if s not in all_results]
if untested_splits:
    recommendations.append(
        f"SKIPPED SPLITS: {untested_splits} - likely due to small test sets. Add more data."
    )

if any(a['bond_f1'] < 0.15 for a in final_analysis.values()):
    recommendations.append(
        "WEAK TRANSFER: Some splits barely beat chance. "
        "Consider: more training data, longer training, or different model architecture"
    )

chinese_f1 = None
for a in final_analysis.values():
    if 'classical_chinese' in a.get('per_language_f1', {}):
        chinese_f1 = a['per_language_f1']['classical_chinese']['f1']
        break

if chinese_f1 is not None and chinese_f1 < 0.1:
    recommendations.append(
        f"CHINESE TRANSFER WEAK (F1={chinese_f1:.3f}): "
        "Chinese corpus is small (~200 samples). Add more classical texts or use data augmentation."
    )

print()
for i, rec in enumerate(recommendations, 1):
    print(f"  {i}. {rec}")
    print()

if not recommendations:
    print("  No critical issues found. Consider:")
    print("  • Publishing replication notebook")
    print("  • Running on additional language pairs")
    print("  • Testing with different encoder architectures")

# ============================================================================
# SECTION 7: Export Results
# ============================================================================

print("\n" + "─"*70)
print("7. EXPORTING RESULTS")
print("─"*70)

export_data = {
    'version': 'BIP_v10.2',
    'hardware': {
        'gpu': GPU_TIER if 'GPU_TIER' in dir() else 'unknown',
        'vram_gb': VRAM_GB if 'VRAM_GB' in dir() else 0,
    },
    'final_verdict': FINAL,
    'verdict_explanation': MSG,
    'per_split_verdicts': verdicts,
    'per_split_analysis': {k: {kk: vv for kk, vv in v.items() if kk != 'per_language_f1'} 
                          for k, v in final_analysis.items()},
    'claims': {
        'supported': supported_claims,
        'falsified': falsified_claims,
        'untested': untested_claims,
    },
    'experiment_time_minutes': (time.time() - EXPERIMENT_START) / 60 if 'EXPERIMENT_START' in dir() else None,
}

# Save locally
with open('results/final_analysis.json', 'w') as f:
    json.dump(export_data, f, indent=2, default=str)

# Save to Drive
if 'SAVE_DIR' in dir():
    import shutil
    shutil.copy('results/final_analysis.json', f'{SAVE_DIR}/final_analysis.json')
    print(f"  Saved to: {SAVE_DIR}/final_analysis.json")

print(f"  Saved to: results/final_analysis.json")

# ============================================================================
# SECTION 8: Summary Card
# ============================================================================

print("\n" + "="*70)
print("SUMMARY CARD (for README/paper)")
print("="*70)

print(f"""
┌─────────────────────────────────────────────────────────────────────┐
│  BIP v10.2 RESULTS                                                  │
├─────────────────────────────────────────────────────────────────────┤
│  Verdict: {FINAL:50s}     │
│                                                                     │
│  Splits Evaluated: {len(all_results)}                                                │
│  Successful (transfer + invariant): {len(successes)}                                 │
│  Leaking (transfer but encodes language): {len(leaks)}                               │
│                                                                     │
│  Key Metrics:                                                       │""")

for split_name, a in final_analysis.items():
    v = verdicts[split_name]
    probe_str = f"{a['probe_lang_acc']:.0%}" if a['probe_tested'] else "N/A"
    print(f"│    {split_name:25s} F1={a['bond_f1']:.3f}  Probe={probe_str:5s}  {v['verdict']:8s} │")

print(f"""│                                                                     │
│  Falsification: {len(falsified_claims)} claim(s) falsified, {len(supported_claims)} supported              │
└─────────────────────────────────────────────────────────────────────┘
""")

print("="*70)
print("Analysis complete!")
print("="*70)
