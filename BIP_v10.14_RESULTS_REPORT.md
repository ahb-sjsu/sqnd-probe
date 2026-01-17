# BIP v10.12 Experiment Results Report

**Date:** 2026-01-16
**Run:** BIP_v10_12_01-16-2026_02.ipynb
**Next Version:** v10.14.1

---

## Executive Summary

Cross-cultural moral transfer **failed** in all targeted experiments, achieving at-chance or below-chance performance. However, mixed-language training shows strong signal (3.7x chance), indicating moral structure exists but is confounded with linguistic/cultural features. The model learns *language-specific* moral patterns rather than *universal* ones.

---

## Environment

| Parameter | Value |
|-----------|-------|
| GPU | NVIDIA L4 (23.8 GB VRAM) |
| Backbone | LaBSE (sentence-transformers/LaBSE) |
| Batch Size | 1024 |
| Corpus Size | 109,294 passages |
| Languages | 11 (english, sanskrit, pali, hebrew, arabic, french, classical_chinese, spanish, greek, aramaic, latin) |

### GPU Utilization Issue
Only **1.9-3.8 GB** (8%) of available 23.8 GB was used. Batch sizes need significant increase.

---

## Corpus Distribution

| Language | Passages | Status |
|----------|----------|--------|
| english | 50,000 | OK |
| sanskrit | 15,000 | OK |
| pali | 10,000 | OK |
| hebrew | 7,985 | OK |
| arabic | 6,235 | OK |
| french | 5,000 | OK |
| classical_chinese | 4,449 | OK |
| spanish | 4,320 | OK |
| greek | 3,157 | OK |
| aramaic | 2,015 | OK |
| latin | 1,133 | OK |
| **TOTAL** | **109,294** | |

---

## Cross-Cultural Transfer Results

### Targeted Experiments

| Experiment | Train | Test | Bond F1 | vs Chance | Status |
|------------|-------|------|---------|-----------|--------|
| hebrew_to_others | hebrew | all-other | 0.088 | 0.9x | FAIL |
| semitic_to_indic | semitic | indic | 0.079 | 0.8x | FAIL |
| confucian_to_buddhist | confucian | buddhist | 0.091 | 0.9x | FAIL |
| ancient_to_modern | ancient | modern | 0.096 | 1.0x | WEAK |
| east_to_west | east-asia | western | 0.135 | 1.3x | WEAK |
| semitic_to_chinese | semitic | chinese | 0.060 | 0.6x | FAIL |
| jewish_to_islamic | hebrew | arabic | 0.079 | 0.8x | FAIL |
| stoic_to_confucian | stoic | confucian | 0.052 | 0.5x | FAIL |
| hindu_to_buddhist | hindu | buddhist | 0.105 | 1.0x | WEAK |

### Mixed Baseline (Control)

| Metric | Value |
|--------|-------|
| Bond F1 (macro) | **0.374** |
| vs Chance | **3.7x** |
| Bond Accuracy | 66.4% |
| Language Accuracy | 12.8% (target: ~20%) |

#### Per-Language Breakdown (Mixed Baseline)

| Language | F1 | n |
|----------|-----|------|
| english | 0.376 | 25,467 |
| spanish | 0.333 | 821 |
| classical_chinese | 0.258 | 803 |
| greek | 0.249 | 590 |
| latin | 0.246 | 202 |
| sanskrit | 0.193 | 2,692 |
| french | 0.165 | 889 |
| arabic | 0.150 | 1,168 |
| pali | 0.124 | 1,808 |
| hebrew | 0.102 | 1,553 |
| aramaic | 0.100 | 414 |

---

## Linear Probe Analysis

Testing whether representations are language/period invariant:

| Probe | Accuracy | Chance | Interpretation |
|-------|----------|--------|----------------|
| Language | 98.8% | 14.3% | **NOT invariant** - encodes language |
| Period | 95.0% | 14.3% | **NOT invariant** - encodes time period |

**Conclusion:** LaBSE embeddings strongly preserve linguistic and temporal information. The adversarial training (λ=0.30) is insufficient to remove these confounds.

---

## Geometric Analysis

| Axis | Result | Status |
|------|--------|--------|
| Obligation/Permission | 100% transfer accuracy | STRONG |
| Harm/Care | r=0.323 correlation with obl/perm | CORRELATED |
| Role Swap Consistency | 0.594 ± 0.106 | VARIABLE |
| PCA Components (90% var) | 4 | HIGH-DIM |

**Interpretation:** Deontic structure (obligation vs permission) transfers well across languages, suggesting this axis is more universal than specific moral content categories.

---

## Diagnosis

### Root Cause
LaBSE embeddings preserve too much linguistic/cultural signal. The model learns to classify moral bonds *within* each language but cannot transfer *across* languages because:

1. **Adversarial λ too weak** (0.30) - language signal overwhelms moral signal
2. **Encoder frozen** - cannot adapt representations to be more invariant
3. **Low prescriptive density** - only 8-18% of passages are prescriptive
4. **No parallel anchors** - no translated texts to force cross-lingual alignment

### Evidence
- Mixed baseline works (3.7x) because it sees all languages during training
- Cross-cultural transfer fails because test language was never seen
- Linear probes achieve 98.8% language accuracy = representations are language-specific

---

## Recommended Changes for v10.14.1

### 1. Increase Adversarial Strength
```python
ADV_LAMBDA_MAX = 0.50  # was 0.30
ADV_LAMBDA_WARMUP = 3  # was 5 (reach max faster)
```

### 2. Increase GPU Utilization
```python
# L4/A100 batch sizes (currently using only 8% of 23.8GB)
LaBSE: 2048 → 4096
mBERT: 2048 → 4096
XLM-R: 1024 → 2048
```

### 3. Add Prescriptive Filtering
Pre-filter passages to those with higher prescriptive probability before training.

### 4. Consider Unfreezing Top Layers
Allow fine-tuning of top 2-4 transformer layers to adapt representations.

### 5. Add Cross-Lingual Anchors
Include parallel translations (e.g., Bible in multiple languages) to create explicit alignment points.

---

## Files Saved

| File | Location |
|------|----------|
| all_splits.json | /content/drive/MyDrive/BIP_v10.14/ |
| best_hebrew_to_others.pt | /content/drive/MyDrive/BIP_v10.14/ |
| best_semitic_to_indic.pt | /content/drive/MyDrive/BIP_v10.14/ |
| best_confucian_to_buddhist.pt | /content/drive/MyDrive/BIP_v10.14/ |
| best_ancient_to_modern.pt | /content/drive/MyDrive/BIP_v10.14/ |
| best_mixed_baseline.pt | /content/drive/MyDrive/BIP_v10.14/ |

---

## Next Steps

1. **v10.14.1**: Implement recommended parameter changes
2. **Re-run**: Execute full pipeline with new settings
3. **Evaluate**: Compare cross-cultural transfer performance
4. **Iterate**: If still failing, consider unfreezing encoder layers

---

*Report generated: 2026-01-16*
