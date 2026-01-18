# BIP v10.14.4 Results Analysis

**Date:** January 2026
**Version:** v10.14.4
**Hardware:** NVIDIA L4 (22.5 GB VRAM), 56.9 GB RAM

---

## Executive Summary

BIP v10.14.4 successfully trained 10 cross-cultural transfer models on 109,769 passages across 11 languages. The geometric analysis reveals **strong deontic structure** (100% obligation/permission transfer) but the linear probe test shows the model has **not achieved invariance** to surface features (language: 98.6%, period: 95.0% accuracy when chance is 14.3% and 12.5% respectively).

**Key Finding:** The model learns strong moral structure but fails to suppress language/period information. This indicates adversarial training is insufficient and needs strengthening in v10.15.

---

## 1. Corpus Summary

| Language | Passages | Source |
|----------|----------|--------|
| English | 50,000 | Dear Abby, Hendrycks Ethics, Gutenberg |
| Sanskrit | 15,000 | Itihasa |
| Pali | 10,000 | SuttaCentral |
| Hebrew | 7,985 | Sefaria |
| Arabic | 6,235 | Tanzil Quran |
| French | 5,000 | Montaigne, Pascal, Rousseau |
| Classical Chinese | 4,924 | ctext.org (Confucian, Daoist, Mohist) + CBETA |
| Spanish | 4,320 | Don Quixote, La Celestina |
| Greek | 3,157 | Perseus (Aristotle, Plato) |
| Aramaic | 2,015 | Sefaria Talmud |
| Latin | 1,133 | Perseus (Stoics) |
| **TOTAL** | **109,769** | |

---

## 2. Cross-Cultural Transfer Splits

| Experiment | Train | Test | Transfer Type |
|------------|-------|------|---------------|
| hebrew_to_others | 7,985 | 101,784 | Single source → all |
| semitic_to_indic | 16,235 | 25,000 | Middle East → South Asia |
| confucian_to_buddhist | 1,141 | 13,277 | Chinese philosophy → Buddhist |
| ancient_to_modern | 43,884 | 46,628 | Pre-modern → Modern |
| east_to_west | 4,924 | 54,290 | Asian → Western |
| semitic_to_chinese | 16,235 | 4,924 | Abrahamic → Chinese |
| jewish_to_islamic | 7,985 | 6,235 | Judaism → Islam |
| stoic_to_confucian | 7,662 | 1,141 | Greek Stoicism → Confucianism |
| daoist_to_buddhist | 81 | 13,277 | Daoism → Buddhism |
| hindu_to_buddhist | 15,000 | 13,277 | Hinduism → Buddhism |
| mixed_baseline | 76,838 | 32,931 | Mixed → Mixed |

---

## 3. Results

### 3.1 Linear Probe Test (Invariance Check)

**Goal:** If BIP achieves cultural invariance, a linear probe should perform at **chance level** when predicting language or period from the bond embedding.

| Probe | Accuracy | Chance | Status |
|-------|----------|--------|--------|
| Language | **98.6%** | 14.3% | NOT invariant |
| Period | **95.0%** | 12.5% | NOT invariant |

**Interpretation:** The model's z-space still encodes enough information to predict language and period with near-perfect accuracy. This means:
1. Surface features leak into the bond embedding
2. Adversarial training is not strong enough
3. The encoder may be memorizing language-specific patterns

### 3.2 Fuzz Testing (Critical Finding)

**Goal:** Structural moral perturbations should move embeddings MORE than surface perturbations.

| Perturbation | N | Mean Distance | 95% CI |
|--------------|---|---------------|--------|
| **Surface: irrelevant_detail** | 160 | 0.0883 | [0.081, 0.096] |
| **Surface: name_change** | 7 | 0.0335 | [0.018, 0.052] |
| **Surface: synonym** | 12 | 0.0036 | [0.002, 0.005] |
| Structural: obl→perm | 6 | 0.0201 | [0.013, 0.029] |
| Structural: add_harm | 13 | 0.0182 | [0.010, 0.030] |
| Structural: role_swap | 8 | 0.0114 | [0.004, 0.023] |
| Structural: violation→fulfillment | 9 | 0.0143 | [0.009, 0.021] |

**Aggregate:**
- Surface mean: **0.0804**
- Structural mean: **0.0160**
- Ratio: **0.20x** (should be >2.0x)
- Cohen's d: **-1.371** (large effect in WRONG direction)
- Verdict: **NOT_SUPPORTED**

**Critical Interpretation:** The model is doing the OPPOSITE of what we want:
1. Surface perturbations cause LARGE embedding changes
2. Structural moral changes cause SMALL embedding changes
3. The model encodes surface features, not moral structure

### 3.3 Geometric Analysis

| Axis | Metric | Value | Status |
|------|--------|-------|--------|
| Obligation/Permission | Transfer accuracy | 100.0% | **STRONG** |
| Harm/Care | Correlation with Obl/Perm | 0.290 | ORTHOGONAL |
| Role Swap | Mean consistency | 0.681 ± 0.164 | VARIABLE |
| PCA | Components for 90% variance | 6 | HIGH-DIM |

**Interpretation:**
1. **Deontic axis is strong** - The model learns to separate obligation from permission and this transfers perfectly across languages
2. **Harm/Care is orthogonal** - Good separation between moral foundations
3. **Role swap is variable** - Agent/patient transformations not fully consistent
4. **High-dimensional structure** - Moral space is not reducible to 1-2 dimensions

### 3.3 Configuration Used

| Parameter | Value |
|-----------|-------|
| Backbone | LaBSE (sentence-transformers) |
| Batch size | 4096 |
| Learning rate | 3.20e-04 |
| Encoder | UNFROZEN (full fine-tuning) |
| Total params | 471,964,851 |

---

## 4. Diagnosis

### Why is invariance failing?

1. **Adversarial weight too low:** The language/period adversaries are not penalizing surface features strongly enough
2. **Encoder unfreezing:** Full fine-tuning may allow the encoder to memorize language-specific patterns rather than abstract them
3. **Gradient balance:** Main task gradients may dominate adversarial gradients

### Evidence for diagnosis:
- 100% deontic transfer suggests the *moral* signal is learned
- 98.6% language probe suggests the *surface* signal is also preserved
- The model is doing BOTH tasks rather than trading off between them

---

## 5. Recommendations for v10.15.1

Given the **inverted fuzz testing results** (surface > structural), we need aggressive architectural changes, not just hyperparameter tuning.

### 5.1 Root Cause Analysis

The model learns to predict bond types but does NOT learn invariance because:
1. **No explicit contrastive signal** - The model never sees "these two passages have same moral content but different surface"
2. **Adversarial training is too weak** - Language/period adversaries don't cover all surface variation
3. **Training objective mismatch** - Minimizing classification loss doesn't require invariance

### 5.2 Add Contrastive Loss for Surface Invariance

```python
# v10.15.1: Explicit contrastive learning
# Positive pairs: same moral content, different surface (paraphrases, translations)
# Negative pairs: different moral content, same surface (different bond types)

CONTRASTIVE_WEIGHT = 0.3
CONTRASTIVE_TEMPERATURE = 0.07  # InfoNCE temperature
USE_HARD_NEGATIVES = True  # Mine hard negatives within batch
```

### 5.3 Much Stronger Adversarial Training

```python
# v10.14.4 (current)
ADV_MAX_LAMBDA = 0.4
ADV_WARMUP_EPOCHS = 7

# v10.15.1 (proposed)
ADV_MAX_LAMBDA = 1.0  # Maximum strength (was 0.4)
ADV_WARMUP_EPOCHS = 2  # Ramp up faster (was 7)
ADV_HEADS = ["language", "period", "source_corpus"]  # Add corpus adversary
GRL_SCALE_EPOCHS = True  # Scale to 1.5x by end of training
```

### 5.4 Freeze Encoder Completely

```python
# v10.15.1: Do NOT unfreeze encoder
# The encoder (LaBSE) already has good multilingual representations
# Unfreezing allows it to memorize surface patterns
UNFREEZE_ENCODER = False  # Keep frozen throughout
FREEZE_ENCODER = True
```

### 5.5 Add Surface Perturbation Augmentation

```python
# v10.15.1: During training, create surface-perturbed versions
# and enforce embedding similarity
SURFACE_AUGMENT = True
SURFACE_AUGMENT_TYPES = ["synonym", "name_swap", "detail_insert"]
AUGMENT_SIMILARITY_WEIGHT = 0.2
```

### 5.6 Architecture Changes

```python
# v10.15.1: Larger adversarial heads to better detect surface features
ADV_HIDDEN_DIM = 1024  # Was 512
ADV_NUM_LAYERS = 4     # Was 3
ADV_DROPOUT = 0.4      # Was 0.3

# Smaller bond embedding to force compression
Z_DIM = 32  # Was 64 - force more abstraction
```

### 5.7 Training Strategy

```python
# v10.15.1: Two-phase training
PHASE_1_EPOCHS = 10  # Train with frozen encoder, strong adversaries
PHASE_2_EPOCHS = 5   # Fine-tune with contrastive loss only

# Early stopping based on fuzz ratio, not just loss
FUZZ_VALIDATION = True  # Run mini fuzz test every N epochs
TARGET_FUZZ_RATIO = 2.0  # Stop when structural/surface > 2.0
```

### 5.8 Fix Fuzz Testing Path

```python
# v10.15.1: Check Drive location first
checkpoint_patterns = [
    f"{SAVE_DIR}/best_*.pt",  # Drive location
    "models/checkpoints/best_*.pt",
    "*.pt",
]
```

---

## 6. Success Criteria for v10.15.1

| Metric | v10.14.4 | v10.15.1 Target | Notes |
|--------|----------|-----------------|-------|
| **Fuzz ratio (structural/surface)** | 0.20x | **> 2.0x** | PRIMARY metric |
| Language probe | 98.6% | < 30% | Near chance (14.3%) |
| Period probe | 95.0% | < 30% | Near chance (12.5%) |
| Obl/Perm transfer | 100% | > 90% | Allow slight degradation |
| Surface perturbation distance | 0.0804 | < 0.02 | Should be SMALL |
| Structural perturbation distance | 0.0160 | > 0.10 | Should be LARGE |

**Primary Goal:** Invert the fuzz ratio from 0.20x to >2.0x

**Acceptable Trade-offs:**
- Some degradation in classification accuracy is OK if invariance improves
- Longer training time is acceptable
- May need multiple iterations

---

## 7. Next Steps

1. **Create v10.15.1** with increased adversarial training
2. **Run full training** (not SKIP_TRAINING mode)
3. **Verify with fuzz testing** after fixing path issues
4. **Iterate** based on probe results

---

## Appendix: Model Files

All 10 trained models saved to Google Drive:
- `best_hebrew_to_others.pt`
- `best_semitic_to_indic.pt`
- `best_confucian_to_buddhist.pt`
- `best_ancient_to_modern.pt`
- `best_semitic_to_chinese.pt`
- `best_jewish_to_islamic.pt`
- `best_stoic_to_confucian.pt`
- `best_daoist_to_buddhist.pt`
- `best_hindu_to_buddhist.pt`
- `best_mixed_baseline.pt`
