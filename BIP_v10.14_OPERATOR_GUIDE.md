# BIP v10.14.4 Operator & Reviewer Guide

## Executive Summary

**Bond Invariance Probing (BIP) v10.14.4** tests whether ethical/moral reasoning structures are invariant across languages and cultures. **v10.14.4** introduces **encoder unfreezing** - the ability to fine-tune the LaBSE encoder itself, not just probe frozen representations. The hypothesis: if moral cognition is universal, the geometric structure of ethical embeddings should remain stable regardless of the surface language.

**Key Metrics:**
- CKA (Centered Kernel Alignment): Measures geometric similarity between language pairs
- RSA (Representational Similarity Analysis): Correlation of pairwise distances
- Linear Probe Accuracy: Whether moral categories transfer across languages

**Target:** CKA > 0.7, RSA > 0.6 across all language pairs for 6-sigma confidence

---

## Table of Contents

1. [Theoretical Foundation](#1-theoretical-foundation)
2. [Linguistic Background](#2-linguistic-background)
3. [Cognitive Science Basis](#3-cognitive-science-basis)
4. [Prerequisites](#4-prerequisites)
5. [Quick Start](#5-quick-start)
6. [Cell-by-Cell Guide](#6-cell-by-cell-guide)
7. [Data Sources](#7-data-sources)
8. [Configuration Options](#8-configuration-options)
9. [Expected Outputs](#9-expected-outputs)
10. [Quality Metrics](#10-quality-metrics)
11. [Troubleshooting](#11-troubleshooting)
12. [Reviewer Checklist](#12-reviewer-checklist)

---

## 1. Theoretical Foundation

### 1.1 The Bond Invariance Hypothesis

The **Bond Invariance Hypothesis** proposes that moral reasoning operates on abstract structural representations ("bonds") that are:

1. **Language-independent**: The underlying moral concepts exist prior to and independent of linguistic expression
2. **Culturally convergent**: Despite surface variation, core moral structures show cross-cultural stability
3. **Geometrically measurable**: These structures manifest as consistent geometric patterns in embedding space

This builds on three theoretical pillars:

#### Moral Universalism (Ethical Theory)
From Kant's categorical imperative to contemporary moral foundations theory (Haidt & Joseph, 2004), there's substantial philosophical argument that certain moral principles are universal:

- **Harm/Care**: Sensitivity to suffering
- **Fairness/Reciprocity**: Justice and rights
- **Loyalty/Betrayal**: Group obligations
- **Authority/Subversion**: Social hierarchy respect
- **Sanctity/Degradation**: Purity concerns

BIP tests whether these foundations produce invariant neural representations across linguistic encoding.

#### Linguistic Relativity vs. Universalism
The Sapir-Whorf hypothesis suggests language shapes thought. BIP empirically tests the counter-hypothesis for moral cognition: that moral concepts are **pre-linguistic** and thus should show cross-linguistic invariance when properly disentangled from surface form.

#### Representation Learning Theory
Modern NLP has shown that pre-trained language models capture semantic structure. BIP extends this by asking: can we isolate **moral semantic structure** that transfers across languages?

### 1.2 The Adversarial Disentanglement Approach

BIP uses **adversarial training** to separate:

```
Representation = Language-Specific Features + Language-Invariant Moral Structure
```

The **gradient reversal layer** forces the model to:
1. **Maximize** moral classification accuracy (preserve moral information)
2. **Minimize** language classification accuracy (remove language-specific information)

This produces representations where moral structure is preserved but linguistic surface features are discarded.

### 1.3 Geometric Invariance Metrics

**CKA (Centered Kernel Alignment)** measures whether two representation spaces have the same geometric structure:

```
CKA(X, Y) = HSIC(X, Y) / sqrt(HSIC(X, X) * HSIC(Y, Y))
```

Where HSIC is the Hilbert-Schmidt Independence Criterion. High CKA indicates that:
- Similar moral concepts cluster together in both languages
- The relative distances between concepts are preserved
- The overall "shape" of moral space is consistent

**RSA (Representational Similarity Analysis)** correlates pairwise distance matrices:

```
RSA(X, Y) = corr(pdist(X), pdist(Y))
```

This tests whether the **relational structure** (which concepts are similar/different) is preserved across languages.

### 1.4 Statistical Requirements

For 6-sigma confidence (p < 0.0000003):
- Minimum 500 passages per language
- Multiple independent sources per language
- Stratified sampling across time periods
- Cross-validation across train/test splits

---

## 2. Linguistic Background

### 2.1 Cross-Linguistic Moral Semantics

Moral language exhibits both **universals** and **variation**:

#### Universal Features
- **Deontic modality**: All languages express obligation/permission (must, should, may)
- **Evaluative predicates**: Good/bad, right/wrong exist universally
- **Agent/patient marking**: Who acts and who is affected
- **Causation**: Moral responsibility requires causal attribution

#### Language-Specific Features (to be disentangled)
- **Lexicalization patterns**: Which concepts get single words vs. phrases
- **Grammatical encoding**: Tense, aspect, evidentiality marking
- **Honorific systems**: Social hierarchy in language (Japanese, Korean)
- **Metaphorical mappings**: MORAL IS UP, PURITY IS CLEANLINESS

### 2.2 Corpus Linguistic Considerations

BIP v10.14 includes carefully selected corpora spanning:

#### Classical Ethical Texts
- **Sanskrit** (Itihasa): Dharmic ethics, karma, duty
- **Pali** (Sutta Pitaka): Buddhist ethics, the Eightfold Path
- **Classical Chinese** (Analects, Mencius): Confucian virtue ethics, 仁 (ren/benevolence)
- **Biblical Hebrew** (Torah, Prophets): Divine command ethics, covenant
- **Aramaic** (Talmud): Halakhic reasoning, case-based ethics
- **Greek** (Plato, Aristotle): Virtue ethics, eudaimonia
- **Latin** (Cicero, Seneca): Stoic ethics, natural law
- **Arabic** (Quran): Islamic ethics, divine guidance

#### Applied Ethics (Responsa Literature)
The **Responsa** (She'elot u-Teshuvot) represent 2000 years of applied ethical reasoning:

| Period | Dates | Characteristics |
|--------|-------|-----------------|
| Geonim | 600-1000 CE | Post-Talmudic authority |
| Rishonim | 1000-1500 CE | Medieval codification |
| Acharonim | 1500-1800 CE | Early modern casuistry |
| Modern | 1800-present | Contemporary application |

This is analogous to the Dear Abby corpus for English: real ethical dilemmas with reasoned responses.

### 2.3 Typological Diversity

The corpus spans major language families:

| Family | Languages | Features |
|--------|-----------|----------|
| Indo-European | Sanskrit, Greek, Latin, English, Spanish, French, Italian | Inflectional morphology |
| Semitic | Hebrew, Aramaic, Arabic | Root-pattern morphology, VSO/SVO |
| Sino-Tibetan | Classical Chinese | Isolating, topic-prominent |
| Austroasiatic | Pali | Inflectional, SOV tendency |

This diversity tests whether moral invariance holds across radically different linguistic structures.

### 2.4 Diachronic Considerations

Moral concepts evolve. BIP controls for this via `time_period` stratification:

- **Ancient** (pre-500 CE): Foundational texts
- **Medieval** (500-1500 CE): Commentary traditions
- **Early Modern** (1500-1800 CE): Enlightenment ethics
- **Modern** (1800-present): Contemporary application

The hypothesis: despite surface evolution, core moral geometry remains stable.

---

## 3. Cognitive Science Basis

### 3.1 Moral Cognition Research

BIP is grounded in empirical moral psychology:

#### Dual-Process Theory (Kahneman, Greene)
Moral judgment involves:
1. **System 1**: Fast, intuitive, emotional responses
2. **System 2**: Slow, deliberative, rational reasoning

Neural correlates:
- Ventromedial prefrontal cortex (vmPFC): Emotional valuation
- Dorsolateral prefrontal cortex (dlPFC): Rational deliberation
- Anterior cingulate cortex (ACC): Conflict monitoring

BIP hypothesis: Both systems converge on similar **output representations** across languages, even if processing differs.

#### Moral Foundations Theory (Haidt, Graham)
Five (later six) innate moral foundations:

| Foundation | Adaptive Challenge | Characteristic Emotions |
|------------|-------------------|------------------------|
| Care/Harm | Protect offspring | Compassion, distress |
| Fairness/Cheating | Reciprocal altruism | Anger, gratitude |
| Loyalty/Betrayal | Coalition formation | Group pride, rage at traitors |
| Authority/Subversion | Hierarchical coordination | Respect, fear |
| Sanctity/Degradation | Pathogen avoidance | Disgust, elevation |
| Liberty/Oppression | Resist dominance | Reactance, righteous anger |

BIP tests whether these foundations produce consistent geometric signatures across languages.

### 3.2 Language and Thought

#### Strong Linguistic Relativity (Whorf)
Language **determines** thought. Moral concepts would be incommensurable across languages.

#### Weak Linguistic Relativity
Language **influences** thought. Moral concepts show systematic variation tied to linguistic structure.

#### Universalist Position (Pinker, Fodor)
Core concepts exist in **language of thought** (mentalese). Linguistic expression is translation from this universal medium.

**BIP tests the universalist hypothesis for moral cognition.** If CKA > 0.7 across unrelated languages, this supports universal moral representations.

### 3.3 Neural Representation of Moral Concepts

fMRI studies show moral processing activates:

- **Theory of Mind network**: Understanding others' intentions
- **Affective network**: Emotional response to moral content
- **Semantic network**: Conceptual representation

Key finding (Dehghani et al., 2017): Cross-cultural moral concepts show similar neural activation patterns despite linguistic differences.

BIP extends this to computational representations: do transformer models exhibit similar cross-linguistic invariance?

### 3.4 The Geometry of Concepts

Conceptual spaces (Gärdenfors, 2000) represent concepts as regions in geometric space:

- **Similarity** = proximity
- **Categories** = convex regions
- **Properties** = dimensions

For moral concepts:
- VIRTUE and VICE should be distant
- Related virtues (courage, justice) should cluster
- Cross-linguistic translations should occupy similar regions

BIP measures this directly via CKA and RSA.

### 3.5 Predictions

| Prediction | If TRUE | If FALSE |
|------------|---------|----------|
| High CKA across languages | Moral universalism supported | Linguistic relativity for ethics |
| RSA correlation > 0.6 | Relational structure preserved | Culture-specific moral relations |
| Linear probe transfer > 60% | Categories transfer cross-linguistically | Language-specific moral categories |
| Surface robustness > 90% | Moral processing ignores form | Overfitting to linguistic surface |
| Structural sensitivity > 70% | Model captures moral content | Model captures superficial patterns |

---

## 4. Prerequisites

### Hardware Requirements

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| GPU | T4 (16GB) | L4 (24GB) or A100 |
| RAM | 32GB | 53GB+ |
| Disk | 50GB free | 100GB free |
| Runtime | Colab Pro | Colab Pro+ / Local |

### Software Requirements

- Python 3.10+
- PyTorch 2.0+
- Transformers 4.30+
- datasets (GitHub, HuggingFace, SuttaCentral, Tanzil,net, Sefaria, ctext.org, )
- requests, pandas, numpy, scipy

### API Access (Optional but Recommended)

| Service | Required For | How to Get |
|---------|--------------|------------|
| Kaggle API | Sanskrit/Arabic datasets | kaggle.com -> Account -> Create API Token |
| HuggingFace | hendrycks/ethics, folk tales | huggingface.co -> Settings -> Access Tokens |

---

## 5. Quick Start

### Colab Setup

1. Open `BIP_v10_14.ipynb` in Google Colab
2. Runtime -> Change runtime type -> **L4 GPU** (or T4 minimum)
3. Run Cell 1 (Configuration)
4. Run Cell 2 (Corpus Loading) - **Takes 10-30 minutes first run**
5. Run remaining cells sequentially

### Local Setup

```bash
# Clone and setup
git clone <repo>
cd sqnd-probe

# Install dependencies
pip install torch transformers datasets requests pandas scipy scikit-learn

# Optional: Configure Kaggle
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# Run notebook
jupyter notebook BIP_v10_14.ipynb
```

---

## 6. Cell-by-Cell Guide

### Pipeline Data Flow Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        BIP v10.12 DATA PIPELINE                          │
└─────────────────────────────────────────────────────────────────────────┘

Cell 1: Configuration
    │
    ├── SAVE_DIR, CACHE_ONLY, RANDOM_SEED
    │
    ▼
Cell 2: Load Corpora ───────────────────────────────────────────┐
    │                                                           │
    │  15 External Sources:                                     │
    │  • Itihasa (Sanskrit)      • SuttaCentral (Pali)         │
    │  • Tanzil (Arabic)         • Sefaria (Hebrew/Aramaic)    │
    │  • ctext.org (Chinese)     • Perseus (Greek/Latin)       │
    │  • Gutenberg (Philosophy)  • Gutenberg (Romance)         │
    │  • HuggingFace (Folklore)  • Mac-STAT (Dear Abby)        │
    │  • GitHub (hendrycks/ethics)                              │
    │                                                           │
    └──► passages.jsonl (~115K passages)                        │
              │                                                  │
              ▼                                                  │
Cell 3: Load Ethics Datasets                                    │
    │                                                           │
    │  • ETHICS dataset (HuggingFace)                          │
    │  • Scruples dataset                                       │
    │  • Additional moral reasoning examples                   │
    │                                                           │
    └──► passages.jsonl (extended)                              │
              │                                                  │
              ▼                                                  │
Cell 4: Patterns + Bond Extraction                              │
    │                                                           │
    │  • Define 320+ moral patterns (7 languages)              │
    │  • Extract bond type, Hohfeld state, negation            │
    │  • Classify prescriptive vs descriptive                  │
    │                                                           │
    └──► bonds.jsonl (~115K bonds)                              │
              │                                                  │
              ├───────────────────────────────────────┐         │
              ▼                                       ▼         │
Cell 5: Generate Splits                         Cell 6: Model   │
    │                                           Architecture    │
    │  • 11 cross-lingual experiments           │               │
    │  • Stratified by language/period          │               │
    │                                           │               │
    └──► all_splits.json                        └──► BIPModel   │
              │                                       │         │
              └───────────────┬───────────────────────┘         │
                              ▼                                  │
                       Cell 7: Training                         │
                              │                                  │
                              │  For each split:                │
                              │  • Load train/test passages     │
                              │  • Train with adversarial loss  │
                              │  • Save checkpoints             │
                              │                                  │
                              └──► model checkpoints             │
                                        │                        │
              ┌─────────────────────────┼─────────────────────┐ │
              ▼                         ▼                     ▼ │
       Cell 8: Analysis          Cell 9: Fuzz          Cell 10: │
       • CKA/RSA metrics         Testing               Save     │
       • Linear probes           • Adversarial         Results  │
       • Transfer tests          • Robustness                   │
                                                                │
                              ▼                                  │
                        Final Results                            │
                        • CKA matrix                             │
                        • RSA scores                             │
                        • Transfer accuracy                      │
└───────────────────────────────────────────────────────────────┘
```

### Critical Files

| File | Created By | Used By | Description |
|------|-----------|---------|-------------|
| `passages.jsonl` | Cells 2, 3 | Cells 4, 5, 6, 7 | Raw text passages |
| `bonds.jsonl` | Cell 4 | Cells 6, 7, 8 | Bond extraction results |
| `all_splits.json` | Cell 5 | Cell 7 | Train/test configurations |
| `checkpoints/*.pt` | Cell 7 | Cells 8, 9 | Trained model weights |

---

### Cell 1: Configuration & Setup

**Purpose:** Set global parameters and mount Google Drive for persistence.

**Key Settings:**

```python
SAVE_DIR = "/content/drive/MyDrive/BIP_v10.12"  # Persistence location
CACHE_ONLY = False      # True = skip downloads, use cached data only
CREATE_DOWNLOAD_ZIP = True  # Create downloadable results package
```

**Expected Output:**
```
Google Drive mounted at /content/drive
Save directory: /content/drive/MyDrive/BIP_v10.12
GPU: NVIDIA L4 (22.5GB)
```

**Reviewer Notes:**
- Verify GPU is detected (CUDA available)
- Check available disk space (should show 50GB+ free)
- Drive mount may require authorization popup

---

### Cell 2: Load Corpora (Self-Contained)

**Purpose:** Download and parse all verified data sources.

**Duration:** 10-30 minutes (first run), <1 minute (cached)

**v10.12 Data Sources:**

| Language | Source | Expected Count | Notes |
|----------|--------|----------------|-------|
| Sanskrit | Itihasa GitHub | ~93,000 | Mahabharata, Ramayana |
| Pali | SuttaCentral API | ~5,000-10,000 | Theravada Canon |
| Arabic | Tanzil.net | 6,236 | Quran (Uthmani) |
| Hebrew | Sefaria GitHub | ~20,000 | Complete Tanakh (39 books) |
| Hebrew | Sefaria (Mishnah) | ~8,000 | 25 tractates from all 6 orders |
| Hebrew | Sefaria (Responsa) | ~10,000+ | 2000 years of ethical Q&A (if INCLUDE_RESPONSA=True) |
| Aramaic | Sefaria GitHub | ~10,000 | 17 Talmud Bavli tractates |
| Chinese | ctext.org API | ~3,000-5,000 | Analects, Mencius, Daodejing |
| Greek | Perseus GitHub | ~3,000-5,000 | Plato, Aristotle, Stoics |
| Latin | Perseus GitHub | ~3,000-5,000 | Cicero, Seneca |
| Spanish | GITenberg (JIT) | ~3,000 | Don Quijote, La Celestina, Lazarillo |
| French | GITenberg (JIT) | ~4,000 | Montaigne, Voltaire, Rousseau, Descartes, Hugo |
| Italian | GITenberg (JIT) | ~2,500 | Machiavelli, Dante, Boccaccio, Ariosto |
| Portuguese | GITenberg (JIT) | ~500 | Camões |
| Romanian | GITenberg (JIT) | ~200 | Eminescu |
| English (Philosophy) | GITenberg (JIT) | ~5,000 | Kant, Mill, Spinoza, Aristotle, Plato |
| English (Folklore) | HuggingFace | ~50,000 | World folklore incl. Native American |
| English (Dear Abby) | Mac-STAT GitHub | ~68,000 | Applied ethics Q&A |
| English (Ethics) | hendrycks/ethics | ~134,000 | Ethics scenarios |

**New in v10.12:**
- **`gutenberg_download(id)`**: Simple download by ID like R's gutenbergr - just list IDs, no repo names
- **JIT (Just-In-Time) loading**: Fetches texts one at a time, stops at target passage count
- **Per-text caching**: Individual text caching in `cache/gutenberg_texts/` and `cache/romance_texts/`
- **Western Philosophy**: 11 texts by ID (Kant, Mill, Spinoza, Aristotle, Plato, Stoics)
- **Romance Languages**: 12 texts by ID (Spanish, French, Italian, Portuguese)
- **Responsa literature**: Geonim, Rishonim, Acharonim, Modern periods
- **Expanded Mishnah**: 5 tractates (Pirkei Avot, Sanhedrin, Bava Kamma, Bava Metzia, Bava Batra)
- **Progress display**: Git clone shows percentage progress
- **Stall detection**: Downloads killed if no progress for 120s (instead of fixed timeout)

**Expected Output:**
```
============================================================
CORPUS SUMMARY
============================================================
  english             : 150,000+ passages  [OK]
  sanskrit            :  15,000 passages  [OK]
  hebrew              :  15,000 passages  [OK]
  arabic              :  10,000 passages  [OK]
  pali                :  10,000 passages  [OK]
  aramaic             :  10,000 passages  [OK]
  classical_chinese   :   5,000 passages  [OK]
  greek               :   5,000 passages  [OK]
  latin               :   5,000 passages  [OK]
  spanish             :   3,000 passages  [OK]
  french              :   4,000 passages  [OK]
  italian             :   2,500 passages  [OK]
  portuguese          :     500 passages  [OK]
  romanian            :     200 passages  [OK]
------------------------------------------------------------
  TOTAL               : 300,000+ passages
============================================================
```

**JIT Loading Output (first run):**
```
[WESTERN PHILOSOPHY]
  Fetching from GITenberg mirrors (JIT)...
    Kant Critique of Practical Reason: 487 passages
    Kant Metaphysical Elements of Ethics: 156 passages
    Kant Critique of Pure Reason: 1,234 passages
    Mill Utilitarianism: 289 passages
    Mill On Liberty: 567 passages
    (reached 5,000 target, stopping)
  Gutenberg: 5,123 passages

[ROMANCE LANGUAGES]
  Fetching Romance philosophy (JIT)...
    Don Quixote: 2,847 passages
    La Celestina: 412 passages
    Lazarillo de Tormes: 189 passages
    Candide (Voltaire): 312 passages
    Social Contract (Rousseau): 567 passages
    Essais de Montaigne: 3,456 passages
    ...
    (reached 10,000 target, stopping)
  Romance Philosophy: 10,234 passages
```

**Sefaria Clone Progress:**
```
Cloning Sefaria-Export from GitHub (~2GB, timeout 10min)...
Progress: 1% 3% 5% 8% 12% 15% 20% 25% 30% 35% 40% 45% 50% ...
    Receiving objects: 75% (45000/60000)
60% 65% 70% 75% 80% 85% 90% 95% 100%
    Clone successful!
```

**Reviewer Notes:**
- Each language should show `[OK]` (500+ passages minimum)
- Responsa should show individual collection counts
- Cached data stored in `data/raw/v10.12/cache/*.json`
- Final output: `data/processed/passages.jsonl`

---

### Cell 3: Load Ethics Datasets

**Purpose:** Load additional ethics datasets for bond extraction training.

**Inputs:**
- `data/processed/passages.jsonl` (from Cells 2, 3)

**Outputs:**
- Extended `data/processed/passages.jsonl` with additional ethics examples

**What It Does:**
1. Loads ETHICS dataset from HuggingFace
2. Loads Scruples dataset
3. Adds modern English moral reasoning examples
4. Extends passages.jsonl with new examples

---

### Cell 4: Patterns + Normalization + Bond Extraction

**Purpose:** Define language-specific patterns and extract moral bonds from all passages.

**Inputs:**
- `data/processed/passages.jsonl` (from Cells 2, 3)

**Outputs:**
- `data/processed/bonds.jsonl` (bond extraction results)

**Key Operations:**
1. **Pattern Definition:** 320+ regex patterns across 7 core languages
2. **Normalization Functions:** Unicode normalization for Hebrew, Arabic, Sanskrit, Pali
3. **Bond Extraction:** Classify each passage into moral categories
4. **Hohfeld Analysis:** Map moral statements to legal positions (obligation/liberty/right/no-right)
5. **Context Detection:** Classify as prescriptive vs descriptive

**Bond Types (8 categories):**
| Type | Description | Example |
|------|-------------|---------|
| DUTY | Obligations | "You must honor your parents" |
| CARE | Compassion | "Help those in need" |
| HARM | Violence/damage | "Do not kill" |
| FAIRNESS | Justice | "Treat others equally" |
| AUTHORITY | Power/hierarchy | "Obey the king" |
| LOYALTY | Group bonds | "Stand by your family" |
| SANCTITY | Purity | "Keep the sabbath holy" |
| NEUTRAL | No moral content | Descriptive text |

**bonds.jsonl Schema:**
```json
{
  "passage_id": "sefaria_1234",
  "bond_type": "DUTY",
  "hohfeld_state": "OBLIGATION",
  "negated": false,
  "modal": "must",
  "confidence": 0.85,
  "context": "prescriptive"
}
```

**Expected Output:**
```
EXTRACTING BONDS FROM PASSAGES
============================================================
  Processing 115,000 passages...
    20,000 processed...
    40,000 processed...
    ...
  Saved 115,000 bonds to data/processed/bonds.jsonl

  Bond type distribution:
    NEUTRAL     : 45,000 (39.1%)
    DUTY        : 22,000 (19.1%)
    CARE        : 15,000 (13.0%)
    HARM        :  8,000 (7.0%)
    FAIRNESS    :  7,000 (6.1%)
    AUTHORITY   :  6,000 (5.2%)
    LOYALTY     :  6,000 (5.2%)
    SANCTITY    :  6,000 (5.2%)
```

---

### Cell 5: Generate Splits

**Purpose:** Create train/validation/test splits with stratification.

**Split Strategy:**
- 80% train, 10% validation, 10% test
- Stratified by language AND time_period
- No source leakage between splits

---

### Cell 6: Model Architecture

**Purpose:** Define the BIP neural network architecture.

**Architecture:**
```
Input: mBERT embeddings (768-dim)
  -> Language Adversary (gradient reversal)
  -> Shared Moral Encoder (512-dim)
  -> Task Heads:
     - Moral Category Classifier
     - Time Period Classifier
     - Cross-lingual Alignment Head
```

---

### Cell 7: Training Loop

**Purpose:** Train the model with adversarial language disentanglement.

**v10.14.4 Encoder Unfreezing:**

When `UNFREEZE_ENCODER=True`, the training loop implements a staged approach:

1. **Phase 1 (Warmup):** Encoder frozen, only probe heads train (epochs 1 to `UNFREEZE_AFTER_EPOCHS-1`)
2. **Phase 2 (Unfreeze):** At epoch `UNFREEZE_AFTER_EPOCHS`:
   - Encoder parameters unfrozen
   - Fresh optimizer created at `ENCODER_LR/100`
   - AMP scaler reset to prevent stale state
   - Batch size re-probed with backward pass (typically reduces from ~512 to ~32-64)
3. **Phase 3 (Warmup LR):** Over 5 epochs, encoder LR warms from 1% to 100% of target

**WiFi-Style Batch Probing:**

The `probe_max_batch()` function uses binary search to find the maximum batch size that fits in GPU memory:
- Starts at low=8, high=target_batch
- Tests each batch size with forward (or forward+backward if encoder trainable)
- Halves range on OOM, doubles on success
- Returns safe batch size with 10% headroom

**Key Metrics During Training:**
- `loss_moral`: Moral classification loss (should decrease)
- `loss_adv`: Adversarial language loss (should INCREASE - disentanglement working)
- `acc_moral`: Moral accuracy (target: >70%)
- `acc_lang`: Language accuracy (target: <20% - random chance)

**Expected Output (with encoder unfreezing):**
```
Encoder mode: UNFROZEN after epoch 2
  Probing max batch size (trainable=False)... 512
  Encoder FROZEN (probe-only mode)
  Trainable: 12,345,678 / 471,234,567 (2.6%)

Epoch 1: Loss=2.69 | Moral=45% | Lang=85%
Epoch 2: Loss=1.82 | Moral=58% | Lang=62%

  >>> UNFREEZING ENCODER at epoch 2 <<<
  Trainable params now: 471,234,567
  Re-probing batch size with encoder trainable...
  [v10.14.4] Encoder trainable - probing with backward pass
  Probing max batch size (trainable=True)... 32
  New batch size: 32 (was 512)
  Encoder LR warmup: 5e-9 -> 5e-7 over 5 epochs

Epoch 2 (cont): Loss=1.75 | Moral=61% | Lang=55%
Epoch 3: Loss=1.42 | Moral=68% | Lang=42%
...
```

---

### Cell 8: Geometric Analysis & Linear Probe

**Purpose:** Measure cross-lingual invariance with CKA and RSA.

---

### Cell 9: Fuzz Testing

**Purpose:** Adversarial robustness testing with structural vs. surface perturbations.

---

### Cell 10: Save & Download Results

**Purpose:** Persist all results and create downloadable package.

---

## 7. Data Sources

### Verified Sources (All URLs tested 2026-01-15)

| Source | URL | License | Content |
|--------|-----|---------|---------|
| Itihasa | github.com/rahular/itihasa | MIT | Sanskrit epics (93K shlokas) |
| SuttaCentral | suttacentral.net/api | CC0 | Pali Canon |
| Tanzil | tanzil.net | Open | Quran (Uthmani script) |
| Sefaria | github.com/Sefaria/Sefaria-Export | CC-BY-NC | Hebrew/Aramaic (~2GB) |
| ctext.org | api.ctext.org | Academic | Chinese classics |
| Perseus | github.com/PerseusDL | CC-BY-SA | Greek/Latin philosophy |
| **Gutenberg** | gutenberg.org | Public Domain | Western/Romance philosophy |
| Mac-STAT | github.com/Mac-STAT/data | CC | Dear Abby (68K letters) |
| HuggingFace | merve/folk-mythology-tales | CC | World folklore |
| HuggingFace | hendrycks/ethics | MIT | Ethics scenarios (134K) |

### Project Gutenberg Downloads (v10.12b)

BIP uses `gutenberg_download(id)` inspired by R's [gutenbergr](https://docs.ropensci.org/gutenbergr/) package - just give it a list of Gutenberg IDs:

```python
# Just maintain a list of IDs - no repo names needed
texts = [
    (5683, "Kant Critique of Practical Reason", "MODERN_ETHICS"),
    (11224, "Mill Utilitarianism", "MODERN_ETHICS"),
    (3800, "Spinoza Ethics", "MODERN_ETHICS"),
    ...
]

def gutenberg_download(gutenberg_id: int) -> str | None:
    """Download text from Project Gutenberg by ID."""
    url = f"https://www.gutenberg.org/ebooks/{gutenberg_id}.txt.utf-8"
    ...
```

**URL pattern:** `https://www.gutenberg.org/ebooks/{id}.txt.utf-8`

**Western Philosophy & Religion (16 texts by ID):**
| ID | Author | Work |
|----|--------|------|
| 5683 | Kant | Critique of Practical Reason |
| 5684 | Kant | Metaphysical Elements of Ethics |
| 4280 | Kant | Critique of Pure Reason |
| 11224 | Mill | Utilitarianism |
| 34901 | Mill | On Liberty |
| 3800 | Spinoza | Ethics |
| 8438 | Aristotle | Nicomachean Ethics |
| 1497 | Plato | Republic |
| 1656 | Plato | Apology |
| 10661 | Epictetus | Discourses |
| 2680 | Marcus Aurelius | Meditations |
| 10 | Bible | KJV Complete (80 books) |
| 124 | Apocrypha | Deuterocanonical Books |
| 1670 | Luther | Small Catechism |
| 1722 | Luther | Large Catechism |
| 43855 | Franklin | Poor Richard's Almanack |

**Romance Languages (12 texts by ID):**
| ID | Language | Work |
|----|----------|------|
| 996 | Spanish | Don Quixote |
| 1619 | Spanish | La Celestina |
| 320 | Spanish | Lazarillo de Tormes |
| 19942 | French | Candide (Voltaire) |
| 46333 | French | Social Contract (Rousseau) |
| 3600 | French | Essais de Montaigne |
| 18269 | French | Pensées (Pascal) |
| 1232 | Italian | The Prince (Machiavelli) |
| 1004 | Italian | Divine Comedy (Dante) |
| 3726 | Italian | Decameron Vol I (Boccaccio) |
| 13102 | Italian | Decameron Vol II (Boccaccio) |
| 3333 | Portuguese | Os Lusíadas (Camões) |

### Responsa Collections (NEW in v10.12)

| Collection | Period | Content |
|------------|--------|---------|
| Geonim | 600-1000 CE | Post-Talmudic responsa |
| Rishonim | 1000-1500 CE | Medieval authorities |
| Acharonim | 1500-1800 CE | Early modern casuistry |
| Modern | 1800-present | Contemporary responsa |
| Teshuvot Maharsham | 19th century | Major halakhic collection |

---

## 8. Configuration Options

### Cell 1 Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `SAVE_DIR` | Drive path | Where to persist results |
| `CACHE_ONLY` | False | Skip downloads, use cache only |
| `CREATE_DOWNLOAD_ZIP` | True | Create downloadable zip |
| `RANDOM_SEED` | 42 | Reproducibility seed |
| `INCLUDE_RESPONSA` | False | Include Responsa texts (requires 30-50 min git clone) |

#### v10.14.4 Encoder Unfreezing Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `UNFREEZE_ENCODER` | False | Enable encoder fine-tuning (not just probe heads) |
| `UNFREEZE_AFTER_EPOCHS` | 2 | Epochs to train probe heads before unfreezing encoder |
| `ENCODER_LR` | 5e-7 | Learning rate for encoder (very low to prevent catastrophic forgetting) |
| `HEAD_LR` | 1e-3 | Learning rate for probe heads |
| `GRADIENT_ACCUMULATION_STEPS` | 4 | Accumulate gradients for larger effective batch |

**Warning:** Encoder unfreezing requires careful hyperparameter tuning. The encoder (LaBSE, 471M params) can destabilize if the learning rate is too high. Start with defaults.

### Sefaria Download Strategy (v10.14)

The `INCLUDE_RESPONSA` setting controls how Sefaria texts are downloaded:

| Setting | Download Method | Time | Texts |
|---------|-----------------|------|-------|
| `False` (default) | Staged download only | ~5 min | 88 texts (Tanakh, Mishnah, Talmud) |
| `True` | Full git clone | 30-50 min | Everything including Responsa |

**Recommendation:** Start with `INCLUDE_RESPONSA=False` for faster iteration. Enable for full corpus runs.

### Drive Cache Behavior (v10.12b)

When `SAVE_DIR` points to Google Drive, the notebook implements **persistent caching**:

**First Run:**
1. Downloads all corpus data from external sources
2. Saves processed passages to `{SAVE_DIR}/passages.jsonl`
3. Saves individual cache files to `{SAVE_DIR}/corpus_cache/*.json`
4. Saves Sefaria-Export to `{SAVE_DIR}/Sefaria-Export/` (one-time, ~2GB)

**Subsequent Runs:**
1. Checks `{SAVE_DIR}/corpus_cache/` for cached JSON files
2. Restores any missing files to local cache
3. Checks `{SAVE_DIR}/Sefaria-Export/` and restores if present
4. Each loader finds its cache → prints "(cached)" instead of downloading

**Result:** First run takes 10-15 minutes; subsequent runs take ~30 seconds.

**Cache Files Saved:**
| File | Size | Contents |
|------|------|----------|
| `itihasa.json` | ~15MB | 93K Sanskrit shlokas |
| `suttacentral.json` | ~8MB | 42K Pali passages |
| `tanzil.json` | ~2MB | 6K Quran verses |
| `sefaria.json` | ~5MB | 15K Hebrew/Aramaic passages |
| `ctext.json` | ~3MB | 2K Chinese passages |
| `perseus.json` | ~4MB | Greek/Latin passages |
| `gutenberg_texts/*.json` | ~2MB | Western philosophy |
| `romance_texts/*.json` | ~3MB | Romance languages |
| `folk_mythology.json` | ~20MB | 50K folk tales |
| `dear_abby.json` | ~30MB | 68K advice letters |
| `hendrycks_ethics.json` | ~40MB | 134K ethics scenarios |
| `Sefaria-Export/` | ~2GB | Full Sefaria JSON (for Responsa) |

### Training Hyperparameters (Cell 7)

| Parameter | Default | Range | Notes |
|-----------|---------|-------|-------|
| `EPOCHS` | 5 | 3-10 | More epochs = better convergence |
| `BATCH_SIZE` | 32 | 16-64 | Reduce if OOM |
| `LEARNING_RATE` | 2e-5 | 1e-5 to 5e-5 | Standard for BERT fine-tuning |
| `ADV_WEIGHT` | 0.3 | 0.1-0.5 | Higher = more disentanglement |
| `WARMUP_STEPS` | 500 | 100-1000 | LR warmup period |

### JIT Loading Configuration (Cell 2)

JIT (Just-In-Time) loading fetches texts one at a time using `gutenberg_download(id)`. Benefits:
- Faster startup (stops when target reached)
- Less network usage
- Per-text caching (resume after failures)
- Simple to add new texts (just add Gutenberg ID)

| Source | Target Passages | Cache Location |
|--------|-----------------|----------------|
| Gutenberg (English) | 5,000 | `cache/gutenberg_texts/{id}.json` |
| Romance Languages | 10,000 | `cache/romance_texts/{id}.json` |

**Adding new texts:** Just add the Gutenberg ID to the list:
```python
texts = [
    (5683, "Kant Critique of Practical Reason", "MODERN_ETHICS"),
    (12345, "New Text Title", "PERIOD"),  # <- Just add ID
    ...
]
```

To fetch all texts (no early stop), call with `target_passages=0`:
```python
passages = load_gutenberg_philosophy(target_passages=0)  # Fetch all 11 texts
passages = load_romance_philosophy(target_passages=0)    # Fetch all 12 texts
```

### Memory Limits (Cell 2)

| Language | Max Passages | Rationale |
|----------|--------------|-----------|
| english | 50,000 | Multiple large sources |
| sanskrit | 15,000 | Itihasa is huge (93K) |
| hebrew | 15,000 | Sefaria + Responsa |
| aramaic | 10,000 | Talmud |
| Others | 5,000-10,000 | Balance corpus size |

---

## 9. Expected Outputs

### Success Criteria

| Metric | Threshold | Interpretation |
|--------|-----------|----------------|
| CKA (mean) | > 0.70 | Strong geometric invariance |
| CKA (min) | > 0.50 | No catastrophic failures |
| RSA (mean) | > 0.60 | Consistent distance structure |
| Linear Probe (cross-lingual) | > 60% | Transferable representations |
| Surface Robustness | > 90% | Not overfitting to form |
| Structural Sensitivity | > 70% | Capturing moral content |
| Language Adversary Acc | < 20% | Successful disentanglement |

---

## 10. Quality Metrics

### Corpus Quality

```python
# Minimum requirements for 6-sigma confidence
MIN_PASSAGES_PER_LANG = 500
MIN_UNIQUE_SOURCES = 2
MAX_SINGLE_SOURCE_RATIO = 0.8  # No source >80% of language
```

### Invariance Quality

| Indicator | Good | Investigate |
|-----------|------|-------------|
| CKA variance | < 0.1 | > 0.2 (inconsistent) |
| Language outliers | None | Any CKA < 0.4 |
| Probe transfer gap | < 15% | > 25% (poor transfer) |

---

## 11. Troubleshooting

### Common Issues

#### Out of Memory (OOM)
**Solutions:**
1. Reduce `BATCH_SIZE` (32 -> 16 -> 8)
2. Reduce `MAX_PASSAGES_PER_LANG`
3. Use gradient checkpointing

#### Sefaria Clone Timeout/Stall
**Solutions:**
1. Retry (stall detection kills after 120s with no progress)
2. Manual clone: `git clone --depth 1 https://github.com/Sefaria/Sefaria-Export.git`
3. Upload to `{SAVE_DIR}/manual_uploads/Sefaria-Export/`

#### Gutenberg Downloads Failing
**Problem:** Project Gutenberg may block cloud provider IPs (including Google Colab/GCP).

**Solution (v10.12b):** Uses direct download by ID like R's gutenbergr:
```python
url = f"https://www.gutenberg.org/ebooks/{id}.txt.utf-8"
```

This URL pattern works from most locations. If blocked:
1. Check your network connection
2. Delete cached files: `rm -rf data/raw/v10.12/cache/gutenberg_texts/`
3. Clear legacy cache: `rm data/raw/v10.12/cache/gutenberg.json`
4. Try running from a different network

**Adding new texts:** Just find the Gutenberg ID and add it to the list:
- Go to gutenberg.org, find the book
- Copy the ID from the URL (e.g., `gutenberg.org/ebooks/5683` → ID is 5683)
- Add to the texts list: `(5683, "Title", "PERIOD")`

#### Romance Languages Showing 0 Passages (FIXED in v10.12b)
**Previous issue:** 500 paragraph cap per text + complex GITenberg repo lookup
**Fixed in v10.12b:**
- Removed arbitrary 500 cap
- Simple `gutenberg_download(id)` by ID
- JIT loading with individual caching

#### NaN Losses After Encoder Unfreeze (v10.14.4)
**Symptoms:** Training proceeds normally, then immediately after ">>> UNFREEZING ENCODER <<<" you see repeated "NaN loss detected - skipping batch" messages.

**Causes & Solutions:**

| Cause | Solution |
|-------|----------|
| Encoder LR too high | Reduce `ENCODER_LR` (try 1e-7 or 5e-8) |
| Stale optimizer momentum | v10.14.4 creates fresh optimizer on unfreeze (automatic) |
| AMP scaler state mismatch | v10.14.4 resets scaler on unfreeze (automatic) |
| Gradient explosion | Gradient clipping enabled by default (max_norm=1.0) |
| Batch too large for backward | Re-probe runs automatically; if still failing, reduce `BATCH_SIZE` |

**Debug Steps:**
1. Check if NaN appears immediately after unfreeze → LR too high
2. Check if NaN appears after a few batches → gradient explosion, reduce LR
3. Check if OOM appears → batch size too large for backward pass

**Safe Configuration:**
```python
UNFREEZE_ENCODER = True
ENCODER_LR = 5e-7  # Very conservative
HEAD_LR = 1e-3
UNFREEZE_AFTER_EPOCHS = 2
GRADIENT_ACCUMULATION_STEPS = 4
```

#### Low CKA Scores
**Possible Causes:**
1. Insufficient training (increase epochs)
2. Adversarial weight too low (increase λ)
3. Corpus imbalance (check per-language counts)

---

## 12. Reviewer Checklist

### Pre-Run
- [ ] GPU runtime selected (L4 recommended)
- [ ] Google Drive mounted
- [ ] Sufficient disk space (50GB+)

### Cell 2 (Corpus)
- [ ] All languages show `[OK]` status
- [ ] Total passages > 200,000
- [ ] Responsa collections loaded (check for "Geonim", "Rishonim" etc.)
- [ ] Gutenberg downloads show passages (not 0 or "download failed")
- [ ] Western Philosophy: ~5,000 passages (11 texts by ID)
- [ ] Romance Languages: ~10,000 passages (12 texts by ID)
- [ ] JIT loading shows "(reached X target, stopping)" or all texts fetched
- [ ] Cache files created in `cache/gutenberg_texts/` and `cache/romance_texts/`

### Cell 7 (Training)
- [ ] Loss decreasing over epochs
- [ ] Language accuracy DECREASING (disentanglement working)
- [ ] Moral accuracy INCREASING
- [ ] (If UNFREEZE_ENCODER=True) Encoder unfreezes at correct epoch
- [ ] (If UNFREEZE_ENCODER=True) Batch size re-probed after unfreeze
- [ ] (If UNFREEZE_ENCODER=True) No NaN losses after unfreeze
- [ ] (If UNFREEZE_ENCODER=True) LR warmup message appears

### Cell 8 (Analysis)
- [ ] CKA mean > 0.70
- [ ] RSA mean > 0.60
- [ ] Linear probe transfer > 60%

### Post-Run
- [ ] `final_results.json` saved
- [ ] Model checkpoints saved
- [ ] Zip package downloadable

---

## Appendix A: CKA Interpretation

| CKA Range | Interpretation |
|-----------|----------------|
| 0.9 - 1.0 | Near-identical geometry |
| 0.7 - 0.9 | Strong invariance (target) |
| 0.5 - 0.7 | Moderate invariance |
| 0.3 - 0.5 | Weak invariance |
| < 0.3 | Poor invariance |

---

## Appendix B: Version History

| Version | Date | Changes |
|---------|------|---------|
| v10.14.4 | 2026-01-18 | **Encoder unfreezing**: Fine-tune LaBSE encoder (471M params) after warmup epochs; differential learning rates (encoder 5e-7, heads 1e-3); LR warmup from 1% to 100% over 5 epochs after unfreeze; fresh optimizer on unfreeze to avoid stale momentum; AMP scaler reset; WiFi-style batch re-probing with backward pass; NaN detection and handling; gradient accumulation (4 steps default) |
| v10.14.3 | 2026-01-17 | Adversarial training improvements, ADV_HIDDEN_DIM=512, ADV_NUM_LAYERS=3, ADV_DROPOUT=0.3 |
| v10.14 | 2026-01-16 | Bible KJV (80 books), Apocrypha, Luther's Catechisms, Poor Richard's Almanack; Sefaria expanded to 88 texts (39 Tanakh, 26 Mishnah, 17 Talmud); INCLUDE_RESPONSA config (default False); elapsed time tracking for all loaders |
| v10.13 | 2026-01-16 | Configurable experiment matrix, bond extraction pipeline |
| v10.12b | 2026-01-16 | Simplified to `gutenberg_download(id)` like R's gutenbergr - just list IDs, no repo names |
| v10.12a | 2026-01-15 | Responsa literature, expanded Mishnah, fixed loaders, progress display, stall detection |
| v10.11 | 2026-01-14 | Expanded corpus, role augmentation, memory fixes |
| v10.10 | 2026-01-13 | Initial expanded corpus, adversarial training |

---

## Appendix C: Key References

### Moral Psychology
- Haidt, J. (2012). *The Righteous Mind*
- Greene, J. (2013). *Moral Tribes*
- Haidt, J. & Joseph, C. (2004). Intuitive ethics. *Daedalus*

### Linguistics
- Wierzbicka, A. (1996). *Semantics: Primes and Universals*
- Levinson, S. C. (2003). *Space in Language and Cognition*
- Evans, N. & Levinson, S. C. (2009). The myth of language universals. *BBS*

### Cognitive Science
- Kahneman, D. (2011). *Thinking, Fast and Slow*
- Gärdenfors, P. (2000). *Conceptual Spaces*
- Fodor, J. (1975). *The Language of Thought*

### Representation Learning
- Kornblith, S. et al. (2019). Similarity of neural network representations. *ICML*
- Kriegeskorte, N. et al. (2008). Representational similarity analysis. *Frontiers*

---

*Document generated for BIP v10.14.4 - Last updated: 2026-01-18*
