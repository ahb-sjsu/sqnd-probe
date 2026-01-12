# BIP v10.2 Technical Guide

## Bond Invariance Principle: Native-Language Moral Pattern Transfer

**Version:** 10.2
**Last Updated:** January 2025

---

## Table of Contents

1. [Overview](#1-overview)
2. [Theoretical Foundation](#2-theoretical-foundation)
3. [System Requirements](#3-system-requirements)
4. [Configuration Reference](#4-configuration-reference)
5. [Data Pipeline](#5-data-pipeline)
6. [Pattern Matching System](#6-pattern-matching-system)
7. [Context-Aware Extraction](#7-context-aware-extraction)
8. [Model Architecture](#8-model-architecture)
9. [Training Process](#9-training-process)
10. [Evaluation Metrics](#10-evaluation-metrics)
11. [Interpreting Results](#11-interpreting-results)
12. [Troubleshooting](#12-troubleshooting)

---

## 1. Overview

### 1.1 What is BIP?

The Bond Invariance Principle (BIP) hypothesizes that moral concepts share mathematical structure across languages and cultures. This notebook tests whether:

1. Moral patterns can be extracted from native text without translation
2. These patterns transfer across language families
3. The learned representations are language-invariant

### 1.2 Key Innovation

**No English translation bridge.** Unlike typical cross-lingual NLP that translates to English, BIP extracts moral labels directly from native scripts (Hebrew, Aramaic, Arabic, Chinese) using native-language patterns.

### 1.3 What's New in v10.2

| Feature | Description |
|---------|-------------|
| Expanded Chinese corpus | 211 real classical texts (Analects, Mencius, Daodejing, etc.) |
| Expanded Islamic corpus | 152 real Quranic verses and Hadith |
| Grammar-aware extraction | Detects prescriptive vs. descriptive context |
| Confidence weighting | Weights moral statements higher in loss |
| Context auxiliary task | Multi-task learning with context prediction |
| Strict prescriptive test | Option to test only on moral statements |
| Data validation | Warnings for insufficient corpora |

---

## 2. Theoretical Foundation

### 2.1 Bond Types

Nine universal moral bond categories derived from comparative ethics research:

| Bond Type | Description | Example Concepts |
|-----------|-------------|------------------|
| `HARM_PREVENTION` | Protection from physical/emotional harm | Kill, murder, save, protect |
| `RECIPROCITY` | Mutual exchange and return | Repay, exchange, give back |
| `AUTONOMY` | Freedom of choice and self-determination | Free, choose, consent |
| `PROPERTY` | Ownership and material rights | Steal, own, inherit |
| `FAMILY` | Kinship obligations | Honor parents, care for children |
| `AUTHORITY` | Legitimate power and obedience | Obey, command, law |
| `CARE` | Compassion and helping | Mercy, charity, kindness |
| `FAIRNESS` | Justice and equality | Just, equal, fair |
| `CONTRACT` | Promises and agreements | Vow, oath, promise |

### 2.2 Hohfeld Deontic States

Four legal/moral modalities from Wesley Hohfeld's framework:

| State | Meaning | Hebrew | Arabic | Chinese |
|-------|---------|--------|--------|---------|
| `OBLIGATION` | Must do | חייב | يجب | 必 |
| `RIGHT` | Entitled to | זכות | حق | 權 |
| `LIBERTY` | Permitted to | מותר | مباح | 可 |
| `NO_RIGHT` | Forbidden | אסור | حرام | 禁 |

### 2.3 Context Types

Grammar-aware classification of text context:

| Context | Description | Training Weight |
|---------|-------------|-----------------|
| `prescriptive` | Moral instruction ("thou shalt not") | 2.0x (high) |
| `descriptive` | Narrative ("he killed") | 1.0x (medium) |
| `unknown` | Uncertain context | 0.5x (low) |

---

## 3. System Requirements

### 3.1 Hardware Tiers

| Tier | GPU | VRAM | RAM | Batch Size | Recommended For |
|------|-----|------|-----|------------|-----------------|
| L4/A100 | L4 or A100 | ≥22GB | ≥50GB | 512 | Full experiment |
| T4 | T4 | ≥14GB | ≥24GB | 256 | Standard run |
| Small | Any | ≥10GB | ≥12GB | 128 | Testing |
| Minimal | CPU | <10GB | <12GB | 64 | Development only |

### 3.2 Colab Runtime Selection

1. Go to `Runtime` → `Change runtime type`
2. Select GPU: **L4** (recommended) or T4 (free tier)
3. Enable **High-RAM** if available
4. Click Save

### 3.3 Dependencies

Automatically installed by Cell 1:
- `transformers` - Hugging Face transformer models
- `sentence-transformers` - Multilingual embeddings
- `torch` - PyTorch deep learning
- `pandas` - Data manipulation
- `scikit-learn` - Evaluation metrics
- `tqdm` - Progress bars
- `pyyaml` - Configuration files
- `psutil` - Hardware detection

---

## 4. Configuration Reference

### 4.1 Cell 1: Data Source Configuration

```python
USE_DRIVE_DATA = True          # Load pre-processed data from Google Drive
REFRESH_DATA_FROM_SOURCE = False  # Re-download even if Drive data exists
DRIVE_FOLDER = "BIP_v10"       # Google Drive folder name
```

| Setting | True | False |
|---------|------|-------|
| `USE_DRIVE_DATA` | Load from Drive (fast) | Download fresh (slow) |
| `REFRESH_DATA_FROM_SOURCE` | Force re-download | Use cached if available |

### 4.2 Cell 7: Training Configuration

```python
# Splits to train
TRAIN_HEBREW_TO_OTHERS = True
TRAIN_SEMITIC_TO_NON_SEMITIC = True
TRAIN_ANCIENT_TO_MODERN = True
TRAIN_MIXED_BASELINE = True

# Adversarial weights
LANG_WEIGHT = 0.01    # Language adversarial loss weight
PERIOD_WEIGHT = 0.01  # Period adversarial loss weight
N_EPOCHS = 5          # Training epochs per split

# Context-aware training
USE_CONFIDENCE_WEIGHTING = True   # Weight prescriptive examples 2x
USE_CONTEXT_AUXILIARY = True      # Context prediction auxiliary task
CONTEXT_LOSS_WEIGHT = 0.1         # Weight for context loss
STRICT_PRESCRIPTIVE_TEST = False  # Test only on prescriptive examples
```

### 4.3 Recommended Configurations

| Experiment Type | Confidence Weight | Context Aux | Strict Test |
|-----------------|-------------------|-------------|-------------|
| Standard | True | True | False |
| Maximum precision | True | True | True |
| Ablation (baseline) | False | False | False |
| Fast iteration | True | False | False |

---

## 5. Data Pipeline

### 5.1 Corpora Sources

| Corpus | Language | Source | Size (v10.2) |
|--------|----------|--------|--------------|
| Sefaria | Hebrew/Aramaic | GitHub clone | ~200K passages |
| Chinese Classics | Classical Chinese | Built-in | 211 passages |
| Islamic Texts | Arabic | Built-in | 152 passages |
| Dear Abby | English | Kaggle | ~20K letters |

### 5.2 Data Flow

```
┌─────────────────┐
│  Raw Sources    │
│  - Sefaria Git  │
│  - Kaggle CSV   │
│  - Built-in JSON│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Cell 2: Load   │
│  - Download/copy│
│  - Validate     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Cell 4: Extract│
│  - Normalize    │
│  - Pattern match│
│  - Context detect│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Output Files   │
│  - passages.jsonl│
│  - bonds.jsonl  │
└─────────────────┘
```

### 5.3 File Formats

**passages.jsonl** - One JSON object per line:
```json
{
  "id": "sef_abc123",
  "text": "לא תרצח",
  "language": "hebrew",
  "time_period": "BIBLICAL",
  "source": "Exodus",
  "source_type": "sefaria",
  "century": -13
}
```

**bonds.jsonl** - One JSON object per line:
```json
{
  "passage_id": "sef_abc123",
  "bonds": {
    "primary_bond": "HARM_PREVENTION",
    "all_bonds": ["HARM_PREVENTION"],
    "hohfeld": "NO_RIGHT",
    "language": "hebrew",
    "context": "prescriptive",
    "confidence": "high"
  }
}
```

### 5.4 Train/Test Splits

| Split Name | Train Set | Test Set | Purpose |
|------------|-----------|----------|---------|
| `hebrew_to_others` | Hebrew | Aramaic, Chinese, Arabic, English | Single-language transfer |
| `semitic_to_non_semitic` | Hebrew, Aramaic, Arabic | Chinese, English | Language family transfer |
| `ancient_to_modern` | Ancient periods | Modern periods | Temporal transfer |
| `mixed_baseline` | 70% random | 30% random | Upper bound control |

---

## 6. Pattern Matching System

### 6.1 Text Normalization

Each language has custom normalization to handle script variations:

**Hebrew/Aramaic:**
```python
def normalize_hebrew(text):
    # Remove nikud (vowel points): בָּרוּךְ → ברוך
    text = re.sub(r'[\u0591-\u05C7]', '', text)
    # Normalize final letters: ך→כ, ם→מ, ן→נ, ף→פ, ץ→צ
    for final, regular in [('ך','כ'), ('ם','מ'), ...]:
        text = text.replace(final, regular)
    return text
```

**Arabic:**
```python
def normalize_arabic(text):
    # Remove tashkeel (diacritics): مُحَمَّد → محمد
    text = re.sub(r'[\u064B-\u065F]', '', text)
    # Normalize alef variants: أ,إ,آ,ٱ → ا
    # Normalize ta marbuta: ة → ه
    # Normalize alef maksura: ى → ي
    return text
```

### 6.2 Bond Pattern Examples

| Language | Bond | Pattern | Matches |
|----------|------|---------|---------|
| Hebrew | HARM_PREVENTION | `הרג` | הרג, להרוג, הריגה |
| Hebrew | FAMILY | `כבד.*אב` | כבד את אביך |
| Arabic | HARM_PREVENTION | `قتل` | قتل, يقتل, القتل |
| Arabic | PROHIBITION | `حرام` | حرام, محرم |
| Chinese | FAMILY | `孝` | 孝, 孝悌, 不孝 |
| Chinese | OBLIGATION | `必` | 必, 必須 |
| English | HARM_PREVENTION | `\bkill` | kill, killing, killed |

### 6.3 Pattern Matching Priority

Patterns are checked in order; first match wins:
1. HARM_PREVENTION
2. RECIPROCITY
3. AUTONOMY
4. PROPERTY
5. FAMILY
6. AUTHORITY
7. CARE
8. FAIRNESS
9. CONTRACT
10. NONE (default)

---

## 7. Context-Aware Extraction

### 7.1 Context Detection Algorithm

```python
def detect_context(text, language, match_pos, window=30):
    """
    Analyze 30 characters around pattern match for grammatical context.
    """
    window_text = text[match_pos-30 : match_pos+30]

    # Check for deontic markers (prescriptive)
    if has_prohibition_marker(window_text, language):
        return 'prescriptive', 'prohibition'
    if has_obligation_marker(window_text, language):
        return 'prescriptive', 'obligation'
    if has_permission_marker(window_text, language):
        return 'prescriptive', 'permission'

    # Check for simple negation (may be descriptive)
    if has_negation_marker(window_text, language):
        return 'descriptive', 'negated'

    return 'descriptive', None
```

### 7.2 Context Markers by Language

**Hebrew:**
| Type | Markers |
|------|---------|
| Negation | לא, אל, אין, בלי |
| Obligation | חייב, צריך, מוכרח |
| Prohibition | אסור, אל ת- |
| Permission | מותר, רשאי, פטור |

**Arabic:**
| Type | Markers |
|------|---------|
| Negation | لا, ما, ليس, لم |
| Obligation | يجب, واجب, فرض |
| Prohibition | حرام, محرم, لا يجوز |
| Permission | حلال, مباح, جائز |

**Chinese:**
| Type | Markers |
|------|---------|
| Negation | 不, 非, 無, 未 |
| Obligation | 必, 當, 須, 應 |
| Prohibition | 勿, 禁, 莫, 不可 |
| Permission | 可, 得, 許 |

### 7.3 Context Examples

| Text | Bond | Context | Confidence |
|------|------|---------|------------|
| לא תרצח (Thou shalt not kill) | HARM_PREVENTION | prescriptive | high |
| והוא הרג (And he killed) | HARM_PREVENTION | descriptive | medium |
| حرام القتل (Killing is forbidden) | HARM_PREVENTION | prescriptive | high |
| 殺人者死 (Who kills shall die) | HARM_PREVENTION | prescriptive | high |

---

## 8. Model Architecture

### 8.1 Architecture Overview

```
┌─────────────────────────────────────────────────┐
│                  Input Text                     │
│         "לא תרצח" (Thou shalt not kill)         │
└────────────────────┬────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────┐
│     Multilingual MiniLM Encoder (384-dim)       │
│   sentence-transformers/paraphrase-multilingual │
└────────────────────┬────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────┐
│           Z Projection (384 → 64-dim)           │
│     Linear → LayerNorm → GELU → Dropout → Linear│
└────────────────────┬────────────────────────────┘
                     │
          ┌──────────┼──────────┐
          │          │          │
          ▼          ▼          ▼
    ┌──────────┐ ┌──────────┐ ┌───────────┐
    │Bond Head │ │Context   │ │Adversarial│
    │(10 class)│ │Head (3)  │ │Heads      │
    └──────────┘ └──────────┘ └───────────┘
                              ┌─────┴─────┐
                              ▼           ▼
                         ┌────────┐ ┌────────┐
                         │Language│ │Period  │
                         │Head (5)│ │Head(10)│
                         └────────┘ └────────┘
```

### 8.2 Component Details

**Encoder:** `paraphrase-multilingual-MiniLM-L12-v2`
- 12 transformer layers
- 384 hidden dimensions
- Trained on 50+ languages including Hebrew, Arabic, Chinese

**Z Projection:**
```python
nn.Sequential(
    nn.Linear(384, 256),
    nn.LayerNorm(256),
    nn.GELU(),
    nn.Dropout(0.1),
    nn.Linear(256, 64)
)
```

**Task Heads:**
- Bond prediction: 64 → 10 classes
- Hohfeld prediction: 64 → 4 classes
- Context prediction: 64 → 3 classes (auxiliary)

**Adversarial Heads (with gradient reversal):**
- Language prediction: 64 → 5 classes
- Period prediction: 64 → 10 classes

### 8.3 Gradient Reversal Layer

```python
class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x  # Pass through unchanged

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None  # Reverse gradient
```

Purpose: Forces the z-space to NOT encode language/period information, making it invariant.

---

## 9. Training Process

### 9.1 Loss Function

```python
# Main task: Bond prediction (weighted by confidence)
if USE_CONFIDENCE_WEIGHTING:
    loss_bond = F.cross_entropy(bond_pred, bond_labels, reduction='none')
    loss_bond = (loss_bond * sample_weights).mean()  # weights: 2.0/1.0/0.5
else:
    loss_bond = F.cross_entropy(bond_pred, bond_labels)

# Auxiliary task: Context prediction
if USE_CONTEXT_AUXILIARY:
    loss_context = F.cross_entropy(context_pred, context_labels)
else:
    loss_context = 0

# Adversarial tasks: Language and period (gradient reversed)
loss_lang = F.cross_entropy(language_pred, language_labels)
loss_period = F.cross_entropy(period_pred, period_labels)

# Total loss
loss = loss_bond
     + CONTEXT_LOSS_WEIGHT * loss_context  # 0.1
     + LANG_WEIGHT * loss_lang             # 0.01
     + PERIOD_WEIGHT * loss_period         # 0.01
```

### 9.2 Adversarial Lambda Schedule

```python
def get_adv_lambda(epoch, warmup=2):
    """Gradually increase adversarial strength."""
    if epoch <= warmup:
        return 0.1 + 0.9 * (epoch / warmup)
    return 1.0

# Epoch 1: lambda = 0.55 (weak adversarial)
# Epoch 2: lambda = 1.0  (full adversarial)
# Epoch 3+: lambda = 1.0
```

### 9.3 Training Loop

```python
for epoch in range(1, N_EPOCHS + 1):
    model.train()
    for batch in train_loader:
        optimizer.zero_grad()

        # Forward pass with AMP
        with torch.amp.autocast('cuda', enabled=USE_AMP):
            out = model(input_ids, attention_mask, adv_lambda)
            loss = compute_loss(out, batch)

        # Backward pass with gradient scaling
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

    # Save best model
    if avg_loss < best_loss:
        torch.save(model.state_dict(), f'best_{split_name}.pt')
```

### 9.4 Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Learning rate | `2e-5 * (batch/256)` | Scaled by batch size |
| Weight decay | 0.01 | Standard for transformers |
| Gradient clip | 1.0 | Prevent exploding gradients |
| Adversarial weight | 0.01 | Prevent loss explosion |
| Context weight | 0.1 | Auxiliary task |
| Epochs | 5 | Sufficient for convergence |

---

## 10. Evaluation Metrics

### 10.1 Primary Metrics

**Bond F1 (Macro):**
- Measures bond prediction accuracy
- Macro-averaged across all 10 bond classes
- **Target:** >0.13 (>1.3x chance) indicates transfer

**Language Accuracy:**
- Measures if model encodes language information
- **Target:** ~20% (chance for 5 languages) = invariant
- >35% indicates language leakage

### 10.2 Success Criteria

| Metric | Threshold | Interpretation |
|--------|-----------|----------------|
| Bond F1 > 1.3x chance | >0.13 | Transfer works |
| Language acc < 35% | <0.35 | Representation is invariant |
| Both conditions met | - | **SUCCESS** |

### 10.3 Linear Probe Test

Trains a logistic regression on frozen z-embeddings to predict language/period:

```python
# Extract embeddings
z_embeddings = model.get_z(test_data)

# Train probe
probe = LogisticRegression()
probe.fit(z_train, y_train)

# Evaluate
accuracy = probe.score(z_test, y_test)
# If accuracy ≈ chance → representation is invariant
```

### 10.4 Verdict Categories

| Verdict | Criteria | Meaning |
|---------|----------|---------|
| STRONGLY_SUPPORTED | ≥2 successful splits | Multiple transfer paths work |
| SUPPORTED | 1 successful split | At least one transfer works |
| PARTIAL | F1 > 0.13 but leaking | Signal present but not invariant |
| INCONCLUSIVE | No transfer | Hypothesis not supported |

---

## 11. Interpreting Results

### 11.1 Expected Output

```
============================================================
FINAL BIP EVALUATION (v10.2)
============================================================

Hardware: L4/A100 (24GB VRAM, 51GB RAM)

------------------------------------------------------------
CROSS-DOMAIN TRANSFER RESULTS
------------------------------------------------------------

hebrew_to_others:
  Bond F1:     0.180 (1.8x chance) OK
  Language:    10.0% INVARIANT
  Context: 3,421/15,000 prescriptive (22.8%)
  High confidence: 3,421/15,000 (22.8%)
  -> SUCCESS

semitic_to_non_semitic:
  Bond F1:     0.156 (1.6x chance) OK
  Language:    18.2% INVARIANT
  Context: 892/5,000 prescriptive (17.8%)
  High confidence: 892/5,000 (17.8%)
  -> SUCCESS

ancient_to_modern:
  Bond F1:     0.465 (4.6x chance) OK
  Language:    26.3% INVARIANT
  -> SUCCESS

mixed_baseline:
  Bond F1:     0.734 (7.3x chance) OK
  Language:    57.2% LEAKING
  -> PARTIAL

------------------------------------------------------------
VERDICT: STRONGLY_SUPPORTED
Multiple independent transfer paths demonstrate universal structure
------------------------------------------------------------
```

### 11.2 Result Analysis Guide

**Good Results:**
- Bond F1 increasing across epochs
- Language accuracy near chance (20%)
- Multiple splits showing SUCCESS
- Prescriptive % > 15%

**Warning Signs:**
- Bond F1 < 0.1 (no transfer)
- Language accuracy > 50% (heavy leakage)
- Very low prescriptive % (data quality issue)
- All splits FAIL

### 11.3 Per-Language Analysis

```
Per-language:
  aramaic:      F1=0.234 (n=45,231)
  hebrew:       F1=0.198 (n=123,456)
  english:      F1=0.167 (n=18,234)
  arabic:       F1=0.145 (n=152)
  chinese:      F1=0.089 (n=211)
```

- Higher n = more reliable F1
- Low n languages need more data
- Compare F1 across languages for consistency

---

## 12. Troubleshooting

### 12.1 Common Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `CUDA out of memory` | Batch too large | Reduce BATCH_SIZE or use smaller GPU tier |
| `No training data` | Drive sync issue | Set REFRESH_DATA_FROM_SOURCE=True |
| `Test set too small` | Missing corpus | Upload dear_abby.csv to Drive |
| `Kaggle download failed` | API not configured | Upload CSV manually to Drive |
| `SyntaxError: unterminated string` | Notebook corruption | Re-download notebook |

### 12.2 Data Issues

**semitic_to_non_semitic FAIL:**
```
Bond F1: 0.007 (0.1x chance) WEAK
```
Cause: Insufficient non-Semitic data (Chinese + English < 500 samples)

Fix:
1. Upload real `dear_abby.csv` to Drive (20K samples)
2. Set `REFRESH_DATA_FROM_SOURCE = True`
3. Rerun from Cell 1

### 12.3 Performance Optimization

| Issue | Solution |
|-------|----------|
| Training too slow | Use L4 GPU instead of T4 |
| Memory errors | Reduce MAX_PER_LANG |
| Low F1 scores | Increase N_EPOCHS to 7-10 |
| High language leakage | Reduce LANG_WEIGHT to 0.005 |

### 12.4 Validation Checklist

Before running:
- [ ] GPU runtime selected (L4 recommended)
- [ ] Drive mounted successfully
- [ ] dear_abby.csv in BIP_v10 folder (>10KB)
- [ ] All corpus counts > 100

After Cell 4:
- [ ] Hebrew passages > 10,000
- [ ] English passages > 1,000
- [ ] Chinese passages > 100
- [ ] Prescriptive % > 10%

After training:
- [ ] At least 2 splits show SUCCESS
- [ ] Language accuracy < 35% on successful splits
- [ ] Models saved to Drive

---

## Appendix A: File Structure

```
/content/
├── drive/MyDrive/BIP_v10/
│   ├── passages.jsonl      # Processed text corpus
│   ├── bonds.jsonl         # Extracted bonds + context
│   ├── all_splits.json     # Train/test split definitions
│   ├── dear_abby.csv       # English corpus (upload manually)
│   ├── best_hebrew_to_others.pt
│   ├── best_semitic_to_non_semitic.pt
│   ├── best_ancient_to_modern.pt
│   ├── best_mixed_baseline.pt
│   └── final_results.json
├── data/
│   ├── raw/
│   │   ├── Sefaria-Export/  # Cloned from GitHub
│   │   ├── chinese/
│   │   ├── islamic/
│   │   └── dear_abby.csv
│   ├── processed/
│   │   ├── passages.jsonl
│   │   └── bonds.jsonl
│   └── splits/
│       └── all_splits.json
├── models/checkpoints/
│   └── best_*.pt
└── results/
    └── final_results.json
```

---

## Appendix B: API Reference

### BIPModel

```python
class BIPModel(nn.Module):
    def __init__(self, z_dim=64):
        """
        Initialize BIP model.

        Args:
            z_dim: Dimension of bond embedding space (default: 64)
        """

    def forward(self, input_ids, attention_mask, adv_lambda=1.0):
        """
        Forward pass.

        Args:
            input_ids: Tokenized input [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            adv_lambda: Adversarial gradient reversal strength

        Returns:
            dict with keys:
                - bond_pred: [batch, 10] bond logits
                - hohfeld_pred: [batch, 4] hohfeld logits
                - context_pred: [batch, 3] context logits
                - language_pred: [batch, 5] language logits
                - period_pred: [batch, 10] period logits
                - z: [batch, 64] bond embedding
        """
```

### NativeDataset

```python
class NativeDataset(Dataset):
    def __init__(self, ids_set, passages_file, bonds_file, tokenizer, max_len=128):
        """
        Dataset for native-language moral texts.

        Args:
            ids_set: Set of passage IDs to include
            passages_file: Path to passages.jsonl
            bonds_file: Path to bonds.jsonl
            tokenizer: HuggingFace tokenizer
            max_len: Maximum sequence length (default: 128)
        """

    def __getitem__(self, idx):
        """
        Returns:
            dict with keys:
                - input_ids: [max_len] token IDs
                - attention_mask: [max_len] attention mask
                - bond_label: int (0-9)
                - language_label: int (0-4)
                - period_label: int (0-9)
                - hohfeld_label: int (0-3)
                - context_label: int (0-2)
                - sample_weight: float (0.5, 1.0, or 2.0)
                - context: str
                - confidence: str
                - language: str
        """
```

---

## Appendix C: Citation

If using this work, please cite:

```bibtex
@software{bip_v10,
  title = {Bond Invariance Principle: Native-Language Moral Pattern Transfer},
  version = {10.2},
  year = {2025},
  note = {Experimental notebook for cross-lingual moral concept transfer}
}
```

---

*End of Technical Guide*
