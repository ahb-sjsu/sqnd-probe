# BIP v10.3: Cross-Lingual Moral Pattern Transfer

## Overview

The **Bond Invariance Principle (BIP)** experiments test whether moral/ethical patterns transfer across languages without using English as a translation bridge. This notebook trains a model on moral texts in one language family and tests whether it can recognize the same moral patterns in unrelated languages.

**Key Hypothesis:** Deontic moral structures (obligations, prohibitions, permissions) are language-invariant and can be detected through native-language pattern matching.

## v10.3 Features

### Expanded Corpora

| Language | Source | Size | Access Method |
|----------|--------|------|---------------|
| Hebrew | [Sefaria](https://github.com/Sefaria/Sefaria-Export) | ~100K passages | GitHub ZIP |
| Aramaic | Sefaria (Talmud) | ~100K passages | GitHub ZIP |
| Classical Chinese | [ctext.org](https://ctext.org/tools/api) | 2-5K passages | REST API |
| Arabic | [Kaggle quran-nlp](https://www.kaggle.com/datasets/alizahidraja/quran-nlp) | 6K+ verses | Kaggle CLI |
| Arabic (fallback) | [Tanzil.net](https://tanzil.net/download/) | 6,236 verses | Direct HTTP |
| English (advice) | [Kaggle Dear Abby](https://www.kaggle.com/datasets/thedevastator/20000-dear-abby-questions) | ~68K letters | Kaggle CLI |
| English (classics) | [MIT Classics Archive](https://classics.mit.edu/) | 3-5K paragraphs | Direct HTTP |

### Classical Chinese Texts (ctext.org)
- 論語 (Analects of Confucius)
- 孟子 (Mencius)
- 道德經 (Tao Te Ching)
- 荀子 (Xunzi)
- 孝經 (Classic of Filial Piety)
- 禮記 (Book of Rites)
- 墨子 (Mozi)
- 莊子 (Zhuangzi)
- 論衡 (Lunheng)
- 韓非子 (Han Feizi)

### Western Philosophy Classics (MIT)
- Aristotle: *Nicomachean Ethics*, *Politics*
- Plato: *Republic*, *Laws*
- Marcus Aurelius: *Meditations*
- Epictetus: *Enchiridion*
- Cicero: *De Officiis*

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    XLM-RoBERTa Base                         │
│                  (Multilingual Encoder)                     │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                   Shared Representation                      │
│                      (768-dim)                              │
└─────────────────────────────────────────────────────────────┘
           │                    │                    │
           ▼                    ▼                    ▼
    ┌────────────┐      ┌────────────┐      ┌────────────┐
    │   Bond     │      │  Language  │      │   Period   │
    │ Classifier │      │ Adversary  │      │ Adversary  │
    │ (9 types)  │      │  (GRL)     │      │   (GRL)    │
    └────────────┘      └────────────┘      └────────────┘
           │                    │                    │
           ▼                    ▼                    ▼
    ┌────────────┐      ┌────────────┐      ┌────────────┐
    │  Context   │      │  Predict   │      │  Predict   │
    │   Head     │      │  Language  │      │  Period    │
    │(prescriptive)│    │ (confuse)  │      │ (confuse)  │
    └────────────┘      └────────────┘      └────────────┘
```

### Bond Types (9 classes)
| Bond | Description | Example Pattern |
|------|-------------|-----------------|
| HARM_PREVENTION | Prohibitions against harm | "thou shalt not kill" |
| RECIPROCITY | Mutual obligations | "do unto others" |
| AUTONOMY | Personal freedom/choice | "right to decide" |
| PROPERTY | Ownership rights | "shall not steal" |
| FAMILY | Familial duties | "honor thy father" |
| AUTHORITY | Hierarchical obligations | "obey the law" |
| CARE | Duty to protect/nurture | "care for the widow" |
| FAIRNESS | Justice/equality | "just weights" |
| CONTRACT | Agreement obligations | "keep thy covenant" |

### Hohfeld Deontic States
- **OBLIGATION**: Must do X
- **PROHIBITION**: Must not do X (mapped to OBLIGATION)
- **PERMISSION**: May do X
- **RIGHT**: Entitled to X

## Configuration

### Cell 1 Settings

```python
# Data source
USE_DRIVE_DATA = True          # Use cached data from Google Drive
REFRESH_DATA_FROM_SOURCE = False  # Re-download all corpora
DRIVE_FOLDER = '/content/drive/MyDrive/BIP_v10'

# Training
USE_CONFIDENCE_WEIGHTING = True   # Weight prescriptive examples higher
USE_CONTEXT_AUXILIARY = True      # Train context prediction head
CONTEXT_LOSS_WEIGHT = 0.1         # Auxiliary task weight
STRICT_PRESCRIPTIVE_TEST = False  # Filter test set to prescriptive only
```

### Hardware Auto-Detection

| GPU | Batch Size | Max/Lang | Learning Rate |
|-----|------------|----------|---------------|
| L4/A100 | 512 | 100,000 | 4e-5 |
| T4 | 256 | 50,000 | 3e-5 |
| CPU | 32 | 10,000 | 2e-5 |

## Running the Experiment

### First Run (Download Data)
```python
USE_DRIVE_DATA = False
REFRESH_DATA_FROM_SOURCE = True
```

### Subsequent Runs (Use Cache)
```python
USE_DRIVE_DATA = True
REFRESH_DATA_FROM_SOURCE = False
```

### Kaggle Authentication
For Dear Abby and Quran-NLP datasets, you need Kaggle credentials:
1. Create account at kaggle.com
2. Go to Settings → API → Create New Token
3. Upload `kaggle.json` to Colab or set as Colab secret

If Kaggle fails, the notebook falls back to:
- Tanzil.net for Arabic (Quran verses)
- MIT Classics for additional English

## Evaluation Splits

| Split | Train | Test | Purpose |
|-------|-------|------|---------|
| hebrew_to_others | Hebrew | Aramaic, Arabic, Chinese, English | Single-source transfer |
| semitic_to_non_semitic | Hebrew, Aramaic, Arabic | Chinese, English | Cross-family transfer |
| ancient_to_modern | Biblical, Classical | Modern texts | Temporal transfer |

### Success Criteria
- **PASS**: p < 0.05 (binomial test), accuracy > 1.5× chance
- **FAIL**: p ≥ 0.05 or accuracy ≤ chance

## Expected Results

### Good Results
```
hebrew_to_others:        PASS (2.1x chance, p<0.001)
semitic_to_non_semitic:  PASS (1.8x chance, p<0.01)
ancient_to_modern:       PASS (2.3x chance, p<0.001)
```

### Minimum Corpus Sizes
| Language | Minimum | Recommended |
|----------|---------|-------------|
| Hebrew | 10,000 | 50,000+ |
| Aramaic | 5,000 | 50,000+ |
| Classical Chinese | 500 | 2,000+ |
| Arabic | 500 | 5,000+ |
| English | 5,000 | 50,000+ |

## Context-Aware Extraction

The v10.2+ notebooks detect whether moral statements are **prescriptive** (commands) or **descriptive** (reports):

```python
CONTEXT_MARKERS = {
    'hebrew': {
        'obligation': [r'חייב', r'צריך', r'מוכרח'],  # must, need, obligated
        'prohibition': [r'אסור', r'אל'],              # forbidden, don't
        'permission': [r'מותר', r'רשאי'],             # permitted, allowed
    },
    'classical_chinese': {
        'obligation': [r'必', r'當', r'宜'],          # must, should, ought
        'prohibition': [r'不可', r'勿', r'毋'],       # cannot, don't
    },
    # ... similar for other languages
}
```

Prescriptive examples receive 2× weight during training.

## Files

| File | Description |
|------|-------------|
| `BIP_v10.3_expanded.ipynb` | Main notebook with all corpora |
| `BIP_v10.2_Technical_Guide.md` | Detailed technical documentation |
| `BIP_v10.2_expanded.ipynb` | Previous version (Kaggle only) |

## API Rate Limits

| Source | Rate Limit | Handling |
|--------|------------|----------|
| ctext.org | Unauthenticated: limited | 0.2s delay between requests |
| Kaggle | Authenticated | Requires kaggle.json |
| Tanzil.net | None | Direct download |
| MIT Classics | None | Direct download |

## Troubleshooting

### "NameError: by_lang is not defined"
The corpus adequacy check ran before data was loaded. Ensure `SKIP_PROCESSING` is correctly set based on whether Drive data exists.

### "Kaggle API not found"
```python
!pip install kaggle
# Upload kaggle.json or set KAGGLE_USERNAME/KAGGLE_KEY secrets
```

### "ctext.org timeout"
The API may be slow. Increase timeout or reduce number of texts:
```python
CHINESE_TEXTS = CHINESE_TEXTS[:5]  # Fewer texts
```

### Low Chinese/Arabic counts
If counts are below minimum, ensure `REFRESH_DATA_FROM_SOURCE = True` on first run.

## Citation

If you use this work, please cite:

```bibtex
@software{bip_v10,
  title = {Bond Invariance Principle: Cross-Lingual Moral Pattern Transfer},
  version = {10.3},
  year = {2025},
  url = {https://github.com/ahb-sjsu/sqnd-probe}
}
```

## License

Apache 2.0 (code), various (corpora - see individual sources)
