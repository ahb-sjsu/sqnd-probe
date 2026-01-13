# BIP v10.4: Parallel Streaming Multi-Corpus Loader

## What's New in v10.4

### Parallel Threaded Downloads
All 8 corpus sources download **simultaneously** using `ThreadPoolExecutor`:

```
┌─────────────────────────────────────────────────────────────┐
│                 8 PARALLEL DOWNLOAD THREADS                  │
├─────────┬─────────┬─────────┬─────────┬─────────┬─────────┬─┤
│ Sefaria │ UniMoral│  Bible  │UN Corpus│ CText   │ Tanzil  │…│
└────┬────┴────┬────┴────┬────┴────┬────┴────┬────┴────┬────┴─┘
     │         │         │         │         │         │
     ▼         ▼         ▼         ▼         ▼         ▼
┌─────────────────────────────────────────────────────────────┐
│              THREAD-SAFE QUEUE (100K capacity)               │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
                   Bond Extraction + Training
```

### New Corpus Sources

| Corpus | Size | Languages | Content Type | Speed |
|--------|------|-----------|--------------|-------|
| **[UN Parallel](https://huggingface.co/datasets/Helsinki-NLP/un_pc)** | 59GB (streamed) | ar, zh, en, fr, ru, es | Legal/rights | HF streaming |
| **[UniMoral](https://huggingface.co/datasets/shivaniku/UniMoral)** | ~50MB | ar, zh, en, hi, ru, es | Labeled moral dilemmas | HF instant |
| **[Bible 100-lang](https://github.com/christos-c/bible-corpus)** | ~200MB | he, ar, zh, en + 96 more | Parallel moral verses | GitHub raw |
| **[Hendrycks Ethics](https://huggingface.co/datasets/hendrycks/ethics)** | ~20MB | en | Justice, virtue, duty | HF streaming |
| **[Sefaria](https://github.com/Sefaria/Sefaria-Export)** | ~8GB | he, aramaic | Torah, Talmud, Midrash | GitHub ZIP |
| **[ctext.org](https://ctext.org/tools/api)** | API | classical_chinese | Confucian, Taoist texts | REST API |
| **[Tanzil](https://tanzil.net/download/)** | ~5MB | ar | Quran (verified text) | Direct HTTP |
| **[MIT Classics](https://classics.mit.edu/)** | ~2MB | en | Greek/Roman philosophy | Direct HTTP |

### Total Potential Corpus Size

| Language | Sources | Estimated Passages |
|----------|---------|-------------------|
| Hebrew | Sefaria, Bible | 100K+ |
| Aramaic | Sefaria | 100K+ |
| Arabic | UN, UniMoral, Bible, Tanzil | 50K+ |
| Classical Chinese | UN, UniMoral, Bible, ctext.org | 10K+ |
| English | UN, UniMoral, Bible, Hendrycks, MIT | 100K+ |

## Architecture

### Parallel Download System

```python
# 8 threads download simultaneously
with ThreadPoolExecutor(max_workers=8) as executor:
    futures = {
        executor.submit(load_sefaria_thread): "Sefaria",
        executor.submit(load_unimoral_thread): "UniMoral",
        executor.submit(load_un_corpus_thread): "UN Corpus",
        executor.submit(load_bible_thread): "Bible",
        executor.submit(load_hendrycks_ethics_thread): "Ethics",
        executor.submit(load_ctext_thread): "CText",
        executor.submit(load_tanzil_thread): "Tanzil",
        executor.submit(load_mit_classics_thread): "MIT Classics",
    }
```

### Thread-Safe Queue

```python
passage_queue = Queue(maxsize=100000)  # Shared across all threads
corpus_stats = defaultdict(int)         # Protected by stats_lock
```

## Model Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    XLM-RoBERTa Base                         │
│                  (Multilingual Encoder)                     │
└─────────────────────────────────────────────────────────────┘
                            │
           ┌────────────────┼────────────────┐
           ▼                ▼                ▼
    ┌────────────┐   ┌────────────┐   ┌────────────┐
    │   Bond     │   │  Language  │   │   Period   │
    │ Classifier │   │ Adversary  │   │ Adversary  │
    │ (9 types)  │   │   (GRL)    │   │   (GRL)    │
    └────────────┘   └────────────┘   └────────────┘
```

### Bond Types
- HARM_PREVENTION, RECIPROCITY, AUTONOMY, PROPERTY
- FAMILY, AUTHORITY, CARE, FAIRNESS, CONTRACT

### Hohfeld Deontic States
- OBLIGATION, PROHIBITION, PERMISSION, RIGHT

## Configuration

```python
# Cell 1 Settings
USE_DRIVE_DATA = True           # Use cached processed data
REFRESH_DATA_FROM_SOURCE = True # Download all corpora fresh
DRIVE_FOLDER = '/content/drive/MyDrive/BIP_v10'

# Training
USE_CONFIDENCE_WEIGHTING = True
USE_CONTEXT_AUXILIARY = True
STRICT_PRESCRIPTIVE_TEST = False
```

## Running the Experiment

### First Run (Download Everything)
```python
USE_DRIVE_DATA = False
REFRESH_DATA_FROM_SOURCE = True
```

Expected output:
```
PARALLEL STREAMING DOWNLOAD
Max per language: 100,000
========================================

Launching 8 download threads...
[Tanzil] Done: 6,236 verses
[MIT Classics] Done: 1,200 passages
[CText] Done: 2,500 passages
[Bible hebrew] 31,102 verses
[Bible arabic] 31,102 verses
...
Progress: 8/8 complete

Total passages collected: 250,000+
```

### Subsequent Runs (Use Cache)
```python
USE_DRIVE_DATA = True
REFRESH_DATA_FROM_SOURCE = False
```

## Hardware Requirements

| GPU | Batch Size | Download Time | Training Time |
|-----|------------|---------------|---------------|
| L4/A100 | 512 | ~5-10 min | ~15 min |
| T4 | 256 | ~5-10 min | ~25 min |
| CPU | 32 | ~10-15 min | ~2 hours |

## Dependencies

```python
# Installed automatically in Colab
pip install datasets transformers torch tqdm requests
```

## Evaluation Splits

| Split | Train Languages | Test Languages |
|-------|-----------------|----------------|
| hebrew_to_others | Hebrew | Aramaic, Arabic, Chinese, English |
| semitic_to_non_semitic | Hebrew, Aramaic, Arabic | Chinese, English |
| ancient_to_modern | Biblical, Classical | Modern texts |

### Success Criteria
- **PASS**: p < 0.05, accuracy > 1.5× chance
- **Strong PASS**: p < 0.001, accuracy > 2× chance

## Files

| File | Description |
|------|-------------|
| `BIP_v10.4_expanded.ipynb` | Main notebook with parallel downloads |
| `BIP_v10.4_README.md` | This documentation |
| `BIP_v10.3_expanded.ipynb` | Previous version (sequential) |
| `BIP_v10.2_Technical_Guide.md` | Detailed technical documentation |

## Data Sources & Licenses

| Source | License | Citation Required |
|--------|---------|-------------------|
| UN Parallel Corpus | UN Terms | Yes |
| UniMoral | CC-BY | Yes |
| Bible Corpus | Public Domain | Preferred |
| Hendrycks Ethics | MIT | Yes |
| Sefaria | CC-BY-NC | Yes |
| ctext.org | Terms of Use | Yes |
| Tanzil | Open | Yes |
| MIT Classics | Public Domain | No |

## Citation

```bibtex
@software{bip_v104,
  title = {Bond Invariance Principle: Parallel Multi-Corpus Moral Transfer},
  version = {10.4},
  year = {2025},
  url = {https://github.com/ahb-sjsu/sqnd-probe}
}
```

## Changelog

### v10.4
- Parallel threaded downloads (8 simultaneous)
- Added UN Parallel Corpus (HuggingFace streaming)
- Added UniMoral dataset (6 languages, labeled moral scenarios)
- Added Bible parallel corpus (100 languages)
- Added Hendrycks Ethics dataset (justice, virtue, duty)
- Thread-safe queue for passage collection

### v10.3
- Added ctext.org API for Classical Chinese
- Added MIT Classics for Western philosophy
- Added Tanzil.net fallback for Arabic

### v10.2
- Context-aware bond extraction
- Confidence weighting in training
- Prescriptive/descriptive detection
