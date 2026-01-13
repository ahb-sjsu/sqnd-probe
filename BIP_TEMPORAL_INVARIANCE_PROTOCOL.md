# BIP TEMPORAL INVARIANCE EXPERIMENT
## Complete Experimental Protocol v1.0

**Principal Investigator:** Andrew H. Bond  
**Date:** January 2026  
**Repository:** ahb-sjsu/sqnd-probe  

---

## ABSTRACT

This protocol tests the hypothesis that the **Bond Invariance Principle (BIP)** correctly predicts which aspects of moral reasoning are temporally invariant. We train models on ancient Hebrew ethical texts (Sefaria corpus, ~500 BCE to 1800 CE) and test transfer to modern American advice columns (Dear Abby, 1956-2020). If BIP is correct, bond-level features (agent-patient-relation-consent structure) should transfer across 2000 years with no significant accuracy drop, while label-level features (names, cultural framing, specific language) should not.

**Primary Hypothesis:** Moral judgments that depend on bond structure are temporally invariant.

**Statistical Power:** If correct, demonstrable at >10σ (conservative) to >50σ (combined).

---

## TABLE OF CONTENTS

1. [Environment Setup](#1-environment-setup)
2. [Data Acquisition](#2-data-acquisition)
3. [Data Preprocessing](#3-data-preprocessing)
4. [Bond Structure Extraction](#4-bond-structure-extraction)
5. [Corpus Split Generation](#5-corpus-split-generation)
6. [Model Architecture](#6-model-architecture)
7. [Training Procedure](#7-training-procedure)
8. [Evaluation Protocol](#8-evaluation-protocol)
9. [Statistical Analysis](#9-statistical-analysis)
10. [Expected Results](#10-expected-results)
11. [Failure Modes](#11-failure-modes)
12. [Execution Commands](#12-execution-commands)

---

## 1. ENVIRONMENT SETUP

### 1.1 Directory Structure

```bash
# Run from ahb-sjsu/sqnd-probe
mkdir -p data/{raw,processed,splits}
mkdir -p models/{checkpoints,final}
mkdir -p results/{metrics,figures,logs}
mkdir -p src/{data,models,evaluation,utils}
```

### 1.2 Dependencies

Create `requirements.txt`:

```
# Core
torch>=2.0.0
transformers>=4.35.0
datasets>=2.14.0
tokenizers>=0.14.0

# Data processing
numpy>=1.24.0
pandas>=2.0.0
scipy>=1.11.0
scikit-learn>=1.3.0

# NLP
spacy>=3.6.0
nltk>=3.8.0
sentence-transformers>=2.2.0

# Hebrew/Aramaic support
hebrew-tokenizer>=2.3.0
pyarabic>=0.6.0

# Visualization
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.15.0

# Experiment tracking
wandb>=0.15.0
tensorboard>=2.14.0

# Utilities
tqdm>=4.65.0
pyyaml>=6.0.0
jsonlines>=3.1.0
```

### 1.3 Installation Commands

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_lg

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('averaged_perceptron_tagger')"
```

### 1.4 Configuration File

Create `config.yaml`:

```yaml
# config.yaml
experiment:
  name: "bip_temporal_invariance_v1"
  seed: 42
  device: "cuda"  # or "cpu" or "mps"

data:
  sefaria_path: "data/raw/Sefaria-Export"
  dear_abby_path: "data/raw/dear_abby.csv"
  processed_path: "data/processed"
  max_sefaria_passages: null  # null = all
  max_abby_passages: null
  min_passage_length: 50  # characters
  max_passage_length: 2000

preprocessing:
  lowercase: false
  remove_punctuation: false
  normalize_unicode: true

bond_extraction:
  method: "hybrid"  # "rule", "llm", "hybrid"
  llm_model: "claude-3-haiku"  # for hybrid/llm
  confidence_threshold: 0.7

model:
  encoder: "sentence-transformers/all-mpnet-base-v2"
  d_model: 768
  d_bond: 128  # bond/invariant latent dimension
  d_label: 64  # label/temporal latent dimension
  n_hohfeld_classes: 4
  n_time_periods: 9
  dropout: 0.1

training:
  batch_size: 32
  learning_rate: 2e-5
  weight_decay: 0.01
  warmup_steps: 1000
  max_epochs: 50
  early_stopping_patience: 5
  gradient_clip: 1.0
  
  # Loss weights
  lambda_bip: 2.0        # BIP contrastive loss
  lambda_adversarial: 1.0  # Time prediction adversarial
  lambda_reconstruction: 0.1
  lambda_hohfeld: 1.0

evaluation:
  metrics:
    - "accuracy"
    - "f1_macro"
    - "transfer_gap"
    - "embedding_similarity"
    - "time_correlation"
  significance_level: 0.001  # Very conservative

logging:
  wandb_project: "bip-temporal-invariance"
  log_interval: 100
  save_interval: 1000
```

---

## 2. DATA ACQUISITION

### 2.1 Sefaria Corpus

```bash
# Clone Sefaria-Export (warning: ~8.3 GB)
cd data/raw
git clone --depth 1 https://github.com/Sefaria/Sefaria-Export.git

# Verify structure
ls Sefaria-Export/json/
# Should see: Tanakh, Mishnah, Talmud, Midrash, Halakhah, etc.
```

### 2.2 Dear Abby Corpus

Option A: Kaggle dataset
```bash
# If you have the Kaggle dataset
cp /path/to/dear_abby_dataset.csv data/raw/dear_abby.csv
```

Option B: Create collection script (if needed)
```python
# src/data/collect_dear_abby.py
"""
Script to collect Dear Abby data from available sources.
Modify based on your data source.
"""
```

### 2.3 Data Verification

Create `src/data/verify_data.py`:

```python
#!/usr/bin/env python3
"""Verify data acquisition was successful."""

import os
import json
from pathlib import Path
from collections import defaultdict

def verify_sefaria(path: str) -> dict:
    """Verify Sefaria corpus structure and count files."""
    sefaria_path = Path(path)
    
    if not sefaria_path.exists():
        return {"status": "ERROR", "message": f"Path not found: {path}"}
    
    json_path = sefaria_path / "json"
    if not json_path.exists():
        return {"status": "ERROR", "message": "json/ directory not found"}
    
    # Count files by category
    counts = defaultdict(int)
    total_size = 0
    
    for json_file in json_path.rglob("*.json"):
        category = json_file.relative_to(json_path).parts[0]
        counts[category] += 1
        total_size += json_file.stat().st_size
    
    return {
        "status": "OK",
        "categories": dict(counts),
        "total_files": sum(counts.values()),
        "total_size_mb": total_size / (1024 * 1024)
    }

def verify_dear_abby(path: str) -> dict:
    """Verify Dear Abby dataset."""
    abby_path = Path(path)
    
    if not abby_path.exists():
        return {"status": "ERROR", "message": f"Path not found: {path}"}
    
    if abby_path.suffix == '.csv':
        import pandas as pd
        df = pd.read_csv(abby_path, nrows=5)
        full_df = pd.read_csv(abby_path)
        return {
            "status": "OK",
            "columns": list(df.columns),
            "total_rows": len(full_df),
            "sample": df.head(2).to_dict()
        }
    elif abby_path.suffix == '.json':
        with open(abby_path) as f:
            data = json.load(f)
        return {
            "status": "OK",
            "type": "json",
            "total_items": len(data) if isinstance(data, list) else len(data.get('letters', []))
        }
    else:
        return {"status": "ERROR", "message": f"Unknown format: {abby_path.suffix}"}

if __name__ == "__main__":
    print("=" * 60)
    print("DATA VERIFICATION")
    print("=" * 60)
    
    print("\n[1] Sefaria Corpus:")
    sefaria_result = verify_sefaria("data/raw/Sefaria-Export")
    print(json.dumps(sefaria_result, indent=2))
    
    print("\n[2] Dear Abby Corpus:")
    abby_result = verify_dear_abby("data/raw/dear_abby.csv")
    print(json.dumps(abby_result, indent=2))
    
    print("\n" + "=" * 60)
    if sefaria_result["status"] == "OK" and abby_result["status"] == "OK":
        print("✓ All data sources verified successfully")
    else:
        print("✗ Data verification failed - check errors above")
```

---

## 3. DATA PREPROCESSING

Create `src/data/preprocess.py`:

```python
#!/usr/bin/env python3
"""
Preprocess Sefaria and Dear Abby corpora into unified format.
"""

import json
import hashlib
import re
import unicodedata
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Set, Optional, Tuple
from enum import Enum
from collections import defaultdict
from tqdm import tqdm
import yaml

# =============================================================================
# ENUMS AND DATA CLASSES
# =============================================================================

class TimePeriod(Enum):
    BIBLICAL = 0        # -1000 to -500 BCE
    SECOND_TEMPLE = 1   # -500 BCE to 70 CE
    TANNAITIC = 2       # 70-200 CE
    AMORAIC = 3         # 200-500 CE
    GEONIC = 4          # 500-1000 CE
    RISHONIM = 5        # 1000-1500 CE
    ACHRONIM = 6        # 1500-1800 CE
    MODERN_HEBREW = 7   # 1800-1950
    DEAR_ABBY = 8       # 1956-2020

class BondType(Enum):
    HARM_PREVENTION = 0
    RECIPROCITY = 1
    AUTONOMY = 2
    PROPERTY = 3
    FAMILY = 4
    AUTHORITY = 5
    EMERGENCY = 6
    CONTRACT = 7
    CARE = 8
    FAIRNESS = 9

class HohfeldianState(Enum):
    RIGHT = 0
    OBLIGATION = 1
    LIBERTY = 2
    NO_RIGHT = 3

class ConsentStatus(Enum):
    EXPLICIT_YES = 0
    IMPLICIT_YES = 1
    CONTESTED = 2
    IMPLICIT_NO = 3
    EXPLICIT_NO = 4
    IMPOSSIBLE = 5

@dataclass
class Passage:
    id: str
    text_original: str
    text_english: str
    time_period: str  # Store as string for JSON serialization
    century: int
    source: str
    source_type: str
    category: str
    language: str = "hebrew"
    word_count: int = 0
    has_dispute: bool = False
    consensus_tier: str = "unknown"
    bond_types: List[str] = field(default_factory=list)
    
    def to_dict(self):
        return asdict(self)

# =============================================================================
# SEFARIA LOADER
# =============================================================================

class SefariaLoader:
    """Load and preprocess Sefaria corpus."""
    
    CATEGORY_TO_PERIOD = {
        'Tanakh': TimePeriod.BIBLICAL,
        'Torah': TimePeriod.BIBLICAL,
        'Prophets': TimePeriod.BIBLICAL,
        'Writings': TimePeriod.BIBLICAL,
        'Mishnah': TimePeriod.TANNAITIC,
        'Tosefta': TimePeriod.TANNAITIC,
        'Talmud': TimePeriod.AMORAIC,
        'Bavli': TimePeriod.AMORAIC,
        'Yerushalmi': TimePeriod.AMORAIC,
        'Midrash': TimePeriod.AMORAIC,
        'Halakhah': TimePeriod.RISHONIM,
        'Kabbalah': TimePeriod.RISHONIM,
        'Musar': TimePeriod.ACHRONIM,
        'Responsa': TimePeriod.ACHRONIM,
    }
    
    PERIOD_TO_CENTURY = {
        TimePeriod.BIBLICAL: -6,
        TimePeriod.SECOND_TEMPLE: -2,
        TimePeriod.TANNAITIC: 2,
        TimePeriod.AMORAIC: 4,
        TimePeriod.GEONIC: 8,
        TimePeriod.RISHONIM: 12,
        TimePeriod.ACHRONIM: 17,
        TimePeriod.MODERN_HEBREW: 20,
    }
    
    # Prioritize these tractates (ethical content)
    PRIORITY_TRACTATES = {
        'Pirkei Avot', 'Avot',
        'Bava Kamma', 'Bava Metzia', 'Bava Batra',
        'Sanhedrin', 'Makkot',
        'Gittin', 'Kiddushin', 'Ketubot',
        'Shabbat', 'Yoma', 'Berakhot',
        'Nedarim', 'Nazir', 'Sotah',
    }
    
    def __init__(self, base_path: str, config: dict):
        self.base_path = Path(base_path)
        self.json_path = self.base_path / "json"
        self.config = config
        self.min_length = config.get('min_passage_length', 50)
        self.max_length = config.get('max_passage_length', 2000)
        
    def load(self, max_passages: int = None) -> List[Passage]:
        """Load all passages from Sefaria."""
        passages = []
        
        print("Loading Sefaria corpus...")
        
        # Priority 1: Talmud (richest ethical content)
        talmud_path = self.json_path / "Talmud" / "Bavli"
        if talmud_path.exists():
            for tractate_dir in tqdm(list(talmud_path.iterdir()), desc="Talmud"):
                passages.extend(self._load_tractate(tractate_dir))
        
        # Priority 2: Mishnah
        mishnah_path = self.json_path / "Mishnah"
        if mishnah_path.exists():
            for tractate_dir in tqdm(list(mishnah_path.iterdir()), desc="Mishnah"):
                passages.extend(self._load_tractate(tractate_dir))
        
        # Priority 3: Midrash
        midrash_path = self.json_path / "Midrash"
        if midrash_path.exists():
            for subdir in tqdm(list(midrash_path.iterdir()), desc="Midrash"):
                if subdir.is_dir():
                    for item in subdir.iterdir():
                        passages.extend(self._load_tractate(item))
        
        # Priority 4: Halakhah (legal literature)
        halakhah_path = self.json_path / "Halakhah"
        if halakhah_path.exists():
            for subdir in tqdm(list(halakhah_path.iterdir()), desc="Halakhah"):
                if subdir.is_dir():
                    for item in subdir.iterdir():
                        passages.extend(self._load_tractate(item))
        
        if max_passages:
            passages = passages[:max_passages]
        
        print(f"Loaded {len(passages)} passages from Sefaria")
        return passages
    
    def _load_tractate(self, path: Path) -> List[Passage]:
        """Load a tractate or text directory."""
        passages = []
        
        if path.is_file() and path.suffix == '.json':
            passages.extend(self._parse_json(path))
        elif path.is_dir():
            # Look for merged.json first
            merged = path / "merged.json"
            if merged.exists():
                passages.extend(self._parse_json(merged))
            else:
                # Load individual files
                for f in path.glob("*.json"):
                    passages.extend(self._parse_json(f))
        
        return passages
    
    def _parse_json(self, path: Path) -> List[Passage]:
        """Parse a Sefaria JSON file."""
        passages = []
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return []
        
        # Infer metadata from path
        rel_path = path.relative_to(self.json_path)
        category = str(rel_path.parts[0]) if rel_path.parts else "unknown"
        time_period = self._infer_period(category)
        century = self.PERIOD_TO_CENTURY.get(time_period, 0)
        
        # Extract text
        if isinstance(data, dict):
            hebrew = data.get('he', data.get('text', []))
            english = data.get('text', data.get('en', []))
            
            passages.extend(self._flatten(
                hebrew, english,
                source_base=path.stem,
                category=category,
                time_period=time_period,
                century=century
            ))
        
        return passages
    
    def _flatten(
        self, 
        hebrew, 
        english,
        source_base: str,
        category: str,
        time_period: TimePeriod,
        century: int,
        ref: str = ""
    ) -> List[Passage]:
        """Recursively flatten nested text structures."""
        passages = []
        
        if isinstance(hebrew, str) and isinstance(english, str):
            # Clean and validate
            hebrew = self._clean_text(hebrew)
            english = self._clean_text(english)
            
            if len(english) >= self.min_length and len(english) <= self.max_length:
                pid = hashlib.md5(f"{source_base}:{ref}:{hebrew[:50]}".encode()).hexdigest()[:12]
                
                passages.append(Passage(
                    id=f"sefaria_{pid}",
                    text_original=hebrew,
                    text_english=english,
                    time_period=time_period.name,
                    century=century,
                    source=f"{source_base} {ref}".strip(),
                    source_type="sefaria",
                    category=category,
                    language="hebrew",
                    word_count=len(english.split())
                ))
        
        elif isinstance(hebrew, list) and isinstance(english, list):
            for i, (h, e) in enumerate(zip(hebrew, english)):
                new_ref = f"{ref}.{i+1}" if ref else str(i+1)
                passages.extend(self._flatten(
                    h, e, source_base, category, time_period, century, new_ref
                ))
        
        elif isinstance(hebrew, list):
            for i, h in enumerate(hebrew):
                e = english[i] if isinstance(english, list) and i < len(english) else str(h)
                new_ref = f"{ref}.{i+1}" if ref else str(i+1)
                passages.extend(self._flatten(
                    h, e, source_base, category, time_period, century, new_ref
                ))
        
        return passages
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        if not isinstance(text, str):
            return ""
        
        # Normalize unicode
        text = unicodedata.normalize('NFC', text)
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Normalize whitespace
        text = ' '.join(text.split())
        
        return text.strip()
    
    def _infer_period(self, category: str) -> TimePeriod:
        """Infer time period from category."""
        for key, period in self.CATEGORY_TO_PERIOD.items():
            if key.lower() in category.lower():
                return period
        return TimePeriod.AMORAIC

# =============================================================================
# DEAR ABBY LOADER
# =============================================================================

class DearAbbyLoader:
    """Load and preprocess Dear Abby corpus."""
    
    CATEGORY_PATTERNS = {
        'family': r'\b(mother|father|parent|sibling|brother|sister|child|daughter|son|family|mom|dad)\b',
        'marriage': r'\b(husband|wife|spouse|married|wedding|divorce|marriage)\b',
        'relationship': r'\b(boyfriend|girlfriend|dating|relationship|partner|romance)\b',
        'work': r'\b(boss|coworker|job|work|office|employee|career|colleague)\b',
        'neighbor': r'\b(neighbor|neighbourhood|community)\b',
        'friendship': r'\b(friend|friendship|buddy|pal)\b',
        'money': r'\b(money|debt|loan|inherit|financial|pay|owe)\b',
        'health': r'\b(doctor|health|medical|illness|sick|disease)\b',
        'etiquette': r'\b(rude|polite|manner|etiquette|proper|thank)\b',
    }
    
    def __init__(self, path: str, config: dict):
        self.path = Path(path)
        self.config = config
        self.min_length = config.get('min_passage_length', 50)
        self.max_length = config.get('max_passage_length', 2000)
    
    def load(self, max_passages: int = None) -> List[Passage]:
        """Load Dear Abby passages."""
        print("Loading Dear Abby corpus...")
        
        if self.path.suffix == '.csv':
            passages = self._load_csv()
        elif self.path.suffix == '.json':
            passages = self._load_json()
        else:
            raise ValueError(f"Unknown format: {self.path.suffix}")
        
        if max_passages:
            passages = passages[:max_passages]
        
        print(f"Loaded {len(passages)} passages from Dear Abby")
        return passages
    
    def _load_csv(self) -> List[Passage]:
        """Load from CSV."""
        import pandas as pd
        
        df = pd.read_csv(self.path)
        passages = []
        
        # Try to identify columns
        date_col = self._find_column(df, ['date', 'Date', 'DATE'])
        question_col = self._find_column(df, ['question', 'Question', 'letter', 'Letter', 'text'])
        answer_col = self._find_column(df, ['answer', 'Answer', 'response', 'Response', 'reply'])
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Dear Abby"):
            date = str(row.get(date_col, '')) if date_col else ''
            question = str(row.get(question_col, ''))
            answer = str(row.get(answer_col, ''))
            
            if not question or not answer:
                continue
            
            # Combine Q&A
            full_text = f"QUESTION: {question}\n\nANSWER: {answer}"
            
            if len(full_text) < self.min_length or len(full_text) > self.max_length:
                continue
            
            year = self._extract_year(date)
            category = self._infer_category(full_text)
            
            pid = hashlib.md5(f"abby:{date}:{question[:50]}".encode()).hexdigest()[:12]
            
            passages.append(Passage(
                id=f"abby_{pid}",
                text_original=full_text,
                text_english=full_text,
                time_period=TimePeriod.DEAR_ABBY.name,
                century=20 if year < 2000 else 21,
                source=f"Dear Abby {date}",
                source_type="dear_abby",
                category=category,
                language="english",
                word_count=len(full_text.split())
            ))
        
        return passages
    
    def _load_json(self) -> List[Passage]:
        """Load from JSON."""
        with open(self.path) as f:
            data = json.load(f)
        
        items = data if isinstance(data, list) else data.get('letters', [])
        passages = []
        
        for item in tqdm(items, desc="Dear Abby"):
            date = item.get('date', '')
            question = item.get('question', item.get('letter', ''))
            answer = item.get('answer', item.get('response', ''))
            
            if not question or not answer:
                continue
            
            full_text = f"QUESTION: {question}\n\nANSWER: {answer}"
            
            if len(full_text) < self.min_length or len(full_text) > self.max_length:
                continue
            
            year = self._extract_year(date)
            category = self._infer_category(full_text)
            
            pid = hashlib.md5(f"abby:{date}:{question[:50]}".encode()).hexdigest()[:12]
            
            passages.append(Passage(
                id=f"abby_{pid}",
                text_original=full_text,
                text_english=full_text,
                time_period=TimePeriod.DEAR_ABBY.name,
                century=20 if year < 2000 else 21,
                source=f"Dear Abby {date}",
                source_type="dear_abby",
                category=category,
                language="english",
                word_count=len(full_text.split())
            ))
        
        return passages
    
    def _find_column(self, df, candidates: List[str]) -> Optional[str]:
        """Find a column from candidates."""
        for c in candidates:
            if c in df.columns:
                return c
        return None
    
    def _extract_year(self, date_str: str) -> int:
        """Extract year from date string."""
        match = re.search(r'(19|20)\d{2}', str(date_str))
        return int(match.group()) if match else 1990
    
    def _infer_category(self, text: str) -> str:
        """Infer category from content."""
        text_lower = text.lower()
        for category, pattern in self.CATEGORY_PATTERNS.items():
            if re.search(pattern, text_lower):
                return category
        return "general"

# =============================================================================
# MAIN PREPROCESSING PIPELINE
# =============================================================================

def preprocess_all(config_path: str = "config.yaml"):
    """Run full preprocessing pipeline."""
    
    # Load config
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    data_config = config.get('data', {})
    prep_config = config.get('preprocessing', {})
    
    # Load corpora
    sefaria_loader = SefariaLoader(
        data_config.get('sefaria_path', 'data/raw/Sefaria-Export'),
        data_config
    )
    abby_loader = DearAbbyLoader(
        data_config.get('dear_abby_path', 'data/raw/dear_abby.csv'),
        data_config
    )
    
    sefaria_passages = sefaria_loader.load(data_config.get('max_sefaria_passages'))
    abby_passages = abby_loader.load(data_config.get('max_abby_passages'))
    
    all_passages = sefaria_passages + abby_passages
    
    # Print statistics
    print("\n" + "=" * 60)
    print("CORPUS STATISTICS")
    print("=" * 60)
    
    by_period = defaultdict(int)
    by_source = defaultdict(int)
    
    for p in all_passages:
        by_period[p.time_period] += 1
        by_source[p.source_type] += 1
    
    print(f"\nTotal passages: {len(all_passages)}")
    print(f"\nBy source:")
    for source, count in sorted(by_source.items()):
        print(f"  {source}: {count:,}")
    
    print(f"\nBy time period:")
    for period, count in sorted(by_period.items()):
        print(f"  {period}: {count:,}")
    
    # Save
    output_path = Path(data_config.get('processed_path', 'data/processed'))
    output_path.mkdir(parents=True, exist_ok=True)
    
    passages_file = output_path / "passages.jsonl"
    with open(passages_file, 'w') as f:
        for p in all_passages:
            f.write(json.dumps(p.to_dict()) + '\n')
    
    print(f"\nSaved to {passages_file}")
    
    # Also save statistics
    stats = {
        'total_passages': len(all_passages),
        'by_period': dict(by_period),
        'by_source': dict(by_source),
        'sefaria_count': len(sefaria_passages),
        'abby_count': len(abby_passages)
    }
    
    with open(output_path / "corpus_stats.json", 'w') as f:
        json.dump(stats, f, indent=2)
    
    return all_passages

if __name__ == "__main__":
    preprocess_all()
```

---

## 4. BOND STRUCTURE EXTRACTION

Create `src/data/extract_bonds.py`:

```python
#!/usr/bin/env python3
"""
Extract bond structures from passages.

Bond = (agent, patient, relation, consent)

This is the key step that enables BIP testing.
"""

import json
import re
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Set, Optional, Tuple
from enum import Enum
from collections import defaultdict
from tqdm import tqdm
import yaml

# Reuse enums from preprocess.py
from preprocess import (
    BondType, HohfeldianState, ConsentStatus, 
    TimePeriod, Passage
)

# =============================================================================
# BOND DATA STRUCTURES
# =============================================================================

@dataclass
class Bond:
    """A single moral bond."""
    agent: str
    patient: str
    relation: str  # BondType name
    consent: str   # ConsentStatus name
    
    def to_tuple(self) -> tuple:
        return (self.agent, self.patient, self.relation, self.consent)
    
    def to_dict(self) -> dict:
        return asdict(self)

@dataclass
class BondStructure:
    """Complete bond structure of a passage."""
    bonds: List[dict]  # List of Bond.to_dict()
    primary_relation: str
    hohfeld_state: Optional[str] = None
    signature: str = ""  # Canonical representation for isomorphism
    
    def to_dict(self) -> dict:
        return asdict(self)

# =============================================================================
# PATTERN-BASED BOND EXTRACTOR
# =============================================================================

class BondExtractor:
    """
    Extract bond structures using pattern matching.
    
    This is the rule-based version. For production, combine with
    LLM-based extraction for ambiguous cases.
    """
    
    # Agent/patient role patterns (abstract roles, not specific names)
    ROLE_PATTERNS = {
        'SELF': [
            r'\b(I|me|my|myself|mine)\b',
            r'\bone\'s own\b',
        ],
        'OTHER': [
            r'\b(you|your|yours|yourself)\b',
            r'\b(they|them|their|theirs)\b',
            r'\b(he|him|his|she|her|hers)\b',
            r'\b(another|other|someone|anyone|person|people|individual)\b',
            r'\b(neighbor|neighbour|fellow)\b',
        ],
        'FAMILY': [
            r'\b(mother|father|parent|parents)\b',
            r'\b(son|daughter|child|children|kid|kids)\b',
            r'\b(brother|sister|sibling|siblings)\b',
            r'\b(husband|wife|spouse)\b',
            r'\b(grandfather|grandmother|grandparent)\b',
            r'\b(uncle|aunt|cousin|nephew|niece)\b',
            r'\b(family|relative|kin)\b',
        ],
        'AUTHORITY': [
            r'\b(rabbi|sage|teacher|master|rav)\b',
            r'\b(court|judge|beit din|sanhedrin)\b',
            r'\b(king|ruler|government|authority)\b',
            r'\b(God|Hashem|Lord|divine)\b',
            r'\b(boss|employer|supervisor)\b',
        ],
        'COMMUNITY': [
            r'\b(community|society|public|nation)\b',
            r'\b(Israel|people|congregation)\b',
            r'\b(group|organization|collective)\b',
        ],
        'VULNERABLE': [
            r'\b(orphan|widow|poor|needy)\b',
            r'\b(stranger|convert|ger)\b',
            r'\b(sick|ill|injured|dying)\b',
            r'\b(child|infant|minor)\b',
        ],
    }
    
    # Relation patterns
    RELATION_PATTERNS = {
        BondType.HARM_PREVENTION: [
            r'\b(kill|murder|slay|put to death)\b',
            r'\b(harm|hurt|injure|damage|wound)\b',
            r'\b(save|rescue|protect|preserve)\b',
            r'\b(life|death|danger|peril|risk)\b',
            r'\b(blood|violence|strike|smite)\b',
        ],
        BondType.RECIPROCITY: [
            r'\b(return|repay|restore|give back)\b',
            r'\b(owe|debt|obligation|due)\b',
            r'\b(equal|same|likewise|similarly)\b',
            r'\b(mutual|reciprocal|exchange)\b',
            r'\b(hateful|love|do unto)\b',
        ],
        BondType.AUTONOMY: [
            r'\b(choose|decision|free will|liberty)\b',
            r'\b(consent|agree|accept|refuse)\b',
            r'\b(force|coerce|compel|require)\b',
            r'\b(right|entitle|privilege)\b',
            r'\b(self-determination|agency)\b',
        ],
        BondType.PROPERTY: [
            r'\b(property|possession|belonging)\b',
            r'\b(own|ownership|owner)\b',
            r'\b(steal|theft|rob|take)\b',
            r'\b(buy|sell|trade|exchange)\b',
            r'\b(land|field|house|goods)\b',
            r'\b(lost|found|return)\b',
        ],
        BondType.FAMILY: [
            r'\b(honor|respect|revere)\b.*\b(parent|father|mother)\b',
            r'\b(marry|marriage|betroth|wed)\b',
            r'\b(divorce|separate|put away)\b',
            r'\b(inherit|inheritance|heir)\b',
            r'\b(support|provide|sustain)\b.*\b(family|child|wife)\b',
        ],
        BondType.AUTHORITY: [
            r'\b(obey|follow|heed|submit)\b',
            r'\b(command|decree|order|law)\b',
            r'\b(permit|allow|forbid|prohibit)\b',
            r'\b(judge|rule|decide|arbitrate)\b',
            r'\b(teach|instruct|guide)\b',
        ],
        BondType.EMERGENCY: [
            r'\b(emergency|urgent|immediate)\b',
            r'\b(life-threatening|mortal danger)\b',
            r'\b(pikuach nefesh|save a life)\b',
            r'\b(override|supersede|set aside)\b',
            r'\b(Sabbath|shabbat|shabbos)\b.*\b(save|life)\b',
        ],
        BondType.CONTRACT: [
            r'\b(promise|vow|oath|swear)\b',
            r'\b(agree|agreement|contract)\b',
            r'\b(commit|commitment|pledge)\b',
            r'\b(word|keep|break|fulfill)\b',
            r'\b(neder|shevua)\b',
        ],
        BondType.CARE: [
            r'\b(care|caring|tend|look after)\b',
            r'\b(help|assist|aid|support)\b',
            r'\b(feed|clothe|shelter|provide)\b',
            r'\b(visit|comfort|console)\b',
            r'\b(tzedakah|charity|give)\b',
        ],
        BondType.FAIRNESS: [
            r'\b(fair|just|justice|righteous)\b',
            r'\b(equal|equally|same|impartial)\b',
            r'\b(deserve|merit|earn|worthy)\b',
            r'\b(bias|favor|prejudice|partial)\b',
            r'\b(mishpat|din|tzedek)\b',
        ],
    }
    
    # Hohfeldian state patterns
    HOHFELD_PATTERNS = {
        HohfeldianState.OBLIGATION: [
            r'\b(must|shall|have to|need to)\b',
            r'\b(obligat|duty|bound|require)\b',
            r'\b(command|mitzvah|chayav)\b',
            r'\b(responsible|accountable)\b',
        ],
        HohfeldianState.RIGHT: [
            r'\b(right to|entitled|may claim)\b',
            r'\b(deserve|owed|due)\b',
            r'\b(demand|expect|require from)\b',
        ],
        HohfeldianState.LIBERTY: [
            r'\b(may|can|permitted|allowed)\b',
            r'\b(free to|liberty to|privilege)\b',
            r'\b(optional|voluntary|choice)\b',
            r'\b(mutar|reshut)\b',
        ],
        HohfeldianState.NO_RIGHT: [
            r'\b(cannot demand|no claim|not entitled)\b',
            r'\b(no right to|may not require)\b',
        ],
    }
    
    # Consent patterns
    CONSENT_PATTERNS = {
        ConsentStatus.EXPLICIT_YES: [
            r'\b(agree|agreed|consent|consented)\b',
            r'\b(accept|accepted|willing|willingly)\b',
            r'\b(yes|approve|approved)\b',
        ],
        ConsentStatus.EXPLICIT_NO: [
            r'\b(refuse|refused|reject|rejected)\b',
            r'\b(no|deny|denied|unwilling)\b',
            r'\b(against|oppose|opposed)\b',
        ],
        ConsentStatus.CONTESTED: [
            r'\b(dispute|disputed|disagree)\b',
            r'\b(debate|debated|argue|argued)\b',
            r'\b(machlok|machloket|controversy)\b',
            r'\b(some say|others say|one says)\b',
            r'\b(Rabbi \w+ says.*Rabbi \w+ says)\b',
        ],
        ConsentStatus.IMPOSSIBLE: [
            r'\b(future generation|unborn)\b',
            r'\b(animal|beast|creature)\b',
            r'\b(deceased|dead|departed)\b',
            r'\b(cannot consent|unable to agree)\b',
        ],
    }
    
    def __init__(self, config: dict = None):
        self.config = config or {}
        
        # Compile patterns
        self.role_compiled = {
            role: [re.compile(p, re.IGNORECASE) for p in patterns]
            for role, patterns in self.ROLE_PATTERNS.items()
        }
        self.relation_compiled = {
            rel: [re.compile(p, re.IGNORECASE) for p in patterns]
            for rel, patterns in self.RELATION_PATTERNS.items()
        }
        self.hohfeld_compiled = {
            state: [re.compile(p, re.IGNORECASE) for p in patterns]
            for state, patterns in self.HOHFELD_PATTERNS.items()
        }
        self.consent_compiled = {
            status: [re.compile(p, re.IGNORECASE) for p in patterns]
            for status, patterns in self.CONSENT_PATTERNS.items()
        }
    
    def extract(self, passage: Passage) -> BondStructure:
        """Extract bond structure from passage."""
        text = passage.text_english
        
        # Extract components
        roles = self._extract_roles(text)
        relations = self._extract_relations(text)
        hohfeld = self._extract_hohfeld(text)
        consent = self._extract_consent(text)
        
        # Build bonds
        bonds = self._build_bonds(roles, relations, consent)
        
        # Determine primary relation
        primary = relations[0].name if relations else BondType.CARE.name
        
        # Compute signature for isomorphism checking
        signature = self._compute_signature(bonds)
        
        return BondStructure(
            bonds=[b.to_dict() for b in bonds],
            primary_relation=primary,
            hohfeld_state=hohfeld.name if hohfeld else None,
            signature=signature
        )
    
    def _extract_roles(self, text: str) -> List[str]:
        """Extract moral roles from text."""
        roles = []
        for role, patterns in self.role_compiled.items():
            for pattern in patterns:
                if pattern.search(text):
                    roles.append(role)
                    break
        
        # Default if none found
        if not roles:
            roles = ['SELF', 'OTHER']
        
        return list(set(roles))
    
    def _extract_relations(self, text: str) -> List[BondType]:
        """Extract relation types from text."""
        relations = []
        relation_scores = {}
        
        for rel_type, patterns in self.relation_compiled.items():
            count = 0
            for pattern in patterns:
                count += len(pattern.findall(text))
            if count > 0:
                relation_scores[rel_type] = count
        
        # Sort by frequency
        relations = sorted(relation_scores.keys(), 
                          key=lambda r: relation_scores[r], 
                          reverse=True)
        
        if not relations:
            relations = [BondType.CARE]
        
        return relations
    
    def _extract_hohfeld(self, text: str) -> Optional[HohfeldianState]:
        """Extract Hohfeldian state."""
        state_scores = {}
        
        for state, patterns in self.hohfeld_compiled.items():
            count = 0
            for pattern in patterns:
                count += len(pattern.findall(text))
            if count > 0:
                state_scores[state] = count
        
        if state_scores:
            return max(state_scores.keys(), key=lambda s: state_scores[s])
        return None
    
    def _extract_consent(self, text: str) -> ConsentStatus:
        """Extract consent status."""
        for status, patterns in self.consent_compiled.items():
            for pattern in patterns:
                if pattern.search(text):
                    return status
        return ConsentStatus.IMPLICIT_YES
    
    def _build_bonds(
        self,
        roles: List[str],
        relations: List[BondType],
        consent: ConsentStatus
    ) -> List[Bond]:
        """Build bonds from extracted components."""
        bonds = []
        
        # Limit to top 3 relations to avoid explosion
        for rel in relations[:3]:
            for agent in roles:
                for patient in roles:
                    # Skip certain reflexive bonds
                    if agent == patient and rel not in [BondType.AUTONOMY, BondType.CARE]:
                        continue
                    
                    bonds.append(Bond(
                        agent=agent,
                        patient=patient,
                        relation=rel.name,
                        consent=consent.name
                    ))
        
        return bonds
    
    def _compute_signature(self, bonds: List[Bond]) -> str:
        """
        Compute a canonical signature for bond structure.
        
        This enables isomorphism checking: two passages with
        the same signature have isomorphic bond structures.
        """
        if not bonds:
            return "empty"
        
        # Canonicalize roles by first appearance
        role_map = {}
        role_idx = 0
        
        canonical = []
        for bond in sorted(bonds, key=lambda b: (b.relation, b.agent, b.patient)):
            if bond.agent not in role_map:
                role_map[bond.agent] = f"R{role_idx}"
                role_idx += 1
            if bond.patient not in role_map:
                role_map[bond.patient] = f"R{role_idx}"
                role_idx += 1
            
            canonical.append((
                role_map[bond.agent],
                role_map[bond.patient],
                bond.relation,
                bond.consent
            ))
        
        # Sort for consistency
        canonical.sort()
        
        # Create signature string
        sig_parts = [f"{a}-{p}:{r}/{c}" for a, p, r, c in canonical]
        return "|".join(sig_parts)

# =============================================================================
# CONSENSUS CLASSIFIER
# =============================================================================

class ConsensusClassifier:
    """Classify passages by consensus tier."""
    
    DISPUTE_PATTERNS = [
        re.compile(r'\bdispute[ds]?\b', re.IGNORECASE),
        re.compile(r'\bdisagree[ds]?\b', re.IGNORECASE),
        re.compile(r'\bmachlok', re.IGNORECASE),
        re.compile(r'\bdebate[ds]?\b', re.IGNORECASE),
        re.compile(r'Rabbi \w+ says', re.IGNORECASE),
        re.compile(r'\bsome say\b', re.IGNORECASE),
        re.compile(r'\bothers say\b', re.IGNORECASE),
        re.compile(r'\bone opinion\b', re.IGNORECASE),
        re.compile(r'\banother view\b', re.IGNORECASE),
    ]
    
    UNIVERSAL_PATTERNS = [
        re.compile(r'\beveryone agrees\b', re.IGNORECASE),
        re.compile(r'\bundisputed\b', re.IGNORECASE),
        re.compile(r'\buniversally\b', re.IGNORECASE),
        re.compile(r'\balways\b', re.IGNORECASE),
        re.compile(r'\bnever\b.*\b(kill|murder|steal)\b', re.IGNORECASE),
        re.compile(r'save[s]? a life', re.IGNORECASE),
        re.compile(r'pikuach nefesh', re.IGNORECASE),
        re.compile(r'love your neighbor', re.IGNORECASE),
        re.compile(r'golden rule', re.IGNORECASE),
    ]
    
    def classify(self, text: str) -> Tuple[str, bool]:
        """
        Classify consensus tier.
        
        Returns: (tier, has_dispute)
        """
        has_dispute = any(p.search(text) for p in self.DISPUTE_PATTERNS)
        has_universal = any(p.search(text) for p in self.UNIVERSAL_PATTERNS)
        
        if has_dispute and not has_universal:
            return "contested", True
        elif has_universal:
            return "universal", False
        else:
            return "high", False

# =============================================================================
# MAIN EXTRACTION PIPELINE
# =============================================================================

def extract_bonds_all(config_path: str = "config.yaml"):
    """Run bond extraction on all passages."""
    
    # Load config
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Load passages
    processed_path = Path(config['data']['processed_path'])
    passages_file = processed_path / "passages.jsonl"
    
    print(f"Loading passages from {passages_file}...")
    passages = []
    with open(passages_file) as f:
        for line in f:
            data = json.loads(line)
            passages.append(Passage(**data))
    
    print(f"Loaded {len(passages)} passages")
    
    # Initialize extractors
    bond_extractor = BondExtractor(config.get('bond_extraction', {}))
    consensus_classifier = ConsensusClassifier()
    
    # Extract bonds
    print("Extracting bond structures...")
    bond_structures = []
    
    for passage in tqdm(passages):
        # Extract bond structure
        bond_struct = bond_extractor.extract(passage)
        
        # Classify consensus
        tier, has_dispute = consensus_classifier.classify(passage.text_english)
        
        # Update passage
        passage.bond_types = list(set(b['relation'] for b in bond_struct.bonds))
        passage.consensus_tier = tier
        passage.has_dispute = has_dispute
        
        bond_structures.append({
            'passage_id': passage.id,
            'bond_structure': bond_struct.to_dict()
        })
    
    # Save updated passages
    print("Saving updated passages...")
    with open(passages_file, 'w') as f:
        for p in passages:
            f.write(json.dumps(p.to_dict()) + '\n')
    
    # Save bond structures separately
    bonds_file = processed_path / "bond_structures.jsonl"
    with open(bonds_file, 'w') as f:
        for bs in bond_structures:
            f.write(json.dumps(bs) + '\n')
    
    # Print statistics
    print("\n" + "=" * 60)
    print("BOND EXTRACTION STATISTICS")
    print("=" * 60)
    
    # Count by relation type
    relation_counts = defaultdict(int)
    for bs in bond_structures:
        for bond in bs['bond_structure']['bonds']:
            relation_counts[bond['relation']] += 1
    
    print("\nBonds by relation type:")
    for rel, count in sorted(relation_counts.items(), key=lambda x: -x[1]):
        print(f"  {rel}: {count:,}")
    
    # Count by consensus
    consensus_counts = defaultdict(int)
    for p in passages:
        consensus_counts[p.consensus_tier] += 1
    
    print("\nPassages by consensus tier:")
    for tier, count in sorted(consensus_counts.items()):
        print(f"  {tier}: {count:,}")
    
    # Count unique signatures (for isomorphism)
    signatures = set(bs['bond_structure']['signature'] for bs in bond_structures)
    print(f"\nUnique bond signatures: {len(signatures):,}")
    
    print(f"\nBond structures saved to {bonds_file}")
    
    return passages, bond_structures

if __name__ == "__main__":
    extract_bonds_all()
```

---

## 5. CORPUS SPLIT GENERATION

Create `src/data/generate_splits.py`:

```python
#!/usr/bin/env python3
"""
Generate train/valid/test splits for BIP temporal invariance testing.

Multiple split schemes, each testing different aspects of BIP.
"""

import json
import random
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Tuple, Set
from dataclasses import dataclass
import numpy as np
from tqdm import tqdm
import yaml

from preprocess import Passage, TimePeriod

# =============================================================================
# SPLIT GENERATORS
# =============================================================================

class SplitGenerator:
    """Generate multiple split schemes for comprehensive BIP testing."""
    
    def __init__(self, passages: List[Passage], seed: int = 42):
        self.passages = passages
        self.rng = random.Random(seed)
        self.np_rng = np.random.RandomState(seed)
        
        # Index passages
        self.by_period = self._index_by(lambda p: p.time_period)
        self.by_source = self._index_by(lambda p: p.source_type)
        self.by_consensus = self._index_by(lambda p: p.consensus_tier)
        
        # Load bond signatures for isomorphism matching
        self.signatures = {}
        self._load_signatures()
    
    def _index_by(self, key_fn) -> Dict:
        index = defaultdict(list)
        for p in self.passages:
            index[key_fn(p)].append(p)
        return dict(index)
    
    def _load_signatures(self):
        """Load bond signatures for isomorphism checking."""
        try:
            with open("data/processed/bond_structures.jsonl") as f:
                for line in f:
                    data = json.loads(line)
                    self.signatures[data['passage_id']] = data['bond_structure']['signature']
        except FileNotFoundError:
            print("Warning: bond_structures.jsonl not found")
    
    # =========================================================================
    # SPLIT 1: TEMPORAL HOLDOUT (Primary BIP Test)
    # =========================================================================
    
    def temporal_holdout(self) -> Dict:
        """
        Train: Ancient/Medieval (pre-1500 CE)
        Valid: Early Modern (1500-1900)
        Test: Modern (1900-2020)
        
        THIS IS THE PRIMARY BIP TEST.
        """
        # Define period groups
        train_periods = {
            'BIBLICAL', 'SECOND_TEMPLE', 'TANNAITIC', 
            'AMORAIC', 'GEONIC', 'RISHONIM'
        }
        valid_periods = {'ACHRONIM'}
        test_periods = {'MODERN_HEBREW', 'DEAR_ABBY'}
        
        train = [p for p in self.passages if p.time_period in train_periods]
        valid = [p for p in self.passages if p.time_period in valid_periods]
        test = [p for p in self.passages if p.time_period in test_periods]
        
        # Shuffle
        self.rng.shuffle(train)
        self.rng.shuffle(valid)
        self.rng.shuffle(test)
        
        return {
            'name': 'temporal_holdout',
            'description': 'Train on ancient/medieval, test on modern',
            'train_ids': [p.id for p in train],
            'valid_ids': [p.id for p in valid],
            'test_ids': [p.id for p in test],
            'train_size': len(train),
            'valid_size': len(valid),
            'test_size': len(test)
        }
    
    # =========================================================================
    # SPLIT 2: LEAVE-ONE-ERA-OUT
    # =========================================================================
    
    def leave_era_out(self) -> List[Dict]:
        """
        For each era, train on all others, test on that era.
        
        Tests transfer from every other era to each specific era.
        """
        folds = []
        
        for holdout in self.by_period.keys():
            if holdout not in self.by_period or len(self.by_period[holdout]) < 100:
                continue
            
            test = self.by_period[holdout]
            train_pool = [p for p in self.passages if p.time_period != holdout]
            
            self.rng.shuffle(train_pool)
            split = int(0.9 * len(train_pool))
            train = train_pool[:split]
            valid = train_pool[split:]
            
            folds.append({
                'name': f'leave_out_{holdout}',
                'description': f'Holdout era: {holdout}',
                'holdout_era': holdout,
                'train_ids': [p.id for p in train],
                'valid_ids': [p.id for p in valid],
                'test_ids': [p.id for p in test],
                'train_size': len(train),
                'valid_size': len(valid),
                'test_size': len(test)
            })
        
        return folds
    
    # =========================================================================
    # SPLIT 3: CROSS-CORPUS (Hebrew ↔ Dear Abby)
    # =========================================================================
    
    def cross_corpus(self) -> Dict:
        """
        Two experiments:
        A) Train Sefaria → Test Dear Abby
        B) Train Dear Abby → Test Sefaria
        """
        sefaria = self.by_source.get('sefaria', [])
        abby = self.by_source.get('dear_abby', [])
        
        self.rng.shuffle(sefaria)
        self.rng.shuffle(abby)
        
        sef_split = int(0.9 * len(sefaria))
        abby_split = int(0.9 * len(abby))
        
        return {
            'sefaria_to_abby': {
                'name': 'sefaria_to_abby',
                'description': 'Train on Hebrew corpus, test on Dear Abby',
                'train_ids': [p.id for p in sefaria[:sef_split]],
                'valid_ids': [p.id for p in sefaria[sef_split:]],
                'test_ids': [p.id for p in abby],
                'train_size': sef_split,
                'valid_size': len(sefaria) - sef_split,
                'test_size': len(abby)
            },
            'abby_to_sefaria': {
                'name': 'abby_to_sefaria',
                'description': 'Train on Dear Abby, test on Hebrew corpus',
                'train_ids': [p.id for p in abby[:abby_split]],
                'valid_ids': [p.id for p in abby[abby_split:]],
                'test_ids': [p.id for p in sefaria],
                'train_size': abby_split,
                'valid_size': len(abby) - abby_split,
                'test_size': len(sefaria)
            }
        }
    
    # =========================================================================
    # SPLIT 4: BOND-ISOMORPHIC PAIRS
    # =========================================================================
    
    def isomorphic_pairs(self) -> Dict:
        """
        Find pairs of passages with isomorphic bond structure
        from DIFFERENT time periods.
        
        This is the most direct BIP test.
        """
        print("Finding bond-isomorphic cross-temporal pairs...")
        
        # Group by signature
        by_signature = defaultdict(list)
        for p in self.passages:
            sig = self.signatures.get(p.id, "unknown")
            if sig != "unknown" and sig != "empty":
                by_signature[sig].append(p)
        
        # Find cross-temporal pairs within same signature
        pairs = []
        for sig, passages in tqdm(by_signature.items()):
            if len(passages) < 2:
                continue
            
            # Find pairs from different eras
            for i, p1 in enumerate(passages):
                for p2 in passages[i+1:]:
                    if p1.time_period != p2.time_period:
                        # Order by time (older, newer)
                        t1 = list(TimePeriod.__members__.keys()).index(p1.time_period)
                        t2 = list(TimePeriod.__members__.keys()).index(p2.time_period)
                        if t1 < t2:
                            pairs.append((p1.id, p2.id))
                        else:
                            pairs.append((p2.id, p1.id))
        
        print(f"Found {len(pairs)} isomorphic cross-temporal pairs")
        
        # Split pairs
        self.rng.shuffle(pairs)
        n = len(pairs)
        train_pairs = pairs[:int(0.7*n)]
        valid_pairs = pairs[int(0.7*n):int(0.85*n)]
        test_pairs = pairs[int(0.85*n):]
        
        return {
            'name': 'isomorphic_pairs',
            'description': 'Bond-isomorphic pairs across time periods',
            'train_pairs': train_pairs,
            'valid_pairs': valid_pairs,
            'test_pairs': test_pairs,
            'total_pairs': len(pairs),
            'unique_signatures': len(by_signature)
        }
    
    # =========================================================================
    # SPLIT 5: STRATIFIED RANDOM (Control)
    # =========================================================================
    
    def stratified_random(self, train_ratio=0.7, valid_ratio=0.15) -> Dict:
        """
        Standard stratified split as control condition.
        
        Stratify by: time_period, consensus_tier, source_type
        """
        # Create strata
        strata = defaultdict(list)
        for p in self.passages:
            key = (p.time_period, p.consensus_tier, p.source_type)
            strata[key].append(p)
        
        train, valid, test = [], [], []
        
        for key, passages in strata.items():
            self.rng.shuffle(passages)
            n = len(passages)
            n_train = int(train_ratio * n)
            n_valid = int(valid_ratio * n)
            
            train.extend(passages[:n_train])
            valid.extend(passages[n_train:n_train+n_valid])
            test.extend(passages[n_train+n_valid:])
        
        self.rng.shuffle(train)
        self.rng.shuffle(valid)
        self.rng.shuffle(test)
        
        return {
            'name': 'stratified_random',
            'description': 'Stratified random split (control condition)',
            'train_ids': [p.id for p in train],
            'valid_ids': [p.id for p in valid],
            'test_ids': [p.id for p in test],
            'train_size': len(train),
            'valid_size': len(valid),
            'test_size': len(test)
        }
    
    # =========================================================================
    # GENERATE ALL SPLITS
    # =========================================================================
    
    def generate_all(self) -> Dict:
        """Generate all split schemes."""
        print("\n" + "=" * 60)
        print("GENERATING SPLIT SCHEMES")
        print("=" * 60)
        
        splits = {}
        
        print("\n[1/5] Temporal holdout...")
        splits['temporal_holdout'] = self.temporal_holdout()
        print(f"  Train: {splits['temporal_holdout']['train_size']:,}")
        print(f"  Valid: {splits['temporal_holdout']['valid_size']:,}")
        print(f"  Test: {splits['temporal_holdout']['test_size']:,}")
        
        print("\n[2/5] Leave-one-era-out...")
        splits['leave_era_out'] = self.leave_era_out()
        print(f"  Generated {len(splits['leave_era_out'])} folds")
        
        print("\n[3/5] Cross-corpus...")
        splits['cross_corpus'] = self.cross_corpus()
        print(f"  Sefaria→Abby: {splits['cross_corpus']['sefaria_to_abby']['test_size']:,} test")
        print(f"  Abby→Sefaria: {splits['cross_corpus']['abby_to_sefaria']['test_size']:,} test")
        
        print("\n[4/5] Isomorphic pairs...")
        splits['isomorphic_pairs'] = self.isomorphic_pairs()
        print(f"  Total pairs: {splits['isomorphic_pairs']['total_pairs']:,}")
        
        print("\n[5/5] Stratified random (control)...")
        splits['stratified_random'] = self.stratified_random()
        print(f"  Train: {splits['stratified_random']['train_size']:,}")
        
        return splits

# =============================================================================
# MAIN
# =============================================================================

def generate_splits(config_path: str = "config.yaml"):
    """Generate all splits and save."""
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Load passages
    processed_path = Path(config['data']['processed_path'])
    passages_file = processed_path / "passages.jsonl"
    
    print(f"Loading passages from {passages_file}...")
    passages = []
    with open(passages_file) as f:
        for line in f:
            data = json.loads(line)
            passages.append(Passage(**data))
    print(f"Loaded {len(passages)} passages")
    
    # Generate splits
    generator = SplitGenerator(passages, config['experiment']['seed'])
    splits = generator.generate_all()
    
    # Save
    splits_dir = Path("data/splits")
    splits_dir.mkdir(parents=True, exist_ok=True)
    
    splits_file = splits_dir / "all_splits.json"
    with open(splits_file, 'w') as f:
        json.dump(splits, f, indent=2)
    
    print(f"\nSplits saved to {splits_file}")
    
    return splits

if __name__ == "__main__":
    generate_splits()
```

---

## 6. MODEL ARCHITECTURE

Create `src/models/bip_model.py`:

```python
#!/usr/bin/env python3
"""
BIP Temporal Invariance Model

Architecture:
- Encoder: Pretrained transformer (freezable)
- Disentangler: Splits representation into z_bond (invariant) and z_label (temporal)
- Adversarial time classifier: Tries to predict era from z_bond (we want it to fail)
- Hohfeldian classifier: Predicts moral state from z_bond
- Reconstruction decoder: Optional, ensures information preservation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from typing import Dict, Optional, Tuple
import math

# =============================================================================
# MODEL COMPONENTS
# =============================================================================

class GradientReversal(torch.autograd.Function):
    """
    Gradient reversal layer for adversarial training.
    
    Forward: identity
    Backward: negate gradient and scale by lambda
    """
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()
    
    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None

def gradient_reversal(x, lambda_=1.0):
    return GradientReversal.apply(x, lambda_)


class BIPEncoder(nn.Module):
    """
    Encode passages into latent space.
    
    Uses pretrained transformer + projection layers.
    """
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-mpnet-base-v2",
        d_model: int = 768,
        freeze_encoder: bool = False
    ):
        super().__init__()
        
        self.encoder = AutoModel.from_pretrained(model_name)
        self.d_model = d_model
        
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        
        # Get encoder hidden size
        self.encoder_dim = self.encoder.config.hidden_size
        
        # Projection if dimensions don't match
        if self.encoder_dim != d_model:
            self.projection = nn.Linear(self.encoder_dim, d_model)
        else:
            self.projection = nn.Identity()
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Encode passages.
        
        Returns: [batch, d_model] pooled representation
        """
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        
        # Mean pooling
        hidden = outputs.last_hidden_state  # [batch, seq, hidden]
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        
        return self.projection(pooled)


class BIPDisentangler(nn.Module):
    """
    Disentangle representation into:
    - z_bond: Bond-level features (should be temporally invariant)
    - z_label: Label-level features (should capture temporal variation)
    
    Uses variational approach for regularization.
    """
    
    def __init__(
        self,
        d_model: int = 768,
        d_bond: int = 128,
        d_label: int = 64,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.d_model = d_model
        self.d_bond = d_bond
        self.d_label = d_label
        
        # Bond space projection (variational)
        self.bond_mean = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_bond)
        )
        self.bond_logvar = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_bond)
        )
        
        # Label space projection (variational)
        self.label_mean = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_label)
        )
        self.label_logvar = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_label)
        )
    
    def reparameterize(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick for VAE."""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mean + eps * std
        else:
            return mean
    
    def forward(self, h: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Disentangle representation.
        
        Args:
            h: [batch, d_model] encoder output
            
        Returns:
            dict with z_bond, z_label, and VAE parameters
        """
        # Bond space
        bond_mean = self.bond_mean(h)
        bond_logvar = self.bond_logvar(h)
        z_bond = self.reparameterize(bond_mean, bond_logvar)
        
        # Label space
        label_mean = self.label_mean(h)
        label_logvar = self.label_logvar(h)
        z_label = self.reparameterize(label_mean, label_logvar)
        
        return {
            'z_bond': z_bond,
            'z_label': z_label,
            'bond_mean': bond_mean,
            'bond_logvar': bond_logvar,
            'label_mean': label_mean,
            'label_logvar': label_logvar
        }


class TimeClassifier(nn.Module):
    """
    Adversarial time period classifier.
    
    Used to ensure z_bond is time-invariant.
    We want this to FAIL when given z_bond.
    """
    
    def __init__(self, d_input: int, n_periods: int = 9, hidden_dim: int = 128):
        super().__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(d_input, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, n_periods)
        )
    
    def forward(self, z: torch.Tensor, reverse_grad: bool = False, lambda_: float = 1.0) -> torch.Tensor:
        """
        Predict time period.
        
        Args:
            z: latent representation
            reverse_grad: if True, reverse gradients (adversarial)
            lambda_: gradient reversal scale
        """
        if reverse_grad:
            z = gradient_reversal(z, lambda_)
        return self.classifier(z)


class HohfeldianClassifier(nn.Module):
    """
    Classify passages into Hohfeldian states.
    
    States: Right (0), Obligation (1), Liberty (2), No-Right (3)
    """
    
    def __init__(self, d_input: int, n_classes: int = 4, hidden_dim: int = 64):
        super().__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(d_input, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, n_classes)
        )
    
    def forward(self, z_bond: torch.Tensor) -> torch.Tensor:
        return self.classifier(z_bond)


class BondClassifier(nn.Module):
    """
    Classify primary bond type.
    
    Multi-label classification for bond types present.
    """
    
    def __init__(self, d_input: int, n_bond_types: int = 10, hidden_dim: int = 64):
        super().__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(d_input, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, n_bond_types)
        )
    
    def forward(self, z_bond: torch.Tensor) -> torch.Tensor:
        return self.classifier(z_bond)  # Apply sigmoid for multi-label


# =============================================================================
# FULL MODEL
# =============================================================================

class BIPTemporalInvarianceModel(nn.Module):
    """
    Complete model for BIP temporal invariance testing.
    """
    
    def __init__(self, config: dict):
        super().__init__()
        
        self.config = config
        
        # Components
        self.encoder = BIPEncoder(
            model_name=config.get('encoder', 'sentence-transformers/all-mpnet-base-v2'),
            d_model=config.get('d_model', 768),
            freeze_encoder=config.get('freeze_encoder', False)
        )
        
        self.disentangler = BIPDisentangler(
            d_model=config.get('d_model', 768),
            d_bond=config.get('d_bond', 128),
            d_label=config.get('d_label', 64),
            dropout=config.get('dropout', 0.1)
        )
        
        self.time_classifier = TimeClassifier(
            d_input=config.get('d_bond', 128),
            n_periods=config.get('n_time_periods', 9)
        )
        
        self.time_classifier_label = TimeClassifier(
            d_input=config.get('d_label', 64),
            n_periods=config.get('n_time_periods', 9)
        )
        
        self.hohfeld_classifier = HohfeldianClassifier(
            d_input=config.get('d_bond', 128),
            n_classes=config.get('n_hohfeld_classes', 4)
        )
        
        self.bond_classifier = BondClassifier(
            d_input=config.get('d_bond', 128),
            n_bond_types=10
        )
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        time_labels: Optional[torch.Tensor] = None,
        adversarial_lambda: float = 1.0
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Returns dict with all predictions and latent representations.
        """
        # Encode
        h = self.encoder(input_ids, attention_mask)
        
        # Disentangle
        disentangled = self.disentangler(h)
        z_bond = disentangled['z_bond']
        z_label = disentangled['z_label']
        
        # Predictions
        # Time from z_bond (adversarial - we want this to fail)
        time_pred_bond = self.time_classifier(z_bond, reverse_grad=True, lambda_=adversarial_lambda)
        
        # Time from z_label (should succeed)
        time_pred_label = self.time_classifier_label(z_label)
        
        # Hohfeldian state from z_bond
        hohfeld_pred = self.hohfeld_classifier(z_bond)
        
        # Bond types from z_bond
        bond_pred = self.bond_classifier(z_bond)
        
        return {
            'z_bond': z_bond,
            'z_label': z_label,
            'bond_mean': disentangled['bond_mean'],
            'bond_logvar': disentangled['bond_logvar'],
            'label_mean': disentangled['label_mean'],
            'label_logvar': disentangled['label_logvar'],
            'time_pred_bond': time_pred_bond,
            'time_pred_label': time_pred_label,
            'hohfeld_pred': hohfeld_pred,
            'bond_pred': bond_pred
        }
    
    def encode_bond(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Get bond embedding only (for evaluation)."""
        h = self.encoder(input_ids, attention_mask)
        disentangled = self.disentangler(h)
        return disentangled['z_bond']


# =============================================================================
# LOSS FUNCTIONS
# =============================================================================

class BIPLoss(nn.Module):
    """
    Combined loss for BIP training.
    
    Components:
    1. Adversarial time loss (maximize entropy of time prediction from z_bond)
    2. Time prediction loss (z_label should predict time well)
    3. KL divergence (regularization)
    4. Hohfeldian classification loss
    5. BIP contrastive loss (bond-isomorphic pairs should have similar z_bond)
    """
    
    def __init__(self, config: dict):
        super().__init__()
        
        self.lambda_adv = config.get('lambda_adversarial', 1.0)
        self.lambda_kl = config.get('lambda_kl', 0.1)
        self.lambda_hohfeld = config.get('lambda_hohfeld', 1.0)
        self.lambda_bip = config.get('lambda_bip', 2.0)
        
        self.ce_loss = nn.CrossEntropyLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
    
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        time_labels: torch.Tensor,
        hohfeld_labels: Optional[torch.Tensor] = None,
        bond_labels: Optional[torch.Tensor] = None,
        isomorphic_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute total loss.
        
        Args:
            outputs: model outputs
            time_labels: [batch] time period indices
            hohfeld_labels: [batch] Hohfeldian state indices (optional)
            bond_labels: [batch, n_bonds] multi-hot bond type labels (optional)
            isomorphic_mask: [batch, batch] 1 if pair is bond-isomorphic (optional)
        """
        losses = {}
        
        # 1. Adversarial time loss on z_bond
        # We want maximum entropy = classifier is confused
        time_probs = F.softmax(outputs['time_pred_bond'], dim=-1)
        entropy = -torch.sum(time_probs * torch.log(time_probs + 1e-8), dim=-1)
        # Maximize entropy = minimize negative entropy
        losses['adv'] = -entropy.mean() * self.lambda_adv
        
        # 2. Time prediction loss on z_label (should succeed)
        losses['time'] = self.ce_loss(outputs['time_pred_label'], time_labels)
        
        # 3. KL divergence regularization
        kl_bond = -0.5 * torch.mean(
            1 + outputs['bond_logvar'] - outputs['bond_mean'].pow(2) - outputs['bond_logvar'].exp()
        )
        kl_label = -0.5 * torch.mean(
            1 + outputs['label_logvar'] - outputs['label_mean'].pow(2) - outputs['label_logvar'].exp()
        )
        losses['kl'] = (kl_bond + kl_label) * self.lambda_kl
        
        # 4. Hohfeldian classification loss
        if hohfeld_labels is not None:
            losses['hohfeld'] = self.ce_loss(outputs['hohfeld_pred'], hohfeld_labels) * self.lambda_hohfeld
        else:
            losses['hohfeld'] = torch.tensor(0.0, device=outputs['z_bond'].device)
        
        # 5. Bond type classification loss
        if bond_labels is not None:
            losses['bond'] = self.bce_loss(outputs['bond_pred'], bond_labels.float())
        else:
            losses['bond'] = torch.tensor(0.0, device=outputs['z_bond'].device)
        
        # 6. BIP contrastive loss (if isomorphic pairs provided)
        if isomorphic_mask is not None:
            z_bond = outputs['z_bond']
            # Normalize for cosine similarity
            z_norm = F.normalize(z_bond, dim=-1)
            sim_matrix = torch.mm(z_norm, z_norm.t())
            
            # Positive pairs: high similarity
            # Negative pairs: low similarity
            pos_loss = -torch.log(torch.exp(sim_matrix / 0.07) * isomorphic_mask + 1e-8).mean()
            losses['bip'] = pos_loss * self.lambda_bip
        else:
            losses['bip'] = torch.tensor(0.0, device=outputs['z_bond'].device)
        
        # Total loss
        total = sum(losses.values())
        
        # Convert to floats for logging
        loss_dict = {k: v.item() for k, v in losses.items()}
        loss_dict['total'] = total.item()
        
        return total, loss_dict


# =============================================================================
# TOKENIZER WRAPPER
# =============================================================================

def get_tokenizer(model_name: str = "sentence-transformers/all-mpnet-base-v2"):
    """Get tokenizer for the model."""
    return AutoTokenizer.from_pretrained(model_name)


if __name__ == "__main__":
    # Quick test
    config = {
        'encoder': 'sentence-transformers/all-mpnet-base-v2',
        'd_model': 768,
        'd_bond': 128,
        'd_label': 64,
        'n_time_periods': 9,
        'n_hohfeld_classes': 4
    }
    
    model = BIPTemporalInvarianceModel(config)
    tokenizer = get_tokenizer()
    
    # Test input
    texts = ["This is a test passage about saving a life.", 
             "Another passage about obligations."]
    encoded = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
    
    # Forward pass
    with torch.no_grad():
        outputs = model(encoded['input_ids'], encoded['attention_mask'])
    
    print("Model test passed!")
    print(f"z_bond shape: {outputs['z_bond'].shape}")
    print(f"z_label shape: {outputs['z_label'].shape}")
```

---

## 7. TRAINING PROCEDURE

Create `src/train.py`:

```python
#!/usr/bin/env python3
"""
Training script for BIP temporal invariance model.
"""

import os
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import yaml
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from transformers import AutoTokenizer

# Conditional imports
try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

from models.bip_model import BIPTemporalInvarianceModel, BIPLoss, get_tokenizer

# =============================================================================
# DATASET
# =============================================================================

class MoralPassageDataset(Dataset):
    """Dataset for moral passages."""
    
    # Map time periods to indices
    PERIOD_TO_IDX = {
        'BIBLICAL': 0,
        'SECOND_TEMPLE': 1,
        'TANNAITIC': 2,
        'AMORAIC': 3,
        'GEONIC': 4,
        'RISHONIM': 5,
        'ACHRONIM': 6,
        'MODERN_HEBREW': 7,
        'DEAR_ABBY': 8
    }
    
    HOHFELD_TO_IDX = {
        'RIGHT': 0,
        'OBLIGATION': 1,
        'LIBERTY': 2,
        'NO_RIGHT': 3
    }
    
    BOND_TYPES = [
        'HARM_PREVENTION', 'RECIPROCITY', 'AUTONOMY', 'PROPERTY',
        'FAMILY', 'AUTHORITY', 'EMERGENCY', 'CONTRACT', 'CARE', 'FAIRNESS'
    ]
    
    def __init__(
        self,
        passage_ids: List[str],
        passages_file: str,
        bonds_file: str,
        tokenizer,
        max_length: int = 512
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load passages
        self.passages = {}
        with open(passages_file) as f:
            for line in f:
                p = json.loads(line)
                if p['id'] in passage_ids or not passage_ids:
                    self.passages[p['id']] = p
        
        # Load bond structures
        self.bonds = {}
        if bonds_file and Path(bonds_file).exists():
            with open(bonds_file) as f:
                for line in f:
                    b = json.loads(line)
                    self.bonds[b['passage_id']] = b['bond_structure']
        
        # Filter to requested IDs
        if passage_ids:
            self.ids = [pid for pid in passage_ids if pid in self.passages]
        else:
            self.ids = list(self.passages.keys())
    
    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, idx):
        pid = self.ids[idx]
        passage = self.passages[pid]
        
        # Tokenize
        text = passage['text_english']
        encoded = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Time period label
        time_label = self.PERIOD_TO_IDX.get(passage['time_period'], 0)
        
        # Hohfeldian label (if available)
        bond_struct = self.bonds.get(pid, {})
        hohfeld_str = bond_struct.get('hohfeld_state')
        hohfeld_label = self.HOHFELD_TO_IDX.get(hohfeld_str, 0) if hohfeld_str else 0
        
        # Bond type labels (multi-hot)
        bond_labels = torch.zeros(len(self.BOND_TYPES))
        for bt in passage.get('bond_types', []):
            if bt in self.BOND_TYPES:
                bond_labels[self.BOND_TYPES.index(bt)] = 1
        
        return {
            'input_ids': encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0),
            'time_label': time_label,
            'hohfeld_label': hohfeld_label,
            'bond_labels': bond_labels,
            'passage_id': pid
        }


def collate_fn(batch):
    """Collate batch of samples."""
    return {
        'input_ids': torch.stack([b['input_ids'] for b in batch]),
        'attention_mask': torch.stack([b['attention_mask'] for b in batch]),
        'time_labels': torch.tensor([b['time_label'] for b in batch]),
        'hohfeld_labels': torch.tensor([b['hohfeld_label'] for b in batch]),
        'bond_labels': torch.stack([b['bond_labels'] for b in batch]),
        'passage_ids': [b['passage_id'] for b in batch]
    }


# =============================================================================
# TRAINER
# =============================================================================

class BIPTrainer:
    """Trainer for BIP model."""
    
    def __init__(self, config: dict, split_name: str = "temporal_holdout"):
        self.config = config
        self.split_name = split_name
        self.device = torch.device(config['experiment'].get('device', 'cuda'))
        
        # Create output directories
        self.output_dir = Path(f"models/checkpoints/{split_name}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize model
        self.model = BIPTemporalInvarianceModel(config['model']).to(self.device)
        self.tokenizer = get_tokenizer(config['model']['encoder'])
        
        # Loss
        self.criterion = BIPLoss(config['training'])
        
        # Load splits
        with open("data/splits/all_splits.json") as f:
            all_splits = json.load(f)
        
        if split_name in all_splits:
            split = all_splits[split_name]
        else:
            raise ValueError(f"Unknown split: {split_name}")
        
        # Create datasets
        passages_file = f"{config['data']['processed_path']}/passages.jsonl"
        bonds_file = f"{config['data']['processed_path']}/bond_structures.jsonl"
        
        self.train_dataset = MoralPassageDataset(
            split['train_ids'], passages_file, bonds_file, 
            self.tokenizer, config['model'].get('max_length', 512)
        )
        self.valid_dataset = MoralPassageDataset(
            split['valid_ids'], passages_file, bonds_file,
            self.tokenizer, config['model'].get('max_length', 512)
        )
        self.test_dataset = MoralPassageDataset(
            split['test_ids'], passages_file, bonds_file,
            self.tokenizer, config['model'].get('max_length', 512)
        )
        
        # Dataloaders
        batch_size = config['training']['batch_size']
        self.train_loader = DataLoader(
            self.train_dataset, batch_size=batch_size, shuffle=True, 
            collate_fn=collate_fn, num_workers=4
        )
        self.valid_loader = DataLoader(
            self.valid_dataset, batch_size=batch_size, shuffle=False,
            collate_fn=collate_fn, num_workers=4
        )
        self.test_loader = DataLoader(
            self.test_dataset, batch_size=batch_size, shuffle=False,
            collate_fn=collate_fn, num_workers=4
        )
        
        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
        
        # Scheduler
        total_steps = len(self.train_loader) * config['training']['max_epochs']
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=config['training']['learning_rate'],
            total_steps=total_steps,
            pct_start=0.1
        )
        
        # Tracking
        self.best_valid_loss = float('inf')
        self.patience_counter = 0
        self.global_step = 0
        
        # Wandb
        if HAS_WANDB and config['logging'].get('wandb_project'):
            wandb.init(
                project=config['logging']['wandb_project'],
                name=f"{split_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                config=config
            )
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0
        loss_components = {k: 0 for k in ['adv', 'time', 'kl', 'hohfeld', 'bond', 'bip']}
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        for batch in pbar:
            # Move to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            time_labels = batch['time_labels'].to(self.device)
            hohfeld_labels = batch['hohfeld_labels'].to(self.device)
            bond_labels = batch['bond_labels'].to(self.device)
            
            # Forward
            outputs = self.model(
                input_ids, attention_mask,
                time_labels=time_labels,
                adversarial_lambda=self.config['training']['lambda_adversarial']
            )
            
            # Loss
            loss, loss_dict = self.criterion(
                outputs, time_labels, hohfeld_labels, bond_labels
            )
            
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                self.config['training']['gradient_clip']
            )
            
            self.optimizer.step()
            self.scheduler.step()
            
            # Track
            total_loss += loss.item()
            for k, v in loss_dict.items():
                if k in loss_components:
                    loss_components[k] += v
            
            self.global_step += 1
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
            # Log
            if HAS_WANDB and self.global_step % self.config['logging']['log_interval'] == 0:
                wandb.log({
                    'train/loss': loss.item(),
                    'train/lr': self.scheduler.get_last_lr()[0],
                    **{f'train/{k}': v for k, v in loss_dict.items()}
                }, step=self.global_step)
        
        n_batches = len(self.train_loader)
        return {
            'loss': total_loss / n_batches,
            **{k: v / n_batches for k, v in loss_components.items()}
        }
    
    @torch.no_grad()
    def evaluate(self, loader: DataLoader, prefix: str = "valid") -> Dict[str, float]:
        """Evaluate on a dataset."""
        self.model.eval()
        
        total_loss = 0
        all_time_preds = []
        all_time_labels = []
        all_hohfeld_preds = []
        all_hohfeld_labels = []
        all_z_bonds = []
        
        for batch in tqdm(loader, desc=f"Evaluating {prefix}"):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            time_labels = batch['time_labels'].to(self.device)
            hohfeld_labels = batch['hohfeld_labels'].to(self.device)
            bond_labels = batch['bond_labels'].to(self.device)
            
            outputs = self.model(input_ids, attention_mask, adversarial_lambda=0.0)
            
            loss, _ = self.criterion(outputs, time_labels, hohfeld_labels, bond_labels)
            total_loss += loss.item()
            
            # Collect predictions
            all_time_preds.append(outputs['time_pred_bond'].argmax(dim=-1).cpu())
            all_time_labels.append(time_labels.cpu())
            all_hohfeld_preds.append(outputs['hohfeld_pred'].argmax(dim=-1).cpu())
            all_hohfeld_labels.append(hohfeld_labels.cpu())
            all_z_bonds.append(outputs['z_bond'].cpu())
        
        # Concatenate
        time_preds = torch.cat(all_time_preds)
        time_labels = torch.cat(all_time_labels)
        hohfeld_preds = torch.cat(all_hohfeld_preds)
        hohfeld_labels = torch.cat(all_hohfeld_labels)
        z_bonds = torch.cat(all_z_bonds)
        
        # Metrics
        n_batches = len(loader)
        
        # Time prediction accuracy (from z_bond - should be low!)
        time_acc = (time_preds == time_labels).float().mean().item()
        
        # Hohfeldian accuracy
        hohfeld_acc = (hohfeld_preds == hohfeld_labels).float().mean().item()
        
        metrics = {
            f'{prefix}/loss': total_loss / n_batches,
            f'{prefix}/time_acc_from_bond': time_acc,
            f'{prefix}/hohfeld_acc': hohfeld_acc,
        }
        
        return metrics, z_bonds, time_labels
    
    def train(self):
        """Full training loop."""
        print(f"\nTraining on split: {self.split_name}")
        print(f"Train: {len(self.train_dataset)}, Valid: {len(self.valid_dataset)}, Test: {len(self.test_dataset)}")
        
        max_epochs = self.config['training']['max_epochs']
        patience = self.config['training']['early_stopping_patience']
        
        for epoch in range(1, max_epochs + 1):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch}/{max_epochs}")
            print('='*60)
            
            # Train
            train_metrics = self.train_epoch(epoch)
            print(f"Train loss: {train_metrics['loss']:.4f}")
            
            # Validate
            valid_metrics, _, _ = self.evaluate(self.valid_loader, "valid")
            print(f"Valid loss: {valid_metrics['valid/loss']:.4f}")
            print(f"Valid time_acc_from_bond: {valid_metrics['valid/time_acc_from_bond']:.4f}")
            print(f"Valid hohfeld_acc: {valid_metrics['valid/hohfeld_acc']:.4f}")
            
            # Log
            if HAS_WANDB:
                wandb.log({
                    'epoch': epoch,
                    'train/epoch_loss': train_metrics['loss'],
                    **valid_metrics
                }, step=self.global_step)
            
            # Early stopping
            if valid_metrics['valid/loss'] < self.best_valid_loss:
                self.best_valid_loss = valid_metrics['valid/loss']
                self.patience_counter = 0
                
                # Save best model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_valid_loss': self.best_valid_loss,
                    'config': self.config
                }, self.output_dir / "best_model.pt")
                print("✓ Saved best model")
            else:
                self.patience_counter += 1
                if self.patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
        
        # Final test evaluation
        print("\n" + "="*60)
        print("FINAL TEST EVALUATION")
        print("="*60)
        
        # Load best model
        checkpoint = torch.load(self.output_dir / "best_model.pt")
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        test_metrics, test_z_bonds, test_time_labels = self.evaluate(self.test_loader, "test")
        
        print(f"\nTest Results:")
        for k, v in test_metrics.items():
            print(f"  {k}: {v:.4f}")
        
        # Save test embeddings for analysis
        torch.save({
            'z_bonds': test_z_bonds,
            'time_labels': test_time_labels
        }, self.output_dir / "test_embeddings.pt")
        
        # Save final metrics
        with open(self.output_dir / "test_metrics.json", 'w') as f:
            json.dump(test_metrics, f, indent=2)
        
        if HAS_WANDB:
            wandb.log(test_metrics, step=self.global_step)
            wandb.finish()
        
        return test_metrics


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--split', type=str, default='temporal_holdout',
                        choices=['temporal_holdout', 'stratified_random'])
    args = parser.parse_args()
    
    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    # Set seed
    seed = config['experiment']['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Train
    trainer = BIPTrainer(config, args.split)
    metrics = trainer.train()
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()
```

---

## 8. EVALUATION PROTOCOL

Create `src/evaluate.py`:

```python
#!/usr/bin/env python3
"""
Comprehensive evaluation for BIP temporal invariance.

Implements all statistical tests described in the protocol.
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple
import numpy as np
from scipy import stats
from scipy.stats import pearsonr, spearmanr
import torch
import torch.nn.functional as F
from tqdm import tqdm
import yaml

from models.bip_model import BIPTemporalInvarianceModel, get_tokenizer
from train import MoralPassageDataset, collate_fn
from torch.utils.data import DataLoader

# =============================================================================
# EVALUATION METRICS
# =============================================================================

class BIPEvaluator:
    """
    Comprehensive evaluation of BIP temporal invariance.
    
    Tests:
    1. Temporal Transfer Gap
    2. Bond Embedding Consistency
    3. Adversarial Time Prediction
    4. Cross-Corpus Transfer Matrix
    5. Time-Gap Correlation
    """
    
    def __init__(self, config: dict, model_path: str):
        self.config = config
        self.device = torch.device(config['experiment'].get('device', 'cuda'))
        
        # Load model
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model = BIPTemporalInvarianceModel(config['model']).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        self.tokenizer = get_tokenizer(config['model']['encoder'])
        
        # Load data
        with open("data/splits/all_splits.json") as f:
            self.splits = json.load(f)
        
        self.passages_file = f"{config['data']['processed_path']}/passages.jsonl"
        self.bonds_file = f"{config['data']['processed_path']}/bond_structures.jsonl"
        
        # Load passage metadata
        self.passages = {}
        with open(self.passages_file) as f:
            for line in f:
                p = json.loads(line)
                self.passages[p['id']] = p
        
        # Load bond structures
        self.bonds = {}
        with open(self.bonds_file) as f:
            for line in f:
                b = json.loads(line)
                self.bonds[b['passage_id']] = b['bond_structure']
    
    @torch.no_grad()
    def get_embeddings(self, passage_ids: List[str]) -> Tuple[torch.Tensor, List[int], List[str]]:
        """Get z_bond embeddings for passages."""
        dataset = MoralPassageDataset(
            passage_ids, self.passages_file, self.bonds_file,
            self.tokenizer, self.config['model'].get('max_length', 512)
        )
        loader = DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
        
        all_embeddings = []
        all_time_labels = []
        all_ids = []
        
        for batch in loader:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            
            z_bond = self.model.encode_bond(input_ids, attention_mask)
            
            all_embeddings.append(z_bond.cpu())
            all_time_labels.extend(batch['time_labels'].tolist())
            all_ids.extend(batch['passage_ids'])
        
        return torch.cat(all_embeddings), all_time_labels, all_ids
    
    # =========================================================================
    # TEST 1: TEMPORAL TRANSFER GAP
    # =========================================================================
    
    def test_temporal_transfer_gap(self) -> Dict:
        """
        Compare accuracy on within-era vs cross-era test.
        
        BIP predicts: Gap should be near zero.
        """
        print("\n" + "="*60)
        print("TEST 1: TEMPORAL TRANSFER GAP")
        print("="*60)
        
        split = self.splits['temporal_holdout']
        
        # Get predictions on test set (modern era)
        test_ids = split['test_ids']
        embeddings, time_labels, _ = self.get_embeddings(test_ids)
        
        # Hohfeldian accuracy (what we care about for transfer)
        dataset = MoralPassageDataset(
            test_ids, self.passages_file, self.bonds_file,
            self.tokenizer, self.config['model'].get('max_length', 512)
        )
        loader = DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
        
        correct = 0
        total = 0
        
        for batch in loader:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            hohfeld_labels = batch['hohfeld_labels'].to(self.device)
            
            outputs = self.model(input_ids, attention_mask)
            preds = outputs['hohfeld_pred'].argmax(dim=-1)
            
            correct += (preds == hohfeld_labels).sum().item()
            total += len(hohfeld_labels)
        
        cross_era_acc = correct / total
        
        # Compare to validation (within ancient/medieval era)
        valid_ids = split['valid_ids']
        dataset = MoralPassageDataset(
            valid_ids, self.passages_file, self.bonds_file,
            self.tokenizer, self.config['model'].get('max_length', 512)
        )
        loader = DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
        
        correct = 0
        total = 0
        
        for batch in loader:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            hohfeld_labels = batch['hohfeld_labels'].to(self.device)
            
            outputs = self.model(input_ids, attention_mask)
            preds = outputs['hohfeld_pred'].argmax(dim=-1)
            
            correct += (preds == hohfeld_labels).sum().item()
            total += len(hohfeld_labels)
        
        within_era_acc = correct / total
        
        # Statistical test
        gap = within_era_acc - cross_era_acc
        n_test = len(test_ids)
        
        # Standard error of difference
        se = np.sqrt(
            (within_era_acc * (1 - within_era_acc) / len(valid_ids)) +
            (cross_era_acc * (1 - cross_era_acc) / n_test)
        )
        
        z_score = gap / se if se > 0 else 0
        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
        
        result = {
            'within_era_acc': within_era_acc,
            'cross_era_acc': cross_era_acc,
            'gap': gap,
            'z_score': z_score,
            'p_value': p_value,
            'bip_supported': abs(gap) < 0.05  # Less than 5% gap
        }
        
        print(f"Within-era accuracy: {within_era_acc:.4f}")
        print(f"Cross-era accuracy: {cross_era_acc:.4f}")
        print(f"Gap: {gap:.4f}")
        print(f"Z-score: {z_score:.2f}")
        print(f"P-value: {p_value:.2e}")
        print(f"BIP Supported: {result['bip_supported']}")
        
        return result
    
    # =========================================================================
    # TEST 2: BOND EMBEDDING CONSISTENCY
    # =========================================================================
    
    def test_embedding_consistency(self) -> Dict:
        """
        For bond-isomorphic pairs across eras:
        Measure cosine similarity of z_bond embeddings.
        
        BIP predicts: High similarity regardless of time gap.
        """
        print("\n" + "="*60)
        print("TEST 2: BOND EMBEDDING CONSISTENCY")
        print("="*60)
        
        if 'isomorphic_pairs' not in self.splits:
            return {'error': 'No isomorphic pairs found'}
        
        pairs = self.splits['isomorphic_pairs']['test_pairs']
        
        if not pairs:
            return {'error': 'No test pairs'}
        
        # Get embeddings for all passages in pairs
        all_ids = list(set([p[0] for p in pairs] + [p[1] for p in pairs]))
        embeddings, time_labels, ids = self.get_embeddings(all_ids)
        
        # Create ID to embedding map
        id_to_emb = {pid: embeddings[i] for i, pid in enumerate(ids)}
        id_to_time = {pid: time_labels[i] for i, pid in enumerate(ids)}
        
        # Calculate similarities and time gaps
        similarities = []
        time_gaps = []
        
        for p1_id, p2_id in pairs:
            if p1_id not in id_to_emb or p2_id not in id_to_emb:
                continue
            
            e1 = id_to_emb[p1_id]
            e2 = id_to_emb[p2_id]
            
            sim = F.cosine_similarity(e1.unsqueeze(0), e2.unsqueeze(0)).item()
            
            t1 = id_to_time[p1_id]
            t2 = id_to_time[p2_id]
            gap = abs(t1 - t2)
            
            similarities.append(sim)
            time_gaps.append(gap)
        
        similarities = np.array(similarities)
        time_gaps = np.array(time_gaps)
        
        # Statistics
        mean_sim = np.mean(similarities)
        std_sim = np.std(similarities)
        
        # Correlation with time gap
        if len(similarities) > 2:
            corr, p_corr = pearsonr(similarities, time_gaps)
        else:
            corr, p_corr = 0, 1
        
        result = {
            'n_pairs': len(similarities),
            'mean_similarity': mean_sim,
            'std_similarity': std_sim,
            'min_similarity': np.min(similarities) if len(similarities) > 0 else 0,
            'max_similarity': np.max(similarities) if len(similarities) > 0 else 0,
            'time_gap_correlation': corr,
            'correlation_p_value': p_corr,
            'bip_supported': mean_sim > 0.7 and abs(corr) < 0.2
        }
        
        print(f"N pairs: {result['n_pairs']}")
        print(f"Mean similarity: {mean_sim:.4f} ± {std_sim:.4f}")
        print(f"Correlation with time gap: {corr:.4f} (p={p_corr:.2e})")
        print(f"BIP Supported: {result['bip_supported']}")
        
        return result
    
    # =========================================================================
    # TEST 3: ADVERSARIAL TIME PREDICTION
    # =========================================================================
    
    def test_adversarial_time(self) -> Dict:
        """
        Time classifier accuracy from z_bond should be near chance.
        
        BIP predicts: Accuracy ≈ 1/9 = 11.1%
        """
        print("\n" + "="*60)
        print("TEST 3: ADVERSARIAL TIME PREDICTION")
        print("="*60)
        
        # Get predictions on all test data
        split = self.splits['temporal_holdout']
        test_ids = split['test_ids']
        
        dataset = MoralPassageDataset(
            test_ids, self.passages_file, self.bonds_file,
            self.tokenizer, self.config['model'].get('max_length', 512)
        )
        loader = DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
        
        all_preds = []
        all_labels = []
        
        for batch in loader:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            time_labels = batch['time_labels']
            
            outputs = self.model(input_ids, attention_mask)
            preds = outputs['time_pred_bond'].argmax(dim=-1).cpu()
            
            all_preds.extend(preds.tolist())
            all_labels.extend(time_labels.tolist())
        
        # Accuracy
        correct = sum(p == l for p, l in zip(all_preds, all_labels))
        accuracy = correct / len(all_preds)
        
        # Expected chance level
        n_periods = 9
        chance = 1 / n_periods
        
        # Test if significantly different from chance
        n = len(all_preds)
        se = np.sqrt(chance * (1 - chance) / n)
        z_score = (accuracy - chance) / se
        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
        
        result = {
            'accuracy': accuracy,
            'chance_level': chance,
            'n_samples': n,
            'z_score_from_chance': z_score,
            'p_value': p_value,
            'bip_supported': abs(accuracy - chance) < 0.05  # Within 5% of chance
        }
        
        print(f"Time prediction accuracy: {accuracy:.4f}")
        print(f"Chance level: {chance:.4f}")
        print(f"Z-score from chance: {z_score:.2f}")
        print(f"P-value: {p_value:.2e}")
        print(f"BIP Supported (≈ chance): {result['bip_supported']}")
        
        return result
    
    # =========================================================================
    # TEST 4: CROSS-CORPUS TRANSFER
    # =========================================================================
    
    def test_cross_corpus(self) -> Dict:
        """
        Test Hebrew → Dear Abby and Dear Abby → Hebrew transfer.
        
        BIP predicts: Both directions should work equally well.
        """
        print("\n" + "="*60)
        print("TEST 4: CROSS-CORPUS TRANSFER")
        print("="*60)
        
        cross = self.splits['cross_corpus']
        results = {}
        
        for direction, split in cross.items():
            print(f"\n{direction}:")
            
            test_ids = split['test_ids']
            
            dataset = MoralPassageDataset(
                test_ids, self.passages_file, self.bonds_file,
                self.tokenizer, self.config['model'].get('max_length', 512)
            )
            loader = DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
            
            correct = 0
            total = 0
            
            for batch in loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                hohfeld_labels = batch['hohfeld_labels'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask)
                preds = outputs['hohfeld_pred'].argmax(dim=-1)
                
                correct += (preds == hohfeld_labels).sum().item()
                total += len(hohfeld_labels)
            
            accuracy = correct / total
            results[direction] = accuracy
            print(f"  Accuracy: {accuracy:.4f}")
        
        # Compare directions
        gap = abs(results.get('sefaria_to_abby', 0) - results.get('abby_to_sefaria', 0))
        
        final_result = {
            'sefaria_to_abby': results.get('sefaria_to_abby', 0),
            'abby_to_sefaria': results.get('abby_to_sefaria', 0),
            'direction_gap': gap,
            'bip_supported': gap < 0.1  # Less than 10% difference between directions
        }
        
        print(f"\nDirection gap: {gap:.4f}")
        print(f"BIP Supported (symmetric transfer): {final_result['bip_supported']}")
        
        return final_result
    
    # =========================================================================
    # FULL EVALUATION
    # =========================================================================
    
    def run_all_tests(self) -> Dict:
        """Run all BIP tests and compute combined significance."""
        
        results = {
            'temporal_transfer': self.test_temporal_transfer_gap(),
            'embedding_consistency': self.test_embedding_consistency(),
            'adversarial_time': self.test_adversarial_time(),
            'cross_corpus': self.test_cross_corpus()
        }
        
        # Combined verdict
        n_supported = sum([
            results['temporal_transfer'].get('bip_supported', False),
            results['embedding_consistency'].get('bip_supported', False),
            results['adversarial_time'].get('bip_supported', False),
            results['cross_corpus'].get('bip_supported', False)
        ])
        
        results['summary'] = {
            'tests_passed': n_supported,
            'total_tests': 4,
            'bip_overall': n_supported >= 3  # At least 3 of 4 tests pass
        }
        
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        print(f"Tests passed: {n_supported}/4")
        print(f"BIP Overall: {'SUPPORTED' if results['summary']['bip_overall'] else 'NOT SUPPORTED'}")
        
        return results


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--model', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--output', type=str, default='results/evaluation_results.json')
    args = parser.parse_args()
    
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    evaluator = BIPEvaluator(config, args.model)
    results = evaluator.run_all_tests()
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_path}")

if __name__ == "__main__":
    main()
```

---

## 9-12: REMAINING SECTIONS

Create `run_experiment.sh`:

```bash
#!/bin/bash
# Complete BIP Temporal Invariance Experiment

set -e  # Exit on error

echo "========================================"
echo "BIP TEMPORAL INVARIANCE EXPERIMENT"
echo "========================================"
echo ""

# Step 1: Setup
echo "[1/8] Setting up environment..."
python -m venv venv 2>/dev/null || true
source venv/bin/activate
pip install -r requirements.txt -q

# Step 2: Verify data
echo "[2/8] Verifying data..."
python src/data/verify_data.py

# Step 3: Preprocess
echo "[3/8] Preprocessing corpora..."
python src/data/preprocess.py

# Step 4: Extract bonds
echo "[4/8] Extracting bond structures..."
python src/data/extract_bonds.py

# Step 5: Generate splits
echo "[5/8] Generating train/test splits..."
python src/data/generate_splits.py

# Step 6: Train on temporal holdout (primary BIP test)
echo "[6/8] Training model (temporal holdout)..."
python src/train.py --split temporal_holdout

# Step 7: Train on stratified random (control)
echo "[7/8] Training model (stratified control)..."
python src/train.py --split stratified_random

# Step 8: Full evaluation
echo "[8/8] Running comprehensive evaluation..."
python src/evaluate.py \
    --model models/checkpoints/temporal_holdout/best_model.pt \
    --output results/bip_evaluation.json

echo ""
echo "========================================"
echo "EXPERIMENT COMPLETE"
echo "========================================"
echo ""
echo "Results saved to: results/bip_evaluation.json"
echo ""
cat results/bip_evaluation.json | python -c "import sys,json; d=json.load(sys.stdin); print('BIP SUPPORTED:', d['summary']['bip_overall'])"
```

---

This completes the protocol. Save all files and run:

```bash
chmod +x run_experiment.sh
./run_experiment.sh
```

**Good luck. This is history.**
