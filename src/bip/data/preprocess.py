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
        date_col = self._find_column(df, ['date', 'Date', 'DATE', 'year'])
        question_col = self._find_column(df, ['question', 'Question', 'letter', 'Letter', 'text', 'question_only'])
        answer_col = self._find_column(df, ['answer', 'Answer', 'response', 'Response', 'reply'])

        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Dear Abby"):
            date = str(row.get(date_col, '')) if date_col else ''
            question = str(row.get(question_col, ''))
            answer = str(row.get(answer_col, '')) if answer_col else ''

            if not question or question == 'nan':
                continue

            # Combine Q&A if answer exists
            if answer and answer != 'nan':
                full_text = f"QUESTION: {question}\n\nANSWER: {answer}"
            else:
                full_text = question

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

            if not question:
                continue

            if answer:
                full_text = f"QUESTION: {question}\n\nANSWER: {answer}"
            else:
                full_text = question

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

def preprocess_all(config_path: str = "config_bip.yaml"):
    """Run full preprocessing pipeline."""

    # Load config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    data_config = config.get('data', {})
    prep_config = config.get('preprocessing', {})

    # Load corpora
    sefaria_path = Path(data_config.get('sefaria_path', 'data/raw/Sefaria-Export'))
    sefaria_passages = []

    if sefaria_path.exists():
        sefaria_loader = SefariaLoader(str(sefaria_path), data_config)
        sefaria_passages = sefaria_loader.load(data_config.get('max_sefaria_passages'))
    else:
        print(f"Warning: Sefaria path not found: {sefaria_path}")

    abby_loader = DearAbbyLoader(
        data_config.get('dear_abby_path', 'data/raw/dear_abby.csv'),
        data_config
    )
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
    with open(passages_file, 'w', encoding='utf-8') as f:
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
