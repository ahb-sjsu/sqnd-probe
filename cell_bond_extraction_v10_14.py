# @title 2b. Load Ethics Datasets for Bond Extraction { display-mode: "form" }
# @markdown Load ETHICS, Scruples, and EthicsSuite datasets for bond extraction training
# @markdown These provide modern English moral reasoning examples with labeled judgments

# @markdown ---
# @markdown ### Dataset Selection
LOAD_ETHICS_DATASET = True  # @param {type:"boolean"}
# @markdown hendrycks/ethics: Justice, deontology, virtue, utilitarianism, commonsense

LOAD_SCRUPLES_DATASET = True  # @param {type:"boolean"}
# @markdown allenai/scruples: 32K real-life anecdotes with ethical judgments

LOAD_ETHICSUITE_DATASET = True  # @param {type:"boolean"}
# @markdown LLM-Ethics/EthicsSuite: 20K complex contextualized moral situations

# @markdown ---
# @markdown ### Size Limits (0 = unlimited)
MAX_ETHICS_ITEMS = 50000  # @param {type:"integer"}
MAX_SCRUPLES_ITEMS = 30000  # @param {type:"integer"}
MAX_ETHICSUITE_ITEMS = 20000  # @param {type:"integer"}

# @markdown ---
# @markdown ### Output Options
EXPORT_BIP_FORMAT = True  # @param {type:"boolean"}
# @markdown Export as BIP passages for integration with Cell 2 corpus

CREATE_TRAIN_TEST_SPLIT = True  # @param {type:"boolean"}
TEST_SPLIT_RATIO = 0.2  # @param {type:"number"}

import json
import re
import os
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Tuple
from collections import defaultdict

print("=" * 60)
print("BOND EXTRACTION TRAINING DATA (v10.14)")
print("=" * 60)


# =============================================================================
# BOND SCHEMA
# =============================================================================

@dataclass
class BondAnnotation:
    """A moral bond extracted from text."""
    text: str
    agent: Optional[str]
    patient: Optional[str]
    bond_type: str
    hohfeld_state: str
    context: str
    confidence: float
    source_dataset: str
    source_category: str
    raw_label: str


BOND_TYPES = [
    "OBLIGATION", "PROHIBITION", "PERMISSION", "CLAIM",
    "POWER", "IMMUNITY", "VIRTUE", "VICE", "SUPEREROGATORY"
]

HOHFELD_STATES = [
    "DUTY", "CLAIM", "LIBERTY", "NO_CLAIM",
    "POWER", "LIABILITY", "IMMUNITY", "DISABILITY"
]


# =============================================================================
# ETHICS DATASET LOADER
# =============================================================================

class EthicsLoader:
    """Load hendrycks/ethics dataset."""

    CATEGORY_TO_BOND = {
        "deontology": ("OBLIGATION", "DUTY"),
        "justice": ("CLAIM", "CLAIM"),
        "virtue": ("VIRTUE", "DUTY"),
        "utilitarianism": ("PERMISSION", "LIBERTY"),
        "commonsense": ("OBLIGATION", "DUTY"),
    }

    def load(self, max_items: int = 0) -> List[BondAnnotation]:
        try:
            from datasets import load_dataset
        except ImportError:
            print("  Installing datasets library...")
            os.system("pip install datasets -q")
            from datasets import load_dataset

        annotations = []
        categories = ["commonsense", "deontology", "justice", "utilitarianism", "virtue"]

        for category in categories:
            if max_items > 0 and len(annotations) >= max_items:
                break

            print(f"  Loading ETHICS/{category}...")
            try:
                dataset = load_dataset("hendrycks/ethics", category, trust_remote_code=True)

                for split in ["train", "test"]:
                    if split not in dataset:
                        continue
                    for item in dataset[split]:
                        if max_items > 0 and len(annotations) >= max_items:
                            break

                        text = item.get("input") or item.get("scenario") or item.get("text", "")
                        if not text or len(text) < 10:
                            continue

                        label = item.get("label", 0)
                        bond_type, hohfeld = self.CATEGORY_TO_BOND.get(category, ("OBLIGATION", "DUTY"))

                        # Extract agent/patient
                        agent, patient = self._extract_roles(text)

                        if label == 1:
                            context = "descriptive"
                            if bond_type == "OBLIGATION":
                                bond_type = "PROHIBITION"
                        else:
                            context = "prescriptive"

                        annotations.append(BondAnnotation(
                            text=text[:500],
                            agent=agent,
                            patient=patient,
                            bond_type=bond_type,
                            hohfeld_state=hohfeld,
                            context=context,
                            confidence=0.8,
                            source_dataset="ethics",
                            source_category=category,
                            raw_label=str(label),
                        ))

            except Exception as e:
                print(f"    Warning: {e}")

        return annotations

    def _extract_roles(self, text: str) -> Tuple[Optional[str], Optional[str]]:
        agent = patient = None

        if re.match(r"^I\s+(should|must|ought)", text, re.I):
            agent = "speaker"
        elif re.match(r"^You\s+(should|must|ought)", text, re.I):
            agent = "addressee"
        else:
            match = re.match(r"^([A-Z][a-z]+)\s+(should|must|ought)", text)
            if match:
                agent = match.group(1).lower()

        patient_match = re.search(r"(help|protect|harm|hurt)\s+(\w+)", text, re.I)
        if patient_match:
            p = patient_match.group(2).lower()
            if p not in ["the", "a", "an", "my", "your"]:
                patient = p

        return agent, patient


# =============================================================================
# SCRUPLES DATASET LOADER
# =============================================================================

class ScruplesLoader:
    """Load allenai/scruples dataset."""

    LABEL_TO_BOND = {
        "AUTHOR_WRONG": ("PROHIBITION", "DUTY", "author"),
        "OTHER_WRONG": ("PROHIBITION", "DUTY", "other"),
        "EVERYBODY_WRONG": ("PROHIBITION", "DUTY", "both"),
        "NOBODY_WRONG": ("PERMISSION", "LIBERTY", None),
        "INFO": ("OBLIGATION", "DUTY", None),
    }

    def load(self, max_items: int = 0) -> List[BondAnnotation]:
        try:
            from datasets import load_dataset
        except ImportError:
            os.system("pip install datasets -q")
            from datasets import load_dataset

        annotations = []

        # Load anecdotes
        print("  Loading Scruples/anecdotes...")
        try:
            dataset = load_dataset("allenai/scruples", "anecdotes", trust_remote_code=True)

            for split in ["train", "dev", "test"]:
                if split not in dataset:
                    continue
                for item in dataset[split]:
                    if max_items > 0 and len(annotations) >= max_items:
                        break

                    title = item.get("title", "")
                    text = item.get("text", "")
                    full_text = f"{title}\n{text}" if title else text

                    if len(full_text) < 20:
                        continue

                    label = item.get("binarized_label") or item.get("label", "INFO")
                    if isinstance(label, int):
                        label = "AUTHOR_WRONG" if label == 1 else "NOBODY_WRONG"

                    bond_type, hohfeld, violator = self.LABEL_TO_BOND.get(label, ("OBLIGATION", "DUTY", None))

                    agent = patient = None
                    if violator == "author":
                        agent, patient = "author", "other"
                    elif violator == "other":
                        agent, patient = "other", "author"
                    elif violator == "both":
                        agent = patient = "both"

                    annotations.append(BondAnnotation(
                        text=full_text[:500],
                        agent=agent,
                        patient=patient,
                        bond_type=bond_type,
                        hohfeld_state=hohfeld,
                        context="descriptive",
                        confidence=0.7,
                        source_dataset="scruples",
                        source_category="anecdotes",
                        raw_label=label,
                    ))

        except Exception as e:
            print(f"    Warning: {e}")

        # Load dilemmas
        print("  Loading Scruples/dilemmas...")
        try:
            dataset = load_dataset("allenai/scruples", "dilemmas", trust_remote_code=True)
            dilemma_limit = max_items // 3 if max_items > 0 else 0

            count = 0
            for split in ["train", "dev", "test"]:
                if split not in dataset:
                    continue
                for item in dataset[split]:
                    if dilemma_limit > 0 and count >= dilemma_limit:
                        break

                    action1 = item.get("action1", "")
                    action2 = item.get("action2", "")
                    text = f"Choice A: {action1}\nChoice B: {action2}"

                    if len(text) < 20:
                        continue

                    annotations.append(BondAnnotation(
                        text=text[:500],
                        agent="actor",
                        patient="affected",
                        bond_type="OBLIGATION",
                        hohfeld_state="DUTY",
                        context="hypothetical",
                        confidence=0.6,
                        source_dataset="scruples",
                        source_category="dilemmas",
                        raw_label=str(item.get("label", 0)),
                    ))
                    count += 1

        except Exception as e:
            print(f"    Warning: {e}")

        return annotations


# =============================================================================
# ETHICSUITE LOADER
# =============================================================================

class EthicsSuiteLoader:
    """Load LLM-Ethics/EthicsSuite dataset."""

    def load(self, max_items: int = 0) -> List[BondAnnotation]:
        import urllib.request

        url = "https://raw.githubusercontent.com/llm-ethics/ethicssuite/main/data/ethicsuite.jsonl"
        cache_dir = Path("data/ethics_cache")
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / "ethicsuite.jsonl"

        annotations = []

        print("  Loading EthicsSuite...")
        try:
            if not cache_file.exists():
                print("    Downloading...")
                urllib.request.urlretrieve(url, cache_file)

            category_map = {
                "deontology": ("OBLIGATION", "DUTY"),
                "justice": ("CLAIM", "CLAIM"),
                "virtue": ("VIRTUE", "DUTY"),
                "utilitarianism": ("PERMISSION", "LIBERTY"),
                "commonsense": ("OBLIGATION", "DUTY"),
            }

            with open(cache_file, encoding='utf-8') as f:
                for line in f:
                    if max_items > 0 and len(annotations) >= max_items:
                        break

                    item = json.loads(line)
                    text = item.get("text", "")
                    if len(text) < 20:
                        continue

                    source = item.get("source", "unknown")
                    bond_type, hohfeld = category_map.get(source, ("OBLIGATION", "DUTY"))

                    annotations.append(BondAnnotation(
                        text=text[:500],
                        agent=None,
                        patient=None,
                        bond_type=bond_type,
                        hohfeld_state=hohfeld,
                        context="hypothetical",
                        confidence=0.75,
                        source_dataset="ethicsuite",
                        source_category=source,
                        raw_label=item.get("original_text", "")[:100],
                    ))

        except Exception as e:
            print(f"    Warning: {e}")

        return annotations


# =============================================================================
# MAIN LOADING LOGIC
# =============================================================================

all_bond_annotations = []

if LOAD_ETHICS_DATASET:
    print("\n[1] ETHICS Dataset (hendrycks/ethics)")
    loader = EthicsLoader()
    ethics_anns = loader.load(MAX_ETHICS_ITEMS)
    print(f"    Loaded: {len(ethics_anns):,} annotations")
    all_bond_annotations.extend(ethics_anns)

if LOAD_SCRUPLES_DATASET:
    print("\n[2] Scruples Dataset (allenai/scruples)")
    loader = ScruplesLoader()
    scruples_anns = loader.load(MAX_SCRUPLES_ITEMS)
    print(f"    Loaded: {len(scruples_anns):,} annotations")
    all_bond_annotations.extend(scruples_anns)

if LOAD_ETHICSUITE_DATASET:
    print("\n[3] EthicsSuite Dataset (LLM-Ethics/EthicsSuite)")
    loader = EthicsSuiteLoader()
    suite_anns = loader.load(MAX_ETHICSUITE_ITEMS)
    print(f"    Loaded: {len(suite_anns):,} annotations")
    all_bond_annotations.extend(suite_anns)


# =============================================================================
# STATISTICS
# =============================================================================

print("\n" + "=" * 60)
print("BOND EXTRACTION DATA STATISTICS")
print("=" * 60)

stats = {
    "by_dataset": defaultdict(int),
    "by_bond_type": defaultdict(int),
    "by_hohfeld": defaultdict(int),
    "by_context": defaultdict(int),
    "by_category": defaultdict(int),
    "has_agent": 0,
    "has_patient": 0,
}

for ann in all_bond_annotations:
    stats["by_dataset"][ann.source_dataset] += 1
    stats["by_bond_type"][ann.bond_type] += 1
    stats["by_hohfeld"][ann.hohfeld_state] += 1
    stats["by_context"][ann.context] += 1
    stats["by_category"][ann.source_category] += 1
    if ann.agent:
        stats["has_agent"] += 1
    if ann.patient:
        stats["has_patient"] += 1

print(f"\nTotal annotations: {len(all_bond_annotations):,}")

print("\nBy Dataset:")
for ds, count in sorted(stats["by_dataset"].items()):
    print(f"  {ds}: {count:,}")

print("\nBy Bond Type:")
for bt, count in sorted(stats["by_bond_type"].items(), key=lambda x: -x[1]):
    print(f"  {bt}: {count:,}")

print("\nBy Context:")
for ctx, count in sorted(stats["by_context"].items(), key=lambda x: -x[1]):
    print(f"  {ctx}: {count:,}")

print(f"\nAgent extracted: {stats['has_agent']:,} ({100*stats['has_agent']/max(1,len(all_bond_annotations)):.1f}%)")
print(f"Patient extracted: {stats['has_patient']:,} ({100*stats['has_patient']/max(1,len(all_bond_annotations)):.1f}%)")


# =============================================================================
# EXPORT
# =============================================================================

output_dir = Path("data/bond_training")
output_dir.mkdir(parents=True, exist_ok=True)

# Save all annotations
print("\n" + "=" * 60)
print("SAVING DATA")
print("=" * 60)

with open(output_dir / "bond_annotations.jsonl", "w", encoding="utf-8") as f:
    for ann in all_bond_annotations:
        f.write(json.dumps(asdict(ann), ensure_ascii=False) + "\n")
print(f"Saved: {output_dir / 'bond_annotations.jsonl'}")

# Train/test split
if CREATE_TRAIN_TEST_SPLIT:
    import random
    random.seed(42)
    shuffled = all_bond_annotations.copy()
    random.shuffle(shuffled)
    split_idx = int(len(shuffled) * (1 - TEST_SPLIT_RATIO))
    train_anns = shuffled[:split_idx]
    test_anns = shuffled[split_idx:]

    with open(output_dir / "bond_train.jsonl", "w", encoding="utf-8") as f:
        for ann in train_anns:
            f.write(json.dumps(asdict(ann), ensure_ascii=False) + "\n")

    with open(output_dir / "bond_test.jsonl", "w", encoding="utf-8") as f:
        for ann in test_anns:
            f.write(json.dumps(asdict(ann), ensure_ascii=False) + "\n")

    print(f"Train/Test split: {len(train_anns):,} / {len(test_anns):,}")

# Export in BIP format
if EXPORT_BIP_FORMAT:
    bip_passages = []
    for i, ann in enumerate(all_bond_annotations):
        passage = {
            "id": f"ethics_{ann.source_dataset}_{i}",
            "text": ann.text,
            "language": "english",
            "time_periods": ["MODERN_ETHICS"],
            "tags": ["modern", "english", "western", "ethics", ann.source_category],
            "bonds": [{
                "agent": ann.agent or "unspecified",
                "patient": ann.patient or "unspecified",
                "bond_type": ann.bond_type,
                "hohfeld_state": ann.hohfeld_state,
                "context": ann.context,
                "confidence": ann.confidence,
            }],
            "source": ann.source_dataset,
            "category": ann.source_category,
        }
        bip_passages.append(passage)

    with open(output_dir / "ethics_corpus.jsonl", "w", encoding="utf-8") as f:
        for p in bip_passages:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"BIP format: {output_dir / 'ethics_corpus.jsonl'}")

print("\n" + "=" * 60)
print("BOND EXTRACTION DATA READY")
print("=" * 60)
print(f"Use data/bond_training/bond_train.jsonl for training")
print(f"Use data/bond_training/ethics_corpus.jsonl for BIP integration")
