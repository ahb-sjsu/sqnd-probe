"""
Bond Extraction Training Pipeline
=================================
Uses ETHICS, Scruples, and EthicsSuite datasets to train/validate
bond extraction for the BIP project.

Extracts: (agent, patient, bond_type, context, hohfeld_state)
"""

import json
import re
import os
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Tuple
from collections import defaultdict

# =============================================================================
# BIP BOND SCHEMA
# =============================================================================

@dataclass
class BondAnnotation:
    """A moral bond extracted from text."""
    text: str                      # Original text
    agent: Optional[str]           # Who has the duty/obligation
    patient: Optional[str]         # Who holds the claim/is affected
    bond_type: str                 # OBLIGATION, PROHIBITION, PERMISSION, etc.
    hohfeld_state: str             # DUTY, CLAIM, LIBERTY, POWER, etc.
    context: str                   # prescriptive, descriptive, hypothetical
    confidence: float              # 0.0-1.0
    source_dataset: str            # ethics, scruples, ethicsuite
    source_category: str           # deontology, justice, commonsense, etc.
    raw_label: str                 # Original label from dataset


# Bond type definitions
BOND_TYPES = [
    "OBLIGATION",      # Must do X
    "PROHIBITION",     # Must not do X
    "PERMISSION",      # May do X
    "CLAIM",           # Has right to X
    "POWER",           # Can change normative relations
    "IMMUNITY",        # Cannot have relations changed
    "VIRTUE",          # Character-based moral quality
    "VICE",            # Negative character trait
    "SUPEREROGATORY",  # Beyond duty (praiseworthy but not required)
]

# Hohfeld states
HOHFELD_STATES = [
    "DUTY",      # Correlative of CLAIM
    "CLAIM",     # Correlative of DUTY
    "LIBERTY",   # Correlative of NO-CLAIM
    "NO_CLAIM",  # Correlative of LIBERTY
    "POWER",     # Correlative of LIABILITY
    "LIABILITY", # Correlative of POWER
    "IMMUNITY",  # Correlative of DISABILITY
    "DISABILITY",# Correlative of IMMUNITY
]

# Context types
CONTEXT_TYPES = ["prescriptive", "descriptive", "hypothetical", "unknown"]


# =============================================================================
# ETHICS DATASET LOADER
# =============================================================================

class EthicsDatasetLoader:
    """
    Loader for hendrycks/ethics dataset.
    Categories: commonsense, deontology, justice, utilitarianism, virtue
    """

    # Map ETHICS categories to bond types
    CATEGORY_TO_BOND = {
        "deontology": ("OBLIGATION", "DUTY"),
        "justice": ("CLAIM", "CLAIM"),
        "virtue": ("VIRTUE", "DUTY"),
        "utilitarianism": ("PERMISSION", "LIBERTY"),
        "commonsense": ("OBLIGATION", "DUTY"),
    }

    def __init__(self, cache_dir: str = "data/ethics_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def load(self, use_huggingface: bool = True) -> List[BondAnnotation]:
        """Load ETHICS dataset and convert to bond annotations."""
        annotations = []

        if use_huggingface:
            annotations = self._load_from_huggingface()
        else:
            annotations = self._load_from_github()

        return annotations

    def _load_from_huggingface(self) -> List[BondAnnotation]:
        """Load from HuggingFace datasets."""
        try:
            from datasets import load_dataset
        except ImportError:
            print("Installing datasets library...")
            os.system("pip install datasets -q")
            from datasets import load_dataset

        annotations = []
        categories = ["commonsense", "deontology", "justice", "utilitarianism", "virtue"]

        for category in categories:
            print(f"  Loading ETHICS/{category}...")
            try:
                dataset = load_dataset("hendrycks/ethics", category, trust_remote_code=True)

                for split in ["train", "test"]:
                    if split not in dataset:
                        continue
                    for item in dataset[split]:
                        ann = self._convert_ethics_item(item, category)
                        if ann:
                            annotations.append(ann)

            except Exception as e:
                print(f"    Warning: Could not load {category}: {e}")

        return annotations

    def _load_from_github(self) -> List[BondAnnotation]:
        """Load directly from GitHub raw files."""
        import urllib.request

        base_url = "https://raw.githubusercontent.com/hendrycks/ethics/master"
        annotations = []

        categories = {
            "commonsense": f"{base_url}/ethics/commonsense/cm_train.csv",
            "deontology": f"{base_url}/ethics/deontology/deontology_train.csv",
            "justice": f"{base_url}/ethics/justice/justice_train.csv",
            "utilitarianism": f"{base_url}/ethics/utilitarianism/util_train.csv",
            "virtue": f"{base_url}/ethics/virtue/virtue_train.csv",
        }

        for category, url in categories.items():
            print(f"  Downloading ETHICS/{category}...")
            try:
                cache_file = self.cache_dir / f"ethics_{category}.csv"
                if not cache_file.exists():
                    urllib.request.urlretrieve(url, cache_file)

                with open(cache_file, encoding='utf-8') as f:
                    lines = f.readlines()
                    for line in lines[1:]:  # Skip header
                        parts = line.strip().split(',', 1)
                        if len(parts) >= 2:
                            label = parts[0]
                            text = parts[1].strip('"')
                            item = {"label": int(label) if label.isdigit() else 0, "input": text}
                            ann = self._convert_ethics_item(item, category)
                            if ann:
                                annotations.append(ann)
            except Exception as e:
                print(f"    Warning: Could not load {category}: {e}")

        return annotations

    def _convert_ethics_item(self, item: dict, category: str) -> Optional[BondAnnotation]:
        """Convert ETHICS item to BondAnnotation."""
        # Get text field (varies by category)
        text = item.get("input") or item.get("scenario") or item.get("text", "")
        if not text or len(text) < 10:
            return None

        # Get label
        label = item.get("label", 0)

        # Map category to bond type
        bond_type, hohfeld = self.CATEGORY_TO_BOND.get(category, ("OBLIGATION", "DUTY"))

        # Extract agent/patient from text
        agent, patient = self._extract_agent_patient(text)

        # Determine if this is a violation (label=1 often means wrong/unethical)
        if label == 1:
            # Violation detected - this describes breaking a norm
            context = "descriptive"  # Describing what happened
            if bond_type == "OBLIGATION":
                bond_type = "PROHIBITION"  # They did something wrong
        else:
            context = "prescriptive"  # Describing what should be

        return BondAnnotation(
            text=text[:500],  # Truncate long texts
            agent=agent,
            patient=patient,
            bond_type=bond_type,
            hohfeld_state=hohfeld,
            context=context,
            confidence=0.8,  # Dataset labels are fairly reliable
            source_dataset="ethics",
            source_category=category,
            raw_label=str(label),
        )

    def _extract_agent_patient(self, text: str) -> Tuple[Optional[str], Optional[str]]:
        """Extract agent and patient from text using patterns."""
        agent = None
        patient = None

        # Common patterns
        # "I should..." -> agent = speaker
        if re.match(r"^I\s+(should|must|ought|need|have to)", text, re.I):
            agent = "speaker"

        # "You should..." -> agent = addressee
        elif re.match(r"^You\s+(should|must|ought|need|have to)", text, re.I):
            agent = "addressee"

        # "[Person] should..." -> agent = person
        match = re.match(r"^([A-Z][a-z]+)\s+(should|must|ought|needs to|has to)", text)
        if match:
            agent = match.group(1).lower()

        # Look for patient patterns
        # "...help [person]" or "...to [person]"
        patient_match = re.search(r"(help|assist|protect|harm|hurt|give to|take from)\s+(\w+)", text, re.I)
        if patient_match:
            patient = patient_match.group(2).lower()
            if patient in ["the", "a", "an", "my", "your", "his", "her"]:
                patient = None

        return agent, patient


# =============================================================================
# SCRUPLES DATASET LOADER
# =============================================================================

class ScruplesDatasetLoader:
    """
    Loader for allenai/scruples dataset.
    Contains real-life anecdotes with ethical judgments.
    Labels: AUTHOR_WRONG, OTHER_WRONG, EVERYBODY_WRONG, NOBODY_WRONG, INFO
    """

    # Map Scruples labels to bond info
    LABEL_TO_BOND = {
        "AUTHOR_WRONG": ("PROHIBITION", "DUTY", "author"),      # Author violated norm
        "OTHER_WRONG": ("PROHIBITION", "DUTY", "other"),        # Other party violated
        "EVERYBODY_WRONG": ("PROHIBITION", "DUTY", "both"),     # Both violated
        "NOBODY_WRONG": ("PERMISSION", "LIBERTY", None),        # No violation
        "INFO": ("OBLIGATION", "DUTY", None),                   # Need more info
    }

    def __init__(self, cache_dir: str = "data/scruples_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def load(self, max_items: int = 10000) -> List[BondAnnotation]:
        """Load Scruples dataset."""
        annotations = []

        # Try HuggingFace first
        try:
            annotations = self._load_from_huggingface(max_items)
        except Exception as e:
            print(f"  HuggingFace load failed: {e}")
            print("  Trying GitHub...")
            annotations = self._load_from_github(max_items)

        return annotations

    def _load_from_huggingface(self, max_items: int) -> List[BondAnnotation]:
        """Load from HuggingFace."""
        try:
            from datasets import load_dataset
        except ImportError:
            os.system("pip install datasets -q")
            from datasets import load_dataset

        annotations = []

        print("  Loading Scruples/anecdotes...")
        try:
            dataset = load_dataset("allenai/scruples", "anecdotes", trust_remote_code=True)

            count = 0
            for split in ["train", "dev", "test"]:
                if split not in dataset:
                    continue
                for item in dataset[split]:
                    if count >= max_items:
                        break
                    ann = self._convert_scruples_item(item)
                    if ann:
                        annotations.append(ann)
                        count += 1

        except Exception as e:
            print(f"    Could not load anecdotes: {e}")

        # Also try dilemmas
        print("  Loading Scruples/dilemmas...")
        try:
            dataset = load_dataset("allenai/scruples", "dilemmas", trust_remote_code=True)

            count = 0
            for split in ["train", "dev", "test"]:
                if split not in dataset:
                    continue
                for item in dataset[split]:
                    if count >= max_items // 2:  # Fewer dilemmas
                        break
                    ann = self._convert_dilemma_item(item)
                    if ann:
                        annotations.append(ann)
                        count += 1

        except Exception as e:
            print(f"    Could not load dilemmas: {e}")

        return annotations

    def _load_from_github(self, max_items: int) -> List[BondAnnotation]:
        """Load from GitHub releases."""
        import urllib.request
        import gzip

        # Scruples data URLs
        base_url = "https://github.com/allenai/scruples/raw/main/data"

        annotations = []
        # This is a fallback - actual loading would need the released data files
        print("    Note: Full Scruples data requires downloading from releases")

        return annotations

    def _convert_scruples_item(self, item: dict) -> Optional[BondAnnotation]:
        """Convert Scruples anecdote to BondAnnotation."""
        title = item.get("title", "")
        text = item.get("text", "")

        full_text = f"{title}\n{text}" if title else text
        if len(full_text) < 20:
            return None

        # Get label (binarized or distribution)
        label = item.get("binarized_label") or item.get("label", "INFO")
        if isinstance(label, int):
            label = "AUTHOR_WRONG" if label == 1 else "NOBODY_WRONG"

        # Map to bond info
        bond_type, hohfeld, violator = self.LABEL_TO_BOND.get(
            label, ("OBLIGATION", "DUTY", None)
        )

        # Agent is whoever violated the norm
        agent = None
        patient = None
        if violator == "author":
            agent = "author"
            patient = "other"
        elif violator == "other":
            agent = "other"
            patient = "author"
        elif violator == "both":
            agent = "both"
            patient = "both"

        return BondAnnotation(
            text=full_text[:500],
            agent=agent,
            patient=patient,
            bond_type=bond_type,
            hohfeld_state=hohfeld,
            context="descriptive",  # Real-life anecdotes describe what happened
            confidence=0.7,  # Community judgments have some noise
            source_dataset="scruples",
            source_category="anecdotes",
            raw_label=label,
        )

    def _convert_dilemma_item(self, item: dict) -> Optional[BondAnnotation]:
        """Convert Scruples dilemma to BondAnnotation."""
        action1 = item.get("action1", "")
        action2 = item.get("action2", "")

        # Dilemmas present two choices
        text = f"Choice A: {action1}\nChoice B: {action2}"
        if len(text) < 20:
            return None

        # Get which action is less ethical
        label = item.get("label", 0)  # 0 or 1 indicating which is worse

        return BondAnnotation(
            text=text[:500],
            agent="actor",
            patient="affected",
            bond_type="OBLIGATION",  # Dilemmas about what one should do
            hohfeld_state="DUTY",
            context="hypothetical",  # Dilemmas are hypothetical scenarios
            confidence=0.6,  # Dilemmas are inherently ambiguous
            source_dataset="scruples",
            source_category="dilemmas",
            raw_label=str(label),
        )


# =============================================================================
# ETHICSUITE LOADER
# =============================================================================

class EthicsSuiteLoader:
    """
    Loader for LLM-Ethics/EthicsSuite dataset.
    Contains 20k complex moral situations built on ETHICS.
    """

    def __init__(self, cache_dir: str = "data/ethicsuite_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def load(self, max_items: int = 10000) -> List[BondAnnotation]:
        """Load EthicsSuite dataset."""
        annotations = []

        # Download from GitHub
        import urllib.request

        url = "https://raw.githubusercontent.com/llm-ethics/ethicssuite/main/data/ethicsuite.jsonl"
        cache_file = self.cache_dir / "ethicsuite.jsonl"

        print("  Loading EthicsSuite...")
        try:
            if not cache_file.exists():
                print("    Downloading from GitHub...")
                urllib.request.urlretrieve(url, cache_file)

            with open(cache_file, encoding='utf-8') as f:
                count = 0
                for line in f:
                    if count >= max_items:
                        break
                    item = json.loads(line)
                    ann = self._convert_item(item)
                    if ann:
                        annotations.append(ann)
                        count += 1

        except Exception as e:
            print(f"    Warning: Could not load EthicsSuite: {e}")

        return annotations

    def _convert_item(self, item: dict) -> Optional[BondAnnotation]:
        """Convert EthicsSuite item to BondAnnotation."""
        text = item.get("text", "")
        if len(text) < 20:
            return None

        source = item.get("source", "unknown")  # e.g., "deontology", "justice"
        original = item.get("original_text", "")

        # Map source to bond type (same as ETHICS)
        category_map = {
            "deontology": ("OBLIGATION", "DUTY"),
            "justice": ("CLAIM", "CLAIM"),
            "virtue": ("VIRTUE", "DUTY"),
            "utilitarianism": ("PERMISSION", "LIBERTY"),
            "commonsense": ("OBLIGATION", "DUTY"),
        }

        bond_type, hohfeld = category_map.get(source, ("OBLIGATION", "DUTY"))

        # Extract agent/patient
        agent, patient = self._extract_roles(text)

        return BondAnnotation(
            text=text[:500],
            agent=agent,
            patient=patient,
            bond_type=bond_type,
            hohfeld_state=hohfeld,
            context="hypothetical",  # Complex scenarios are often hypothetical
            confidence=0.75,
            source_dataset="ethicsuite",
            source_category=source,
            raw_label=original[:100] if original else "",
        )

    def _extract_roles(self, text: str) -> Tuple[Optional[str], Optional[str]]:
        """Extract agent and patient from complex text."""
        agent = None
        patient = None

        # Look for named entities or pronouns as agents
        agent_patterns = [
            (r"^([A-Z][a-z]+)\s+(must|should|ought|has to|needs to)", 1),
            (r"^(I|You|He|She|They|We)\s+(must|should|ought)", 1),
            (r"It is (\w+)'s (duty|obligation|responsibility)", 1),
        ]

        for pattern, group in agent_patterns:
            match = re.search(pattern, text)
            if match:
                agent = match.group(group).lower()
                break

        # Look for patient patterns
        patient_patterns = [
            (r"(help|protect|assist|serve|support)\s+(\w+)", 2),
            (r"to\s+(\w+)'s\s+(benefit|welfare|good)", 1),
            (r"(harm|hurt|damage|injure)\s+(\w+)", 2),
        ]

        for pattern, group in patient_patterns:
            match = re.search(pattern, text, re.I)
            if match:
                patient = match.group(group).lower()
                if patient in ["the", "a", "an"]:
                    patient = None
                break

        return agent, patient


# =============================================================================
# UNIFIED PIPELINE
# =============================================================================

class BondExtractionPipeline:
    """
    Unified pipeline for loading and processing all ethics datasets
    for bond extraction training.
    """

    def __init__(self, cache_dir: str = "data/bond_training"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.ethics_loader = EthicsDatasetLoader(cache_dir + "/ethics")
        self.scruples_loader = ScruplesDatasetLoader(cache_dir + "/scruples")
        self.ethicsuite_loader = EthicsSuiteLoader(cache_dir + "/ethicsuite")

    def load_all(self,
                 include_ethics: bool = True,
                 include_scruples: bool = True,
                 include_ethicsuite: bool = True,
                 max_per_dataset: int = 10000) -> List[BondAnnotation]:
        """Load all datasets and return unified annotations."""

        all_annotations = []

        print("=" * 60)
        print("BOND EXTRACTION TRAINING PIPELINE")
        print("=" * 60)

        if include_ethics:
            print("\n[1] Loading ETHICS dataset...")
            ethics_anns = self.ethics_loader.load()
            print(f"    Loaded {len(ethics_anns):,} annotations")
            all_annotations.extend(ethics_anns[:max_per_dataset])

        if include_scruples:
            print("\n[2] Loading Scruples dataset...")
            scruples_anns = self.scruples_loader.load(max_per_dataset)
            print(f"    Loaded {len(scruples_anns):,} annotations")
            all_annotations.extend(scruples_anns)

        if include_ethicsuite:
            print("\n[3] Loading EthicsSuite dataset...")
            suite_anns = self.ethicsuite_loader.load(max_per_dataset)
            print(f"    Loaded {len(suite_anns):,} annotations")
            all_annotations.extend(suite_anns)

        print(f"\n{'=' * 60}")
        print(f"TOTAL: {len(all_annotations):,} bond annotations")
        print("=" * 60)

        return all_annotations

    def get_statistics(self, annotations: List[BondAnnotation]) -> Dict:
        """Get statistics about the annotations."""
        stats = {
            "total": len(annotations),
            "by_dataset": defaultdict(int),
            "by_bond_type": defaultdict(int),
            "by_hohfeld": defaultdict(int),
            "by_context": defaultdict(int),
            "by_category": defaultdict(int),
            "has_agent": 0,
            "has_patient": 0,
        }

        for ann in annotations:
            stats["by_dataset"][ann.source_dataset] += 1
            stats["by_bond_type"][ann.bond_type] += 1
            stats["by_hohfeld"][ann.hohfeld_state] += 1
            stats["by_context"][ann.context] += 1
            stats["by_category"][ann.source_category] += 1
            if ann.agent:
                stats["has_agent"] += 1
            if ann.patient:
                stats["has_patient"] += 1

        return dict(stats)

    def save_training_data(self,
                           annotations: List[BondAnnotation],
                           output_file: str = "bond_training_data.jsonl") -> str:
        """Save annotations in training format."""
        output_path = self.cache_dir / output_file

        with open(output_path, "w", encoding="utf-8") as f:
            for ann in annotations:
                f.write(json.dumps(asdict(ann), ensure_ascii=False) + "\n")

        print(f"Saved {len(annotations):,} annotations to {output_path}")
        return str(output_path)

    def create_train_test_split(self,
                                 annotations: List[BondAnnotation],
                                 test_ratio: float = 0.2,
                                 seed: int = 42) -> Tuple[List, List]:
        """Split annotations into train/test sets."""
        import random
        random.seed(seed)

        shuffled = annotations.copy()
        random.shuffle(shuffled)

        split_idx = int(len(shuffled) * (1 - test_ratio))
        train = shuffled[:split_idx]
        test = shuffled[split_idx:]

        return train, test

    def export_for_bip(self,
                       annotations: List[BondAnnotation],
                       output_file: str = "ethics_corpus.jsonl") -> str:
        """Export in BIP passage format for Cell 2."""
        output_path = self.cache_dir / output_file

        passages = []
        for i, ann in enumerate(annotations):
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
            passages.append(passage)

        with open(output_path, "w", encoding="utf-8") as f:
            for p in passages:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")

        print(f"Exported {len(passages):,} passages to {output_path}")
        return str(output_path)


# =============================================================================
# CLI / MAIN
# =============================================================================

def main():
    """Run the bond extraction pipeline."""
    import argparse

    parser = argparse.ArgumentParser(description="Bond Extraction Training Pipeline")
    parser.add_argument("--ethics", action="store_true", default=True, help="Include ETHICS dataset")
    parser.add_argument("--scruples", action="store_true", default=True, help="Include Scruples dataset")
    parser.add_argument("--ethicsuite", action="store_true", default=True, help="Include EthicsSuite dataset")
    parser.add_argument("--max-per-dataset", type=int, default=10000, help="Max items per dataset")
    parser.add_argument("--output-dir", type=str, default="data/bond_training", help="Output directory")
    parser.add_argument("--export-bip", action="store_true", help="Export in BIP format")

    args = parser.parse_args()

    # Initialize pipeline
    pipeline = BondExtractionPipeline(args.output_dir)

    # Load datasets
    annotations = pipeline.load_all(
        include_ethics=args.ethics,
        include_scruples=args.scruples,
        include_ethicsuite=args.ethicsuite,
        max_per_dataset=args.max_per_dataset,
    )

    # Print statistics
    stats = pipeline.get_statistics(annotations)

    print("\n" + "=" * 60)
    print("STATISTICS")
    print("=" * 60)

    print("\nBy Dataset:")
    for ds, count in sorted(stats["by_dataset"].items()):
        print(f"  {ds}: {count:,}")

    print("\nBy Bond Type:")
    for bt, count in sorted(stats["by_bond_type"].items(), key=lambda x: -x[1]):
        print(f"  {bt}: {count:,}")

    print("\nBy Hohfeld State:")
    for hs, count in sorted(stats["by_hohfeld"].items(), key=lambda x: -x[1]):
        print(f"  {hs}: {count:,}")

    print("\nBy Context:")
    for ctx, count in sorted(stats["by_context"].items(), key=lambda x: -x[1]):
        print(f"  {ctx}: {count:,}")

    print(f"\nAgent extracted: {stats['has_agent']:,} ({100*stats['has_agent']/stats['total']:.1f}%)")
    print(f"Patient extracted: {stats['has_patient']:,} ({100*stats['has_patient']/stats['total']:.1f}%)")

    # Save training data
    print("\n" + "=" * 60)
    print("SAVING DATA")
    print("=" * 60)

    pipeline.save_training_data(annotations)

    # Create train/test split
    train, test = pipeline.create_train_test_split(annotations)
    pipeline.save_training_data(train, "bond_train.jsonl")
    pipeline.save_training_data(test, "bond_test.jsonl")
    print(f"Train: {len(train):,} | Test: {len(test):,}")

    # Export for BIP if requested
    if args.export_bip:
        pipeline.export_for_bip(annotations)

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
