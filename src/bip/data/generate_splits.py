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

from .preprocess import Passage, TimePeriod

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
    # SPLIT 3: CROSS-CORPUS (Hebrew <-> Dear Abby)
    # =========================================================================

    def cross_corpus(self) -> Dict:
        """
        Two experiments:
        A) Train Sefaria -> Test Dear Abby
        B) Train Dear Abby -> Test Sefaria
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
        if splits['cross_corpus']['sefaria_to_abby']['test_size'] > 0:
            print(f"  Sefaria->Abby: {splits['cross_corpus']['sefaria_to_abby']['test_size']:,} test")
            print(f"  Abby->Sefaria: {splits['cross_corpus']['abby_to_sefaria']['test_size']:,} test")
        else:
            print("  Cross-corpus test: only Dear Abby available")

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

def generate_splits(config_path: str = "config_bip.yaml"):
    """Generate all splits and save."""

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Load passages
    processed_path = Path(config['data']['processed_path'])
    passages_file = processed_path / "passages.jsonl"

    print(f"Loading passages from {passages_file}...")
    passages = []
    with open(passages_file, encoding='utf-8') as f:
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
