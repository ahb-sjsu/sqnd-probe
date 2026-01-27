#!/usr/bin/env python3
"""
Batch Extract - Process entire BIP corpus through structured extraction.

Usage:
    python scripts/batch_extract.py --corpus-cache data/corpus_cache.json --output data/bonds.db

Features:
    - Resume capability (tracks processed passages)
    - Rate limiting for API calls
    - Progress tracking with tqdm
    - Saves to SQLite database
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.bip.corpus_extraction import (
    CorpusPassage,
    extract_from_passage,
    get_language_canonical,
)
from src.bip.bond_database import BondDatabase, print_database_stats


def load_corpus_cache(path: Path) -> dict:
    """Load corpus cache from JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def flatten_corpus(corpus_cache: dict) -> list[dict]:
    """Flatten corpus cache to list of passage dicts."""
    passages = []

    for key, entries in corpus_cache.items():
        # Handle tuple keys stored as strings
        if isinstance(key, str) and key.startswith("("):
            # Parse tuple string like "('hebrew', 'BIBLICAL')"
            try:
                parts = key.strip("()").replace("'", "").split(", ")
                lang, period = parts[0], parts[1] if len(parts) > 1 else None
            except (ValueError, IndexError):
                lang, period = key, None
        elif isinstance(key, tuple):
            lang, period = key[0], key[1] if len(key) > 1 else None
        else:
            lang, period = key, None

        for entry in entries:
            if isinstance(entry, str):
                entry = {"text": entry}

            entry["language"] = entry.get("language", lang)
            if period:
                entry["time_period"] = entry.get("time_period", period)

            # Generate unique ID if missing
            if "id" not in entry:
                text_hash = hash(entry.get("text", "")[:100]) % 1000000
                entry["id"] = f"corpus_{lang}_{text_hash}"

            passages.append(entry)

    return passages


def load_processed_ids(checkpoint_path: Path) -> set[str]:
    """Load set of already processed passage IDs."""
    if checkpoint_path.exists():
        with open(checkpoint_path, "r") as f:
            return set(line.strip() for line in f)
    return set()


def save_checkpoint(checkpoint_path: Path, passage_id: str) -> None:
    """Append processed passage ID to checkpoint file."""
    with open(checkpoint_path, "a") as f:
        f.write(f"{passage_id}\n")


def main():
    parser = argparse.ArgumentParser(description="Batch extract moral bonds from corpus")
    parser.add_argument(
        "--corpus-cache",
        type=Path,
        required=True,
        help="Path to corpus cache JSON file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/bonds.db"),
        help="Path to output SQLite database",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to checkpoint file (for resume capability)",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Process only N passages (for testing)",
    )
    parser.add_argument(
        "--languages",
        type=str,
        nargs="+",
        default=None,
        help="Filter to specific languages",
    )
    parser.add_argument(
        "--traditions",
        type=str,
        nargs="+",
        default=None,
        help="Filter to specific traditions",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Delay between API calls (seconds)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="claude-sonnet-4-20250514",
        help="Claude model to use",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be processed without making API calls",
    )
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Just print database statistics",
    )

    args = parser.parse_args()

    # Stats only mode
    if args.stats_only:
        if args.output.exists():
            with BondDatabase(args.output) as db:
                print_database_stats(db)
        else:
            print(f"Database not found: {args.output}")
        return

    # Check API key
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key and not args.dry_run:
        print("Error: ANTHROPIC_API_KEY environment variable not set")
        print("Set it with: export ANTHROPIC_API_KEY=your-key-here")
        sys.exit(1)

    # Load corpus
    print(f"Loading corpus from {args.corpus_cache}...")
    corpus_cache = load_corpus_cache(args.corpus_cache)
    passages = flatten_corpus(corpus_cache)
    print(f"  Found {len(passages)} total passages")

    # Filter by language
    if args.languages:
        canonical_langs = [get_language_canonical(l) for l in args.languages]
        passages = [
            p for p in passages
            if get_language_canonical(p.get("language", "")) in canonical_langs
        ]
        print(f"  After language filter: {len(passages)} passages")

    # Sample if requested
    if args.sample_size:
        import random
        random.seed(42)
        passages = random.sample(passages, min(args.sample_size, len(passages)))
        print(f"  Sampled {len(passages)} passages")

    # Load checkpoint
    checkpoint_path = args.checkpoint or args.output.with_suffix(".checkpoint")
    processed_ids = load_processed_ids(checkpoint_path)
    print(f"  Already processed: {len(processed_ids)} passages")

    # Filter out processed
    passages = [p for p in passages if p.get("id") not in processed_ids]
    print(f"  Remaining to process: {len(passages)} passages")

    if args.dry_run:
        print("\n[DRY RUN] Would process:")
        for p in passages[:10]:
            lang = p.get("language", "unknown")
            text = p.get("text", "")[:50]
            print(f"  {p.get('id')}: [{lang}] {text}...")
        if len(passages) > 10:
            print(f"  ... and {len(passages) - 10} more")
        return

    # Initialize
    try:
        import anthropic
        client = anthropic.Anthropic()
    except ImportError:
        print("Error: anthropic package not installed")
        print("Install with: pip install anthropic")
        sys.exit(1)

    # Try to import tqdm for progress bar
    try:
        from tqdm import tqdm
        use_tqdm = True
    except ImportError:
        use_tqdm = False
        print("(Install tqdm for progress bar: pip install tqdm)")

    # Open database
    db = BondDatabase(args.output)

    # Process passages
    total_bonds = 0
    errors = 0

    iterator = tqdm(passages, desc="Extracting") if use_tqdm else passages

    for i, entry in enumerate(iterator):
        passage = CorpusPassage.from_corpus_entry(entry)

        if not use_tqdm and i % 10 == 0:
            print(f"  Processing {i + 1}/{len(passages)}: {passage.id}")

        try:
            bonds = extract_from_passage(client, passage, model=args.model)

            if bonds:
                db.add_bonds(bonds)
                total_bonds += len(bonds)

            # Save checkpoint
            save_checkpoint(checkpoint_path, passage.id)

            # Rate limiting
            if args.delay > 0:
                time.sleep(args.delay)

        except Exception as e:
            errors += 1
            print(f"\nError processing {passage.id}: {e}")
            continue

    # Final stats
    print(f"\n{'=' * 60}")
    print("EXTRACTION COMPLETE")
    print(f"{'=' * 60}")
    print(f"Passages processed: {len(passages)}")
    print(f"Bonds extracted: {total_bonds}")
    print(f"Errors: {errors}")
    print(f"Database: {args.output}")

    print_database_stats(db)
    db.close()


if __name__ == "__main__":
    main()
