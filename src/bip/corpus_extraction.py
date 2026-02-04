"""
Corpus Extraction Module - Extract structured moral bonds from BIP corpus data.

This module bridges the existing corpus loading infrastructure (Cell 2) with
the structured moral bond extraction system, enabling extraction from the
full multi-lingual corpus.
"""

import asyncio
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .moral_structure import (
    MoralBond,
    extract_bonds_sync,
    parse_extraction_response,
    EXTRACTION_SYSTEM_PROMPT,
    create_extraction_prompt,
)


# =============================================================================
# LANGUAGE AND TRADITION MAPPINGS
# =============================================================================

# Map corpus language codes to our canonical language names
CORPUS_LANGUAGE_MAP = {
    # Hebrew/Jewish
    "hebrew": "hebrew",
    "aramaic": "aramaic",
    # Arabic/Islamic
    "arabic": "arabic",
    # Chinese
    "classical_chinese": "classical_chinese",
    "chinese": "classical_chinese",
    # Sanskrit/Hindu
    "sanskrit": "sanskrit",
    # Pali/Buddhist
    "pali": "pali",
    # Greek/Roman
    "greek": "greek",
    "latin": "latin",
    # Modern
    "english": "english",
}

# Map corpus source types to tradition names
SOURCE_TO_TRADITION = {
    # Jewish
    "tanakh": "biblical",
    "mishnah": "rabbinic",
    "talmud": "rabbinic",
    "midrash": "rabbinic",
    "responsa": "rabbinic",
    "sefaria": "jewish",
    # Islamic
    "quran": "quranic",
    "tanzil": "quranic",
    # Chinese
    "analects": "confucian",
    "mengzi": "confucian",
    "mencius": "confucian",
    "zhuangzi": "daoist",
    "laozi": "daoist",
    "daodejing": "daoist",
    "mozi": "mohist",
    "yijing": "confucian",
    "ctext": "chinese_classical",
    # Indian
    "mahabharata": "dharma",
    "ramayana": "dharma",
    "itihasa": "dharma",
    "dhammapada": "buddhist",
    "suttacentral": "buddhist",
    "majjhima": "buddhist",
    "digha": "buddhist",
    # Greek/Roman
    "aristotle": "greek_philosophy",
    "plato": "greek_philosophy",
    "epictetus": "stoic",
    "marcus_aurelius": "stoic",
    "meditations": "stoic",
    "perseus": "classical",
    # Modern
    "kant": "modern_ethics",
    "mill": "modern_ethics",
    "spinoza": "modern_ethics",
    "gutenberg": "modern_ethics",
    "ethics": "modern_ethics",
    "scruples": "modern_american",
    "dear_abby": "modern_american",
}


def get_tradition_from_source(source: str) -> str:
    """Determine tradition from source string."""
    source_lower = source.lower()
    for key, tradition in SOURCE_TO_TRADITION.items():
        if key in source_lower:
            return tradition
    return "unknown"


def get_language_canonical(language: str) -> str:
    """Map corpus language to canonical name."""
    return CORPUS_LANGUAGE_MAP.get(language.lower(), language.lower())


# =============================================================================
# PASSAGE DATA STRUCTURE
# =============================================================================


@dataclass
class CorpusPassage:
    """A passage from the BIP corpus ready for extraction."""

    id: str
    text: str
    language: str
    tradition: str
    source: str
    period: Optional[str] = None

    @classmethod
    def from_corpus_entry(cls, entry: dict) -> "CorpusPassage":
        """Create from a corpus cache entry."""
        # Handle different entry formats
        text = entry.get("text") or entry.get("text_original") or entry.get("content", "")
        language = get_language_canonical(entry.get("language") or entry.get("lang", "unknown"))
        source = entry.get("source") or entry.get("source_type", "unknown")
        tradition = get_tradition_from_source(source)

        return cls(
            id=entry.get("id", f"passage_{hash(text) % 100000}"),
            text=text,
            language=language,
            tradition=tradition,
            source=source,
            period=entry.get("time_period") or entry.get("period"),
        )


# =============================================================================
# EXTRACTION FUNCTIONS
# =============================================================================


def extract_from_passage(
    client,
    passage: CorpusPassage,
    model: str = "claude-sonnet-4-20250514",
) -> list[MoralBond]:
    """Extract moral bonds from a single corpus passage."""
    if not passage.text or len(passage.text.strip()) < 10:
        return []

    try:
        bonds = extract_bonds_sync(
            client=client,
            text=passage.text,
            language=passage.language,
            tradition=passage.tradition,
            model=model,
        )

        # Add source metadata
        for bond in bonds:
            bond.source_text_id = passage.id

        return bonds
    except Exception as e:
        print(f"Warning: Extraction failed for {passage.id}: {e}")
        return []


def extract_from_corpus_cache(
    corpus_cache: dict,
    client,
    sample_size: Optional[int] = None,
    languages: Optional[list[str]] = None,
    model: str = "claude-sonnet-4-20250514",
    delay: float = 0.5,
    verbose: bool = True,
) -> list[MoralBond]:
    """
    Extract moral bonds from a corpus cache dictionary.

    Args:
        corpus_cache: Dict mapping (language, period) -> list of text entries
        client: Anthropic client
        sample_size: Max passages to process (None = all)
        languages: Filter to specific languages (None = all)
        model: Claude model to use
        delay: Delay between API calls (rate limiting)
        verbose: Print progress

    Returns:
        List of all extracted MoralBond objects
    """
    all_bonds = []
    passages_processed = 0
    total_passages = 0

    # Flatten corpus cache to list of passages
    passages = []
    for key, entries in corpus_cache.items():
        if isinstance(key, tuple):
            lang, period = key
        else:
            lang = key
            period = None

        # Filter by language if specified
        if languages and get_language_canonical(lang) not in languages:
            continue

        for entry in entries:
            if isinstance(entry, str):
                entry = {"text": entry, "language": lang}
            if isinstance(entry, dict):
                entry["language"] = entry.get("language", lang)
                if period:
                    entry["time_period"] = entry.get("time_period", period)
                passages.append(entry)

    total_passages = len(passages)
    if sample_size:
        # Sample evenly across the corpus
        import random

        random.seed(42)
        passages = random.sample(passages, min(sample_size, len(passages)))

    if verbose:
        print(f"Extracting from {len(passages)} passages (total available: {total_passages})")

    for i, entry in enumerate(passages):
        passage = CorpusPassage.from_corpus_entry(entry)

        if verbose and i % 10 == 0:
            print(f"  Processing {i + 1}/{len(passages)}: {passage.tradition} ({passage.language})")

        bonds = extract_from_passage(client, passage, model=model)
        all_bonds.extend(bonds)
        passages_processed += 1

        # Rate limiting
        if delay > 0:
            time.sleep(delay)

    if verbose:
        print(f"\nExtraction complete:")
        print(f"  Passages processed: {passages_processed}")
        print(f"  Bonds extracted: {len(all_bonds)}")

    return all_bonds


async def extract_from_passage_async(
    client,
    passage: CorpusPassage,
    model: str = "claude-sonnet-4-20250514",
) -> list[MoralBond]:
    """Async version of passage extraction."""
    if not passage.text or len(passage.text.strip()) < 10:
        return []

    try:
        message = await client.messages.create(
            model=model,
            max_tokens=2000,
            system=EXTRACTION_SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": create_extraction_prompt(
                        passage.text, passage.language, passage.tradition
                    ),
                }
            ],
        )

        response_text = message.content[0].text
        bonds = parse_extraction_response(response_text, passage.language, passage.tradition)

        for bond in bonds:
            bond.source_text_id = passage.id

        return bonds
    except Exception as e:
        print(f"Warning: Async extraction failed for {passage.id}: {e}")
        return []


async def extract_bonds_batch_async(
    client,
    passages: list[CorpusPassage],
    batch_size: int = 5,
    delay: float = 1.0,
    model: str = "claude-sonnet-4-20250514",
    verbose: bool = True,
) -> list[MoralBond]:
    """
    Extract bonds from multiple passages with batching and rate limiting.

    Args:
        client: Anthropic async client
        passages: List of CorpusPassage objects
        batch_size: Number of concurrent requests
        delay: Delay between batches
        model: Claude model to use
        verbose: Print progress

    Returns:
        List of all extracted MoralBond objects
    """
    all_bonds = []

    for i in range(0, len(passages), batch_size):
        batch = passages[i : i + batch_size]

        if verbose:
            print(
                f"Processing batch {i // batch_size + 1}/{(len(passages) + batch_size - 1) // batch_size}"
            )

        # Process batch concurrently
        tasks = [extract_from_passage_async(client, passage, model=model) for passage in batch]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, list):
                all_bonds.extend(result)
            elif isinstance(result, Exception):
                print(f"  Batch item failed: {result}")

        # Rate limiting between batches
        if delay > 0 and i + batch_size < len(passages):
            await asyncio.sleep(delay)

    return all_bonds


# =============================================================================
# SAVE/LOAD UTILITIES
# =============================================================================


def save_bonds_jsonl(bonds: list[MoralBond], path: Path) -> None:
    """Save bonds to JSONL format."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        for bond in bonds:
            f.write(json.dumps(bond.to_dict(), ensure_ascii=False) + "\n")


def load_bonds_jsonl(path: Path) -> list[MoralBond]:
    """Load bonds from JSONL format."""
    bonds = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                bonds.append(MoralBond.from_dict(data))
    return bonds


def save_extraction_results(
    bonds: list[MoralBond],
    output_dir: Path,
    metadata: Optional[dict] = None,
) -> None:
    """Save extraction results with metadata."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save bonds
    save_bonds_jsonl(bonds, output_dir / "bonds.jsonl")

    # Save metadata
    meta = metadata or {}
    meta["total_bonds"] = len(bonds)
    meta["traditions"] = list(set(b.source_tradition for b in bonds if b.source_tradition))
    meta["languages"] = list(set(b.source_language for b in bonds if b.source_language))

    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


# =============================================================================
# STATISTICS
# =============================================================================


def compute_extraction_stats(bonds: list[MoralBond]) -> dict:
    """Compute statistics about extracted bonds."""
    from collections import Counter

    stats = {
        "total_bonds": len(bonds),
        "by_tradition": Counter(b.source_tradition for b in bonds if b.source_tradition),
        "by_language": Counter(b.source_language for b in bonds if b.source_language),
        "by_bond_type": Counter(b.bond_type.value for b in bonds),
        "by_context": Counter(b.context.value for b in bonds),
        "avg_confidence": sum(b.confidence for b in bonds) / len(bonds) if bonds else 0,
    }
    return stats


def print_extraction_stats(bonds: list[MoralBond]) -> None:
    """Print formatted extraction statistics."""
    stats = compute_extraction_stats(bonds)

    print("=" * 60)
    print("EXTRACTION STATISTICS")
    print("=" * 60)
    print(f"\nTotal bonds: {stats['total_bonds']}")
    print(f"Average confidence: {stats['avg_confidence']:.2f}")

    print("\nBy Tradition:")
    for tradition, count in stats["by_tradition"].most_common():
        print(f"  {tradition:20s}: {count:5d}")

    print("\nBy Language:")
    for lang, count in stats["by_language"].most_common():
        print(f"  {lang:20s}: {count:5d}")

    print("\nBy Bond Type:")
    for bond_type, count in stats["by_bond_type"].most_common():
        print(f"  {bond_type:20s}: {count:5d}")

    print("=" * 60)
