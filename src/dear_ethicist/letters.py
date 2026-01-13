# Copyright (c) 2026 Andrew H. Bond and Claude Opus 4.5
# Department of Computer Engineering, San Jose State University
# Licensed under the MIT License. See LICENSE file for details.

"""
Letter bank and generation for Dear Ethicist.

Loads letter templates from YAML files in the letters/ directory.
"""

from pathlib import Path

import yaml

from dear_ethicist.models import (
    HohfeldianState,
    Letter,
    LetterVoice,
    Party,
    Protocol,
)

# Path to letters directory
LETTERS_DIR = Path(__file__).parent / "letters"


class LetterBank:
    """Repository of letter templates organized by protocol."""

    def __init__(self):
        self._letters: dict[str, Letter] = {}
        self._by_protocol: dict[Protocol, list[str]] = {p: [] for p in Protocol}

    def add(self, letter: Letter) -> None:
        """Add a letter to the bank."""
        self._letters[letter.letter_id] = letter
        self._by_protocol[letter.protocol].append(letter.letter_id)

    def get(self, letter_id: str) -> Letter | None:
        """Get a letter by ID."""
        return self._letters.get(letter_id)

    def get_by_protocol(self, protocol: Protocol) -> list[Letter]:
        """Get all letters for a protocol."""
        return [self._letters[lid] for lid in self._by_protocol[protocol]]

    def list_ids(self, protocol: Protocol | None = None) -> list[str]:
        """List all letter IDs, optionally filtered by protocol."""
        if protocol:
            return self._by_protocol[protocol].copy()
        return list(self._letters.keys())

    def count(self, protocol: Protocol | None = None) -> int:
        """Count letters, optionally filtered by protocol."""
        if protocol:
            return len(self._by_protocol[protocol])
        return len(self._letters)

    def get_all_letters(self) -> list[Letter]:
        """Get all letters in the bank."""
        return list(self._letters.values())


def parse_letter_from_dict(data: dict) -> Letter:
    """Parse a letter from a YAML dictionary."""
    # Parse protocol
    protocol = Protocol(data["protocol"])

    # Parse voice
    voice_str = data.get("voice", "genuinely_stuck")
    voice = LetterVoice(voice_str)

    # Parse parties
    parties = []
    for p in data.get("parties", []):
        expected = None
        if p.get("expected_state"):
            expected = HohfeldianState(p["expected_state"])
        parties.append(
            Party(
                name=p["name"],
                role=p["role"],
                is_writer=p.get("is_writer", False),
                expected_state=expected,
            )
        )

    return Letter(
        letter_id=data["letter_id"],
        protocol=protocol,
        protocol_params=data.get("protocol_params", {}),
        signoff=data["signoff"],
        subject=data["subject"],
        body=data["body"].strip(),
        voice=voice,
        parties=parties,
        questions=data.get("questions", []),
    )


def load_letters_from_yaml(filepath: Path) -> list[Letter]:
    """Load letters from a YAML file."""
    letters = []
    with open(filepath, encoding="utf-8") as f:
        data = yaml.safe_load(f)
        if data:
            for item in data:
                try:
                    letter = parse_letter_from_dict(item)
                    letters.append(letter)
                except Exception as e:
                    print(f"Warning: Failed to parse letter in {filepath}: {e}")
    return letters


def load_all_letters(
    letters_dir: Path = LETTERS_DIR, include_archive: bool = False
) -> list[Letter]:
    """Load all letters from YAML files in the letters directory.

    Args:
        letters_dir: Directory containing letter YAML files.
        include_archive: If True, recursively load from subdirectories (e.g., archive/).
    """
    letters = []
    if letters_dir.exists():
        pattern = "**/*.yaml" if include_archive else "*.yaml"
        for yaml_file in letters_dir.glob(pattern):
            letters.extend(load_letters_from_yaml(yaml_file))
        pattern = "**/*.yml" if include_archive else "*.yml"
        for yml_file in letters_dir.glob(pattern):
            letters.extend(load_letters_from_yaml(yml_file))
    return letters


def create_default_letter_bank() -> LetterBank:
    """Create the default letter bank from YAML files."""
    bank = LetterBank()

    for letter in load_all_letters():
        bank.add(letter)

    return bank


def get_response_options(letter: Letter) -> list[dict]:
    """
    Generate guided response options for a letter.

    Returns options that map to Hohfeldian classifications.
    """
    writer = letter.get_writer()
    other = letter.get_other_party()

    if not writer or not other:
        return []

    options = []

    # Options for the other party's status
    options.append(
        {
            "category": f"{other.name}'s situation",
            "choices": [
                {
                    "label": f"{other.name} is obligated (promise stands)",
                    "state": HohfeldianState.O,
                    "party": other.name,
                },
                {
                    "label": f"{other.name} is free to choose",
                    "state": HohfeldianState.L,
                    "party": other.name,
                },
            ],
        }
    )

    # Options for the writer's status
    options.append(
        {
            "category": "Your claim",
            "choices": [
                {
                    "label": "You have a right to expect this",
                    "state": HohfeldianState.C,
                    "party": writer.name,
                },
                {
                    "label": "You can't demand this",
                    "state": HohfeldianState.N,
                    "party": writer.name,
                },
            ],
        }
    )

    return options
