# Copyright (c) 2026 Andrew H. Bond
# Department of Computer Engineering, San Jose State University
# Licensed under the MIT License. See LICENSE file for details.

"""Tests for corpus extraction module."""

import pytest

from src.bip.corpus_extraction import (
    CorpusPassage,
    compute_extraction_stats,
    get_language_canonical,
    get_tradition_from_source,
    load_bonds_jsonl,
    save_bonds_jsonl,
)
from src.bip.moral_structure import (
    ActionCategory,
    BondType,
    ContextType,
    ModalStrength,
    MoralBond,
    RoleType,
)


class TestLanguageMapping:
    """Tests for language canonicalization."""

    def test_hebrew_mapping(self):
        assert get_language_canonical("hebrew") == "hebrew"
        assert get_language_canonical("Hebrew") == "hebrew"

    def test_chinese_mapping(self):
        assert get_language_canonical("classical_chinese") == "classical_chinese"
        assert get_language_canonical("chinese") == "classical_chinese"

    def test_unknown_language(self):
        assert get_language_canonical("klingon") == "klingon"


class TestTraditionMapping:
    """Tests for tradition detection from source."""

    def test_biblical_sources(self):
        assert get_tradition_from_source("tanakh/genesis") == "biblical"
        assert get_tradition_from_source("Tanakh") == "biblical"

    def test_rabbinic_sources(self):
        assert get_tradition_from_source("mishnah/avot") == "rabbinic"
        assert get_tradition_from_source("talmud/bava_kamma") == "rabbinic"

    def test_confucian_sources(self):
        assert get_tradition_from_source("analects") == "confucian"
        assert get_tradition_from_source("mengzi/1") == "confucian"

    def test_buddhist_sources(self):
        assert get_tradition_from_source("dhammapada") == "buddhist"
        assert get_tradition_from_source("suttacentral/mn1") == "buddhist"

    def test_quranic_sources(self):
        assert get_tradition_from_source("quran/2:255") == "quranic"
        assert get_tradition_from_source("tanzil") == "quranic"

    def test_unknown_source(self):
        assert get_tradition_from_source("random_text") == "unknown"


class TestCorpusPassage:
    """Tests for CorpusPassage creation."""

    def test_from_dict_entry(self):
        entry = {
            "id": "test_001",
            "text": "Honor your father and mother.",
            "language": "hebrew",
            "source": "tanakh/exodus",
        }
        passage = CorpusPassage.from_corpus_entry(entry)

        assert passage.id == "test_001"
        assert passage.text == "Honor your father and mother."
        assert passage.language == "hebrew"
        assert passage.tradition == "biblical"

    def test_from_entry_with_text_original(self):
        entry = {
            "text_original": "כבד את אביך ואת אמך",
            "language": "hebrew",
            "source": "tanakh",
        }
        passage = CorpusPassage.from_corpus_entry(entry)
        assert passage.text == "כבד את אביך ואת אמך"

    def test_generates_id_if_missing(self):
        entry = {"text": "Test passage", "language": "english"}
        passage = CorpusPassage.from_corpus_entry(entry)
        assert passage.id.startswith("passage_")


class TestBondSerialization:
    """Tests for bond save/load."""

    def test_save_and_load_bonds(self, tmp_path):
        bonds = [
            MoralBond(
                bond_type=BondType.OBLIGATION,
                agent_role=RoleType.CHILD,
                patient_role=RoleType.PARENT,
                action=ActionCategory.HONOR,
                modal_strength=ModalStrength.MUST,
                context=ContextType.FAMILY,
                source_language="hebrew",
                source_tradition="biblical",
            ),
            MoralBond(
                bond_type=BondType.PROHIBITION,
                agent_role=RoleType.AGENT,
                patient_role=RoleType.PATIENT,
                action=ActionCategory.KILL,
                modal_strength=ModalStrength.MUST_NOT,
                context=ContextType.GENERIC,
                source_language="hebrew",
                source_tradition="biblical",
            ),
        ]

        path = tmp_path / "bonds.jsonl"
        save_bonds_jsonl(bonds, path)
        loaded = load_bonds_jsonl(path)

        assert len(loaded) == 2
        assert loaded[0].bond_type == BondType.OBLIGATION
        assert loaded[0].agent_role == RoleType.CHILD
        assert loaded[1].bond_type == BondType.PROHIBITION

    def test_handles_unicode(self, tmp_path):
        bond = MoralBond(
            bond_type=BondType.OBLIGATION,
            agent_role=RoleType.CHILD,
            patient_role=RoleType.PARENT,
            action=ActionCategory.HONOR,
            condition="כבד את אביך",  # Hebrew text
            source_language="hebrew",
            source_tradition="biblical",
        )

        path = tmp_path / "unicode_bonds.jsonl"
        save_bonds_jsonl([bond], path)
        loaded = load_bonds_jsonl(path)

        assert loaded[0].condition == "כבד את אביך"


class TestExtractionStats:
    """Tests for extraction statistics."""

    def test_compute_stats(self):
        bonds = [
            MoralBond(
                bond_type=BondType.OBLIGATION,
                agent_role=RoleType.CHILD,
                patient_role=RoleType.PARENT,
                action=ActionCategory.HONOR,
                source_language="hebrew",
                source_tradition="biblical",
                confidence=0.9,
            ),
            MoralBond(
                bond_type=BondType.OBLIGATION,
                agent_role=RoleType.CHILD,
                patient_role=RoleType.PARENT,
                action=ActionCategory.HONOR,
                source_language="arabic",
                source_tradition="quranic",
                confidence=0.8,
            ),
            MoralBond(
                bond_type=BondType.PROHIBITION,
                agent_role=RoleType.AGENT,
                patient_role=RoleType.PATIENT,
                action=ActionCategory.KILL,
                source_language="hebrew",
                source_tradition="biblical",
                confidence=1.0,
            ),
        ]

        stats = compute_extraction_stats(bonds)

        assert stats["total_bonds"] == 3
        assert stats["by_tradition"]["biblical"] == 2
        assert stats["by_tradition"]["quranic"] == 1
        assert stats["by_language"]["hebrew"] == 2
        assert stats["by_bond_type"]["obligation"] == 2
        assert stats["by_bond_type"]["prohibition"] == 1
        assert stats["avg_confidence"] == pytest.approx(0.9, rel=0.01)

    def test_empty_bonds(self):
        stats = compute_extraction_stats([])
        assert stats["total_bonds"] == 0
        assert stats["avg_confidence"] == 0


# =============================================================================
# SAMPLE PASSAGES FOR INTEGRATION TESTING
# =============================================================================

SAMPLE_PASSAGES = {
    "biblical": {
        "text": "Honor your father and your mother, that your days may be long in the land that the LORD your God is giving you.",
        "language": "hebrew",
        "source": "tanakh/exodus/20",
        "expected_bonds": [
            {
                "bond_type": "obligation",
                "agent_role": "child",
                "patient_role": "parent",
                "action": "honor",
            }
        ],
    },
    "confucian": {
        "text": "The Master said: In serving your parents, you may gently remonstrate. If you see that they are not inclined to follow your advice, remain reverent and do not oppose them.",
        "language": "classical_chinese",
        "source": "analects/4",
        "expected_bonds": [
            {
                "bond_type": "obligation",
                "agent_role": "child",
                "patient_role": "parent",
                "action": "obey",
            }
        ],
    },
    "dharma": {
        "text": "The son should serve his parents: by rising to greet them, by serving them, by obeying them, by providing for their needs.",
        "language": "sanskrit",
        "source": "itihasa/dharma",
        "expected_bonds": [
            {
                "bond_type": "obligation",
                "agent_role": "child",
                "patient_role": "parent",
                "action": "honor",
            }
        ],
    },
    "quranic": {
        "text": "And your Lord has decreed that you worship none but Him, and that you be dutiful to your parents.",
        "language": "arabic",
        "source": "quran/17:23",
        "expected_bonds": [
            {
                "bond_type": "obligation",
                "agent_role": "child",
                "patient_role": "parent",
                "action": "honor",
            }
        ],
    },
    "buddhist": {
        "text": "One should not kill any living being, nor cause others to kill, nor approve of others killing.",
        "language": "pali",
        "source": "dhammapada",
        "expected_bonds": [
            {
                "bond_type": "prohibition",
                "agent_role": "agent",
                "patient_role": "patient",
                "action": "kill",
            }
        ],
    },
}


class TestSamplePassages:
    """Tests using sample passages (no API calls)."""

    def test_all_samples_have_required_fields(self):
        for name, sample in SAMPLE_PASSAGES.items():
            assert "text" in sample, f"{name} missing text"
            assert "language" in sample, f"{name} missing language"
            assert "source" in sample, f"{name} missing source"
            assert "expected_bonds" in sample, f"{name} missing expected_bonds"

    def test_sample_passages_create_valid_corpus_passages(self):
        for name, sample in SAMPLE_PASSAGES.items():
            passage = CorpusPassage.from_corpus_entry(sample)
            assert passage.text == sample["text"]
            assert len(passage.text) > 10, f"{name} text too short"


@pytest.mark.integration
class TestLiveExtraction:
    """
    Integration tests that require API calls.

    Run with: pytest -m integration tests/test_corpus_extraction.py
    Requires ANTHROPIC_API_KEY environment variable.
    """

    @pytest.fixture
    def anthropic_client(self):
        """Create Anthropic client if API key available."""
        import os

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            pytest.skip("ANTHROPIC_API_KEY not set")

        import anthropic

        return anthropic.Anthropic()

    def test_extract_biblical_passage(self, anthropic_client):
        """Test extraction from biblical passage."""
        from src.bip.corpus_extraction import extract_from_passage

        sample = SAMPLE_PASSAGES["biblical"]
        passage = CorpusPassage.from_corpus_entry(sample)
        bonds = extract_from_passage(anthropic_client, passage)

        assert len(bonds) >= 1
        # Should find an obligation bond
        obligation_bonds = [b for b in bonds if b.bond_type == BondType.OBLIGATION]
        assert len(obligation_bonds) >= 1

    def test_extract_multiple_traditions(self, anthropic_client):
        """Test extraction from multiple traditions."""
        from src.bip.corpus_extraction import extract_from_passage

        for name in ["biblical", "confucian", "quranic"]:
            sample = SAMPLE_PASSAGES[name]
            passage = CorpusPassage.from_corpus_entry(sample)
            bonds = extract_from_passage(anthropic_client, passage)

            assert len(bonds) >= 1, f"No bonds extracted from {name}"
            assert bonds[0].source_tradition is not None
