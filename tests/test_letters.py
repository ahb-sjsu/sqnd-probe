# Copyright (c) 2026 Andrew H. Bond and Claude Opus 4.5
# Department of Computer Engineering, San Jose State University
# Licensed under the MIT License. See LICENSE file for details.

"""Tests for Dear Ethicist letter loading and parsing."""

import tempfile
from pathlib import Path

from dear_ethicist.letters import (
    LetterBank,
    create_default_letter_bank,
    get_response_options,
    load_all_letters,
    load_letters_from_yaml,
    parse_letter_from_dict,
)
from dear_ethicist.models import (
    HohfeldianState,
    Letter,
    LetterVoice,
    Party,
    Protocol,
)


class TestParseLetterFromDict:
    """Tests for parse_letter_from_dict function."""

    def test_parse_minimal_letter(self):
        """Parse a letter with minimal required fields."""
        data = {
            "letter_id": "test_letter",
            "protocol": "CORRELATIVE",
            "signoff": "TEST PERSON",
            "subject": "Test subject",
            "body": "Dear Ethicist,\n\nThis is a test.",
        }
        letter = parse_letter_from_dict(data)

        assert letter.letter_id == "test_letter"
        assert letter.protocol == Protocol.CORRELATIVE
        assert letter.signoff == "TEST PERSON"
        assert letter.subject == "Test subject"
        assert "This is a test" in letter.body

    def test_parse_letter_with_parties(self):
        """Parse a letter with party definitions."""
        data = {
            "letter_id": "party_test",
            "protocol": "CORRELATIVE",
            "signoff": "TESTER",
            "subject": "Party test",
            "body": "Test body",
            "parties": [
                {"name": "You", "role": "letter_writer", "is_writer": True},
                {"name": "Friend", "role": "friend", "expected_state": "O"},
            ],
        }
        letter = parse_letter_from_dict(data)

        assert len(letter.parties) == 2
        assert letter.parties[0].name == "You"
        assert letter.parties[0].is_writer is True
        assert letter.parties[1].name == "Friend"
        assert letter.parties[1].expected_state == HohfeldianState.O

    def test_parse_letter_with_voice(self):
        """Parse a letter with voice specification."""
        data = {
            "letter_id": "voice_test",
            "protocol": "GATE_DETECTION",
            "signoff": "AGGRIEVED",
            "subject": "Voice test",
            "body": "Test body",
            "voice": "aggrieved",
        }
        letter = parse_letter_from_dict(data)

        assert letter.voice == LetterVoice.AGGRIEVED

    def test_parse_letter_with_protocol_params(self):
        """Parse a letter with protocol parameters."""
        data = {
            "letter_id": "params_test",
            "protocol": "GATE_DETECTION",
            "protocol_params": {"level": 5, "trigger": "only_if_convenient"},
            "signoff": "TESTER",
            "subject": "Params test",
            "body": "Test body",
        }
        letter = parse_letter_from_dict(data)

        assert letter.protocol_params["level"] == 5
        assert letter.protocol_params["trigger"] == "only_if_convenient"

    def test_parse_letter_with_questions(self):
        """Parse a letter with explicit questions."""
        data = {
            "letter_id": "questions_test",
            "protocol": "CORRELATIVE",
            "signoff": "CURIOUS",
            "subject": "Questions test",
            "body": "Test body",
            "questions": [
                "Am I obligated?",
                "Do they have a right?",
            ],
        }
        letter = parse_letter_from_dict(data)

        assert len(letter.questions) == 2
        assert "Am I obligated?" in letter.questions

    def test_parse_all_protocols(self):
        """Parse letters with each protocol type."""
        protocols = [
            "GATE_DETECTION",
            "CORRELATIVE",
            "PATH_DEPENDENCE",
            "CONTEXT_SALIENCE",
            "PHASE_TRANSITION",
        ]
        for proto in protocols:
            data = {
                "letter_id": f"proto_{proto}",
                "protocol": proto,
                "signoff": "TESTER",
                "subject": "Test",
                "body": "Test body",
            }
            letter = parse_letter_from_dict(data)
            assert letter.protocol == Protocol(proto)


class TestLoadLettersFromYaml:
    """Tests for YAML loading."""

    def test_load_single_letter(self):
        """Load a YAML file with a single letter."""
        yaml_content = """
- letter_id: yaml_test
  protocol: CORRELATIVE
  signoff: YAML TESTER
  subject: YAML Test
  body: |
    Dear Ethicist,

    This is from YAML.
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8"
        ) as f:
            f.write(yaml_content)
            f.flush()
            letters = load_letters_from_yaml(Path(f.name))

        assert len(letters) == 1
        assert letters[0].letter_id == "yaml_test"

    def test_load_multiple_letters(self):
        """Load a YAML file with multiple letters."""
        yaml_content = """
- letter_id: letter1
  protocol: CORRELATIVE
  signoff: FIRST
  subject: First
  body: Body 1

- letter_id: letter2
  protocol: GATE_DETECTION
  signoff: SECOND
  subject: Second
  body: Body 2
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8"
        ) as f:
            f.write(yaml_content)
            f.flush()
            letters = load_letters_from_yaml(Path(f.name))

        assert len(letters) == 2
        assert letters[0].letter_id == "letter1"
        assert letters[1].letter_id == "letter2"

    def test_load_empty_file(self):
        """Load an empty YAML file."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8"
        ) as f:
            f.write("")
            f.flush()
            letters = load_letters_from_yaml(Path(f.name))

        assert len(letters) == 0

    def test_load_with_invalid_letter_skipped(self, capsys):
        """Invalid letters should be skipped with warning."""
        yaml_content = """
- letter_id: valid
  protocol: CORRELATIVE
  signoff: VALID
  subject: Valid
  body: Valid body

- letter_id: invalid
  protocol: NOT_A_PROTOCOL
  signoff: INVALID
  subject: Invalid
  body: Invalid body
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8"
        ) as f:
            f.write(yaml_content)
            f.flush()
            letters = load_letters_from_yaml(Path(f.name))

        # Only valid letter loaded
        assert len(letters) == 1
        assert letters[0].letter_id == "valid"

        # Warning printed
        captured = capsys.readouterr()
        assert "Warning" in captured.out


class TestLoadAllLetters:
    """Tests for load_all_letters function."""

    def test_load_from_directory(self):
        """Load letters from a directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create two YAML files
            for i in range(2):
                yaml_content = f"""
- letter_id: dir_letter_{i}
  protocol: CORRELATIVE
  signoff: DIR TEST {i}
  subject: Dir Test {i}
  body: Body {i}
"""
                (Path(tmpdir) / f"test_{i}.yaml").write_text(yaml_content)

            letters = load_all_letters(Path(tmpdir))

        assert len(letters) == 2

    def test_load_with_yml_extension(self):
        """Load .yml files as well as .yaml."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_content = """
- letter_id: yml_test
  protocol: CORRELATIVE
  signoff: YML
  subject: YML Test
  body: Body
"""
            (Path(tmpdir) / "test.yml").write_text(yaml_content)
            letters = load_all_letters(Path(tmpdir))

        assert len(letters) == 1

    def test_load_nonexistent_directory(self):
        """Return empty list for nonexistent directory."""
        letters = load_all_letters(Path("/nonexistent/path"))
        assert len(letters) == 0

    def test_load_with_archive(self):
        """Test include_archive parameter."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create main file
            (Path(tmpdir) / "main.yaml").write_text(
                """
- letter_id: main
  protocol: CORRELATIVE
  signoff: MAIN
  subject: Main
  body: Body
"""
            )
            # Create archive subdirectory
            archive_dir = Path(tmpdir) / "archive"
            archive_dir.mkdir()
            (archive_dir / "archive.yaml").write_text(
                """
- letter_id: archived
  protocol: CORRELATIVE
  signoff: ARCHIVED
  subject: Archived
  body: Body
"""
            )

            # Without archive
            letters_no_archive = load_all_letters(Path(tmpdir), include_archive=False)
            assert len(letters_no_archive) == 1

            # With archive
            letters_with_archive = load_all_letters(Path(tmpdir), include_archive=True)
            assert len(letters_with_archive) == 2


class TestLetterBank:
    """Tests for LetterBank class."""

    def test_add_and_get(self):
        """Add letters and retrieve them."""
        bank = LetterBank()
        letter = Letter(
            letter_id="bank_test",
            protocol=Protocol.CORRELATIVE,
            signoff="BANK",
            subject="Bank Test",
            body="Body",
        )
        bank.add(letter)

        retrieved = bank.get("bank_test")
        assert retrieved is not None
        assert retrieved.letter_id == "bank_test"

    def test_get_nonexistent(self):
        """Get returns None for nonexistent ID."""
        bank = LetterBank()
        assert bank.get("nonexistent") is None

    def test_get_by_protocol(self):
        """Get letters filtered by protocol."""
        bank = LetterBank()
        bank.add(
            Letter(
                letter_id="corr1",
                protocol=Protocol.CORRELATIVE,
                signoff="C1",
                subject="Corr 1",
                body="Body",
            )
        )
        bank.add(
            Letter(
                letter_id="gate1",
                protocol=Protocol.GATE_DETECTION,
                signoff="G1",
                subject="Gate 1",
                body="Body",
            )
        )

        corr_letters = bank.get_by_protocol(Protocol.CORRELATIVE)
        assert len(corr_letters) == 1
        assert corr_letters[0].letter_id == "corr1"

    def test_list_ids(self):
        """List all letter IDs."""
        bank = LetterBank()
        bank.add(
            Letter(
                letter_id="id1",
                protocol=Protocol.CORRELATIVE,
                signoff="S1",
                subject="S1",
                body="Body",
            )
        )
        bank.add(
            Letter(
                letter_id="id2",
                protocol=Protocol.CORRELATIVE,
                signoff="S2",
                subject="S2",
                body="Body",
            )
        )

        ids = bank.list_ids()
        assert "id1" in ids
        assert "id2" in ids

    def test_list_ids_by_protocol(self):
        """List IDs filtered by protocol."""
        bank = LetterBank()
        bank.add(
            Letter(
                letter_id="corr",
                protocol=Protocol.CORRELATIVE,
                signoff="C",
                subject="C",
                body="Body",
            )
        )
        bank.add(
            Letter(
                letter_id="gate",
                protocol=Protocol.GATE_DETECTION,
                signoff="G",
                subject="G",
                body="Body",
            )
        )

        corr_ids = bank.list_ids(Protocol.CORRELATIVE)
        assert corr_ids == ["corr"]

    def test_count(self):
        """Count letters in bank."""
        bank = LetterBank()
        assert bank.count() == 0

        bank.add(
            Letter(
                letter_id="l1",
                protocol=Protocol.CORRELATIVE,
                signoff="S",
                subject="S",
                body="Body",
            )
        )
        assert bank.count() == 1

    def test_count_by_protocol(self):
        """Count letters by protocol."""
        bank = LetterBank()
        bank.add(
            Letter(
                letter_id="c1",
                protocol=Protocol.CORRELATIVE,
                signoff="S",
                subject="S",
                body="Body",
            )
        )
        bank.add(
            Letter(
                letter_id="c2",
                protocol=Protocol.CORRELATIVE,
                signoff="S",
                subject="S",
                body="Body",
            )
        )
        bank.add(
            Letter(
                letter_id="g1",
                protocol=Protocol.GATE_DETECTION,
                signoff="S",
                subject="S",
                body="Body",
            )
        )

        assert bank.count(Protocol.CORRELATIVE) == 2
        assert bank.count(Protocol.GATE_DETECTION) == 1


class TestCreateDefaultLetterBank:
    """Tests for create_default_letter_bank function."""

    def test_loads_letters(self):
        """Default bank loads letters from package."""
        bank = create_default_letter_bank()
        assert bank.count() > 0

    def test_has_multiple_protocols(self):
        """Default bank has letters from multiple protocols."""
        bank = create_default_letter_bank()
        protocols_with_letters = [p for p in Protocol if bank.count(p) > 0]
        assert len(protocols_with_letters) >= 2


class TestGetResponseOptions:
    """Tests for get_response_options function."""

    def test_generates_options_for_two_parties(self):
        """Generate options when letter has writer and other party."""
        letter = Letter(
            letter_id="options_test",
            protocol=Protocol.CORRELATIVE,
            signoff="TEST",
            subject="Test",
            body="Body",
            parties=[
                Party(name="You", role="writer", is_writer=True),
                Party(name="Friend", role="friend"),
            ],
        )
        options = get_response_options(letter)

        assert len(options) == 2
        assert options[0]["category"] == "Friend's situation"
        assert options[1]["category"] == "Your claim"

    def test_returns_empty_for_no_parties(self):
        """Return empty list when letter has no parties."""
        letter = Letter(
            letter_id="no_parties",
            protocol=Protocol.CORRELATIVE,
            signoff="TEST",
            subject="Test",
            body="Body",
            parties=[],
        )
        options = get_response_options(letter)
        assert options == []

    def test_options_contain_hohfeldian_states(self):
        """Options map to Hohfeldian states."""
        letter = Letter(
            letter_id="states_test",
            protocol=Protocol.CORRELATIVE,
            signoff="TEST",
            subject="Test",
            body="Body",
            parties=[
                Party(name="You", role="writer", is_writer=True),
                Party(name="Other", role="other"),
            ],
        )
        options = get_response_options(letter)

        # Check that O, C, L, N states are present
        all_states = set()
        for option_group in options:
            for choice in option_group["choices"]:
                all_states.add(choice["state"])

        assert HohfeldianState.O in all_states or HohfeldianState.L in all_states
        assert HohfeldianState.C in all_states or HohfeldianState.N in all_states
