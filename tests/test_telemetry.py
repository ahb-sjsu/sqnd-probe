# Copyright (c) 2026 Andrew H. Bond
# Department of Computer Engineering, San Jose State University
# Licensed under the MIT License. See LICENSE file for details.

"""Tests for Dear Ethicist telemetry logging and analysis."""

import json
import tempfile
from pathlib import Path

import pytest

from dear_ethicist.models import (
    HohfeldianState,
    Protocol,
    TrialRecord,
    Verdict,
)
from dear_ethicist.telemetry import (
    TelemetryLogger,
    compute_bond_index,
    load_session,
)


class TestTelemetryLogger:
    """Tests for TelemetryLogger class."""

    def test_creates_output_directory(self):
        """Logger creates output directory if it doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            new_dir = Path(tmpdir) / "new_subdir"
            _logger = TelemetryLogger(new_dir)  # noqa: F841
            assert new_dir.exists()

    def test_generates_session_id(self):
        """Logger generates session ID if not provided."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = TelemetryLogger(Path(tmpdir))
            assert logger.session_id is not None
            assert len(logger.session_id) == 8

    def test_uses_provided_session_id(self):
        """Logger uses provided session ID."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = TelemetryLogger(Path(tmpdir), session_id="custom123")
            assert logger.session_id == "custom123"

    def test_log_trial(self):
        """Log a trial record to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = TelemetryLogger(Path(tmpdir))

            record = TrialRecord(
                session_id=logger.session_id,
                letter_id="test_letter",
                day=1,
                week=1,
                protocol=Protocol.CORRELATIVE,
                protocol_params={},
                verdicts=[
                    Verdict(
                        party_name="Test",
                        state=HohfeldianState.O,
                        expected=HohfeldianState.O,
                    )
                ],
            )
            logger.log_trial(record)

            # Verify file exists and contains data
            assert logger.filepath.exists()
            with open(logger.filepath) as f:
                line = f.readline()
                data = json.loads(line)
                assert data["letter_id"] == "test_letter"

    def test_log_multiple_trials(self):
        """Log multiple trial records."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = TelemetryLogger(Path(tmpdir))

            for i in range(3):
                record = TrialRecord(
                    session_id=logger.session_id,
                    letter_id=f"letter_{i}",
                    day=i,
                    week=1,
                    protocol=Protocol.CORRELATIVE,
                    protocol_params={},
                    verdicts=[],
                )
                logger.log_trial(record)

            # Verify all records written
            with open(logger.filepath) as f:
                lines = f.readlines()
                assert len(lines) == 3

    def test_get_session_records(self):
        """Retrieve records from current session."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = TelemetryLogger(Path(tmpdir))

            record = TrialRecord(
                session_id=logger.session_id,
                letter_id="test",
                day=1,
                week=1,
                protocol=Protocol.CORRELATIVE,
                protocol_params={},
                verdicts=[],
            )
            logger.log_trial(record)

            records = logger.get_session_records()
            assert len(records) == 1
            assert records[0].letter_id == "test"

    def test_get_session_records_empty(self):
        """Return empty list when no records exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = TelemetryLogger(Path(tmpdir))
            records = logger.get_session_records()
            assert records == []

    def test_compute_statistics_empty(self):
        """Statistics for empty session."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = TelemetryLogger(Path(tmpdir))
            stats = logger.compute_statistics()
            assert stats["total_trials"] == 0

    def test_compute_statistics_with_records(self):
        """Statistics for session with records."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = TelemetryLogger(Path(tmpdir))

            # Add some records
            for i in range(5):
                record = TrialRecord(
                    session_id=logger.session_id,
                    letter_id=f"letter_{i}",
                    day=i,
                    week=1,
                    protocol=Protocol.CORRELATIVE,
                    protocol_params={},
                    verdicts=[
                        Verdict(
                            party_name="Test",
                            state=HohfeldianState.O,
                            expected=HohfeldianState.O if i < 3 else HohfeldianState.L,
                        )
                    ],
                )
                logger.log_trial(record)

            stats = logger.compute_statistics()
            assert stats["total_trials"] == 5
            assert "by_protocol" in stats
            assert "CORRELATIVE" in stats["by_protocol"]

    def test_compute_statistics_accuracy(self):
        """Statistics correctly compute accuracy."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = TelemetryLogger(Path(tmpdir))

            # 2 correct, 1 incorrect
            records = [
                TrialRecord(
                    session_id=logger.session_id,
                    letter_id="l1",
                    day=1,
                    week=1,
                    protocol=Protocol.CORRELATIVE,
                    protocol_params={},
                    verdicts=[
                        Verdict(party_name="A", state=HohfeldianState.O, expected=HohfeldianState.O)
                    ],
                ),
                TrialRecord(
                    session_id=logger.session_id,
                    letter_id="l2",
                    day=1,
                    week=1,
                    protocol=Protocol.CORRELATIVE,
                    protocol_params={},
                    verdicts=[
                        Verdict(party_name="B", state=HohfeldianState.O, expected=HohfeldianState.O)
                    ],
                ),
                TrialRecord(
                    session_id=logger.session_id,
                    letter_id="l3",
                    day=1,
                    week=1,
                    protocol=Protocol.CORRELATIVE,
                    protocol_params={},
                    verdicts=[
                        Verdict(party_name="C", state=HohfeldianState.O, expected=HohfeldianState.L)
                    ],
                ),
            ]
            for r in records:
                logger.log_trial(r)

            stats = logger.compute_statistics()
            # 2 out of 3 correct
            assert stats["accuracy"] == pytest.approx(2 / 3, rel=0.01)


class TestLoadSession:
    """Tests for load_session function."""

    def test_load_session_file(self):
        """Load records from a session file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            record = TrialRecord(
                session_id="test",
                letter_id="letter1",
                day=1,
                week=1,
                protocol=Protocol.CORRELATIVE,
                protocol_params={},
                verdicts=[],
            )
            f.write(json.dumps(record.to_json()) + "\n")
            f.flush()

            records = load_session(Path(f.name))

        assert len(records) == 1
        assert records[0].letter_id == "letter1"

    def test_load_multiple_records(self):
        """Load multiple records from file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for i in range(5):
                record = TrialRecord(
                    session_id="test",
                    letter_id=f"letter_{i}",
                    day=i,
                    week=1,
                    protocol=Protocol.CORRELATIVE,
                    protocol_params={},
                    verdicts=[],
                )
                f.write(json.dumps(record.to_json()) + "\n")
            f.flush()

            records = load_session(Path(f.name))

        assert len(records) == 5


class TestComputeBondIndex:
    """Tests for compute_bond_index function."""

    def test_perfect_bond_index(self):
        """Perfect alignment gives bond index 0."""
        records = [
            TrialRecord(
                session_id="test",
                letter_id="l1",
                day=1,
                week=1,
                protocol=Protocol.CORRELATIVE,
                protocol_params={},
                verdicts=[
                    Verdict(party_name="A", state=HohfeldianState.O, expected=HohfeldianState.O),
                    Verdict(party_name="B", state=HohfeldianState.C, expected=HohfeldianState.C),
                ],
            ),
        ]
        bond = compute_bond_index(records)
        assert bond == 0.0

    def test_all_defects(self):
        """All defects gives bond index 1/tau."""
        records = [
            TrialRecord(
                session_id="test",
                letter_id="l1",
                day=1,
                week=1,
                protocol=Protocol.CORRELATIVE,
                protocol_params={},
                verdicts=[
                    Verdict(party_name="A", state=HohfeldianState.O, expected=HohfeldianState.L),
                    Verdict(party_name="B", state=HohfeldianState.C, expected=HohfeldianState.N),
                ],
            ),
        ]
        bond = compute_bond_index(records, tau=1.0)
        assert bond == 1.0

    def test_half_defects(self):
        """Half defects gives bond index 0.5/tau."""
        records = [
            TrialRecord(
                session_id="test",
                letter_id="l1",
                day=1,
                week=1,
                protocol=Protocol.CORRELATIVE,
                protocol_params={},
                verdicts=[
                    Verdict(party_name="A", state=HohfeldianState.O, expected=HohfeldianState.O),
                    Verdict(party_name="B", state=HohfeldianState.C, expected=HohfeldianState.N),
                ],
            ),
        ]
        bond = compute_bond_index(records, tau=1.0)
        assert bond == 0.5

    def test_empty_records(self):
        """Empty records gives bond index 0."""
        bond = compute_bond_index([])
        assert bond == 0.0

    def test_no_expected_values(self):
        """Records without expected values give bond index 0."""
        records = [
            TrialRecord(
                session_id="test",
                letter_id="l1",
                day=1,
                week=1,
                protocol=Protocol.CORRELATIVE,
                protocol_params={},
                verdicts=[
                    Verdict(party_name="A", state=HohfeldianState.O, expected=None),
                ],
            ),
        ]
        bond = compute_bond_index(records)
        assert bond == 0.0

    def test_custom_tau(self):
        """Custom tau scales the bond index."""
        records = [
            TrialRecord(
                session_id="test",
                letter_id="l1",
                day=1,
                week=1,
                protocol=Protocol.CORRELATIVE,
                protocol_params={},
                verdicts=[
                    Verdict(party_name="A", state=HohfeldianState.O, expected=HohfeldianState.L),
                ],
            ),
        ]
        # With tau=1, bond=1
        bond1 = compute_bond_index(records, tau=1.0)
        # With tau=2, bond=0.5
        bond2 = compute_bond_index(records, tau=2.0)

        assert bond1 == 1.0
        assert bond2 == 0.5

    def test_multiple_records(self):
        """Bond index computed across multiple records."""
        records = [
            TrialRecord(
                session_id="test",
                letter_id="l1",
                day=1,
                week=1,
                protocol=Protocol.CORRELATIVE,
                protocol_params={},
                verdicts=[
                    Verdict(party_name="A", state=HohfeldianState.O, expected=HohfeldianState.O),
                ],
            ),
            TrialRecord(
                session_id="test",
                letter_id="l2",
                day=2,
                week=1,
                protocol=Protocol.CORRELATIVE,
                protocol_params={},
                verdicts=[
                    Verdict(party_name="B", state=HohfeldianState.L, expected=HohfeldianState.O),
                ],
            ),
        ]
        # 1 defect out of 2 verdicts = 0.5
        bond = compute_bond_index(records, tau=1.0)
        assert bond == 0.5
