"""
Telemetry logging for Dear Ethicist.

Records all player verdicts for SQND analysis.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional
from uuid import uuid4

from dear_ethicist.models import (
    GameState,
    Protocol,
    TrialRecord,
    Verdict,
)


class TelemetryLogger:
    """Logger for experimental telemetry."""

    def __init__(self, output_dir: Path, session_id: Optional[str] = None):
        self.output_dir = Path(output_dir)
        self.session_id = session_id or str(uuid4())[:8]
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.filepath = self.output_dir / f"{self.session_id}_{datetime.now():%Y%m%d_%H%M%S}.jsonl"

    def log_trial(self, record: TrialRecord) -> None:
        """Append a trial record to the telemetry file."""
        with open(self.filepath, "a") as f:
            f.write(json.dumps(record.to_json()) + "\n")

    def get_session_records(self) -> list[TrialRecord]:
        """Load all records from current session."""
        records = []
        if self.filepath.exists():
            with open(self.filepath) as f:
                for line in f:
                    data = json.loads(line)
                    records.append(TrialRecord(**data))
        return records

    def compute_statistics(self) -> dict:
        """Compute summary statistics for the session."""
        records = self.get_session_records()

        if not records:
            return {"total_trials": 0}

        total = len(records)
        correct = 0
        by_protocol: dict[str, dict] = {}

        for record in records:
            protocol = record.protocol.value
            if protocol not in by_protocol:
                by_protocol[protocol] = {"total": 0, "correct": 0}

            by_protocol[protocol]["total"] += 1

            for verdict in record.verdicts:
                if verdict.expected and verdict.state == verdict.expected:
                    correct += 1
                    by_protocol[protocol]["correct"] += 1

        return {
            "total_trials": total,
            "accuracy": correct / total if total > 0 else 0,
            "by_protocol": by_protocol,
        }


def load_session(filepath: Path) -> list[TrialRecord]:
    """Load trial records from a telemetry file."""
    records = []
    with open(filepath) as f:
        for line in f:
            data = json.loads(line)
            records.append(TrialRecord(**data))
    return records


def compute_bond_index(records: list[TrialRecord], tau: float = 1.0) -> float:
    """
    Compute the Bond Index from trial records.

    B_d = D_op / tau

    Where D_op = observed defects (gate failures, symmetry violations,
    path inconsistencies).
    """
    if not records:
        return 0.0

    defects = 0
    total_verdicts = 0

    for record in records:
        for verdict in record.verdicts:
            total_verdicts += 1
            if verdict.expected and verdict.state != verdict.expected:
                defects += 1

    if total_verdicts == 0:
        return 0.0

    d_op = defects / total_verdicts
    return d_op / tau
