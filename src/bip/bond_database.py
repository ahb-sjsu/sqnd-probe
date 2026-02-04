"""
Bond Database - SQLite storage for extracted moral bonds.

Provides efficient querying, filtering, and analysis of large bond collections.
"""

import json
import sqlite3
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Iterator, Optional

from .moral_structure import (
    MoralBond,
    BondType,
    RoleType,
    ActionCategory,
    ContextType,
    ModalStrength,
)


class BondDatabase:
    """SQLite-backed database for moral bonds."""

    def __init__(self, path: Path | str):
        """
        Initialize database connection.

        Args:
            path: Path to SQLite database file (created if doesn't exist)
        """
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path))
        self.conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self) -> None:
        """Create database schema."""
        cursor = self.conn.cursor()

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS bonds (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                bond_type TEXT NOT NULL,
                agent_role TEXT NOT NULL,
                patient_role TEXT NOT NULL,
                action TEXT NOT NULL,
                modal_strength TEXT DEFAULT 'must',
                context TEXT DEFAULT 'generic',
                condition TEXT,
                exception TEXT,
                source_language TEXT,
                source_tradition TEXT,
                source_text_id TEXT,
                confidence REAL DEFAULT 1.0,
                canonical_tuple TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        # Create indexes for common queries
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_bond_type ON bonds(bond_type)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_tradition ON bonds(source_tradition)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_language ON bonds(source_language)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_canonical ON bonds(canonical_tuple)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_agent_role ON bonds(agent_role)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_patient_role ON bonds(patient_role)")

        self.conn.commit()

    def close(self) -> None:
        """Close database connection."""
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    # =========================================================================
    # INSERT OPERATIONS
    # =========================================================================

    def add_bond(self, bond: MoralBond) -> int:
        """Add a single bond to the database. Returns the bond ID."""
        cursor = self.conn.cursor()

        canonical = str(bond.to_canonical_tuple())

        cursor.execute(
            """
            INSERT INTO bonds (
                bond_type, agent_role, patient_role, action,
                modal_strength, context, condition, exception,
                source_language, source_tradition, source_text_id,
                confidence, canonical_tuple
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                bond.bond_type.value,
                bond.agent_role.value,
                bond.patient_role.value,
                bond.action.value,
                bond.modal_strength.value,
                bond.context.value,
                bond.condition,
                bond.exception,
                bond.source_language,
                bond.source_tradition,
                bond.source_text_id,
                bond.confidence,
                canonical,
            ),
        )

        self.conn.commit()
        return cursor.lastrowid

    def add_bonds(self, bonds: list[MoralBond]) -> int:
        """Add multiple bonds. Returns count of bonds added."""
        cursor = self.conn.cursor()

        rows = []
        for bond in bonds:
            canonical = str(bond.to_canonical_tuple())
            rows.append(
                (
                    bond.bond_type.value,
                    bond.agent_role.value,
                    bond.patient_role.value,
                    bond.action.value,
                    bond.modal_strength.value,
                    bond.context.value,
                    bond.condition,
                    bond.exception,
                    bond.source_language,
                    bond.source_tradition,
                    bond.source_text_id,
                    bond.confidence,
                    canonical,
                )
            )

        cursor.executemany(
            """
            INSERT INTO bonds (
                bond_type, agent_role, patient_role, action,
                modal_strength, context, condition, exception,
                source_language, source_tradition, source_text_id,
                confidence, canonical_tuple
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )

        self.conn.commit()
        return len(rows)

    # =========================================================================
    # QUERY OPERATIONS
    # =========================================================================

    def _row_to_bond(self, row: sqlite3.Row) -> MoralBond:
        """Convert a database row to a MoralBond object."""
        return MoralBond(
            bond_type=BondType(row["bond_type"]),
            agent_role=RoleType(row["agent_role"]),
            patient_role=RoleType(row["patient_role"]),
            action=ActionCategory(row["action"]),
            modal_strength=ModalStrength(row["modal_strength"]),
            context=ContextType(row["context"]),
            condition=row["condition"],
            exception=row["exception"],
            source_language=row["source_language"],
            source_tradition=row["source_tradition"],
            source_text_id=row["source_text_id"],
            confidence=row["confidence"],
        )

    def get_all(self, limit: Optional[int] = None) -> list[MoralBond]:
        """Get all bonds, optionally limited."""
        cursor = self.conn.cursor()
        if limit:
            cursor.execute("SELECT * FROM bonds LIMIT ?", (limit,))
        else:
            cursor.execute("SELECT * FROM bonds")
        return [self._row_to_bond(row) for row in cursor.fetchall()]

    def query_by_tradition(self, tradition: str) -> list[MoralBond]:
        """Get all bonds from a specific tradition."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM bonds WHERE source_tradition = ?", (tradition,))
        return [self._row_to_bond(row) for row in cursor.fetchall()]

    def query_by_language(self, language: str) -> list[MoralBond]:
        """Get all bonds from a specific language."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM bonds WHERE source_language = ?", (language,))
        return [self._row_to_bond(row) for row in cursor.fetchall()]

    def query_by_bond_type(self, bond_type: BondType | str) -> list[MoralBond]:
        """Get all bonds of a specific type."""
        if isinstance(bond_type, BondType):
            bond_type = bond_type.value
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM bonds WHERE bond_type = ?", (bond_type,))
        return [self._row_to_bond(row) for row in cursor.fetchall()]

    def query_by_roles(
        self,
        agent_role: Optional[RoleType | str] = None,
        patient_role: Optional[RoleType | str] = None,
    ) -> list[MoralBond]:
        """Get bonds filtered by agent and/or patient roles."""
        cursor = self.conn.cursor()

        conditions = []
        params = []

        if agent_role:
            if isinstance(agent_role, RoleType):
                agent_role = agent_role.value
            conditions.append("agent_role = ?")
            params.append(agent_role)

        if patient_role:
            if isinstance(patient_role, RoleType):
                patient_role = patient_role.value
            conditions.append("patient_role = ?")
            params.append(patient_role)

        if conditions:
            query = f"SELECT * FROM bonds WHERE {' AND '.join(conditions)}"
            cursor.execute(query, params)
        else:
            cursor.execute("SELECT * FROM bonds")

        return [self._row_to_bond(row) for row in cursor.fetchall()]

    def query_universal(self, min_traditions: int = 3) -> list[tuple]:
        """
        Find canonical patterns that appear across multiple traditions.

        Returns list of (canonical_tuple, traditions, count) tuples.
        """
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT canonical_tuple,
                   GROUP_CONCAT(DISTINCT source_tradition) as traditions,
                   COUNT(*) as count
            FROM bonds
            WHERE source_tradition IS NOT NULL
            GROUP BY canonical_tuple
            HAVING COUNT(DISTINCT source_tradition) >= ?
            ORDER BY COUNT(DISTINCT source_tradition) DESC, count DESC
            """,
            (min_traditions,),
        )

        results = []
        for row in cursor.fetchall():
            traditions = row["traditions"].split(",") if row["traditions"] else []
            results.append((row["canonical_tuple"], traditions, row["count"]))

        return results

    def query_by_canonical(self, canonical_tuple: str | tuple) -> list[MoralBond]:
        """Get all bonds matching a canonical tuple."""
        if isinstance(canonical_tuple, tuple):
            canonical_tuple = str(canonical_tuple)
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM bonds WHERE canonical_tuple = ?", (canonical_tuple,))
        return [self._row_to_bond(row) for row in cursor.fetchall()]

    # =========================================================================
    # STATISTICS
    # =========================================================================

    def count(self, tradition: Optional[str] = None) -> int:
        """Count bonds, optionally filtered by tradition."""
        cursor = self.conn.cursor()
        if tradition:
            cursor.execute(
                "SELECT COUNT(*) FROM bonds WHERE source_tradition = ?",
                (tradition,),
            )
        else:
            cursor.execute("SELECT COUNT(*) FROM bonds")
        return cursor.fetchone()[0]

    def get_statistics(self) -> dict:
        """Get comprehensive database statistics."""
        cursor = self.conn.cursor()

        stats = {"total_bonds": self.count()}

        # By tradition
        cursor.execute(
            """
            SELECT source_tradition, COUNT(*) as count
            FROM bonds
            GROUP BY source_tradition
            ORDER BY count DESC
            """
        )
        stats["by_tradition"] = {row["source_tradition"]: row["count"] for row in cursor.fetchall()}

        # By language
        cursor.execute(
            """
            SELECT source_language, COUNT(*) as count
            FROM bonds
            GROUP BY source_language
            ORDER BY count DESC
            """
        )
        stats["by_language"] = {row["source_language"]: row["count"] for row in cursor.fetchall()}

        # By bond type
        cursor.execute(
            """
            SELECT bond_type, COUNT(*) as count
            FROM bonds
            GROUP BY bond_type
            ORDER BY count DESC
            """
        )
        stats["by_bond_type"] = {row["bond_type"]: row["count"] for row in cursor.fetchall()}

        # By action
        cursor.execute(
            """
            SELECT action, COUNT(*) as count
            FROM bonds
            GROUP BY action
            ORDER BY count DESC
            LIMIT 20
            """
        )
        stats["top_actions"] = {row["action"]: row["count"] for row in cursor.fetchall()}

        # Unique canonical patterns
        cursor.execute("SELECT COUNT(DISTINCT canonical_tuple) FROM bonds")
        stats["unique_patterns"] = cursor.fetchone()[0]

        # Average confidence
        cursor.execute("SELECT AVG(confidence) FROM bonds")
        stats["avg_confidence"] = cursor.fetchone()[0] or 0.0

        return stats

    def get_tradition_overlap(self) -> dict[tuple[str, str], int]:
        """
        Compute overlap between traditions.

        Returns dict mapping (tradition1, tradition2) -> count of shared canonical patterns.
        """
        cursor = self.conn.cursor()

        # Get canonical patterns per tradition
        cursor.execute(
            """
            SELECT source_tradition, canonical_tuple
            FROM bonds
            WHERE source_tradition IS NOT NULL
            GROUP BY source_tradition, canonical_tuple
            """
        )

        tradition_patterns = {}
        for row in cursor.fetchall():
            tradition = row["source_tradition"]
            pattern = row["canonical_tuple"]
            if tradition not in tradition_patterns:
                tradition_patterns[tradition] = set()
            tradition_patterns[tradition].add(pattern)

        # Compute pairwise overlap
        overlap = {}
        traditions = list(tradition_patterns.keys())
        for i, t1 in enumerate(traditions):
            for t2 in traditions[i + 1 :]:
                shared = tradition_patterns[t1] & tradition_patterns[t2]
                if shared:
                    overlap[(t1, t2)] = len(shared)

        return overlap

    # =========================================================================
    # EXPORT
    # =========================================================================

    def export_to_jsonl(self, path: Path) -> int:
        """Export all bonds to JSONL format. Returns count exported."""
        bonds = self.get_all()
        with open(path, "w", encoding="utf-8") as f:
            for bond in bonds:
                f.write(json.dumps(bond.to_dict(), ensure_ascii=False) + "\n")
        return len(bonds)

    def import_from_jsonl(self, path: Path) -> int:
        """Import bonds from JSONL format. Returns count imported."""
        bonds = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    bonds.append(MoralBond.from_dict(data))
        return self.add_bonds(bonds)


def print_database_stats(db: BondDatabase) -> None:
    """Print formatted database statistics."""
    stats = db.get_statistics()

    print("=" * 60)
    print("BOND DATABASE STATISTICS")
    print("=" * 60)

    print(f"\nTotal bonds: {stats['total_bonds']}")
    print(f"Unique patterns: {stats['unique_patterns']}")
    print(f"Average confidence: {stats['avg_confidence']:.2f}")

    print("\nBy Tradition:")
    for tradition, count in sorted(stats["by_tradition"].items(), key=lambda x: -x[1]):
        if tradition:
            print(f"  {tradition:20s}: {count:5d}")

    print("\nBy Language:")
    for lang, count in sorted(stats["by_language"].items(), key=lambda x: -x[1]):
        if lang:
            print(f"  {lang:20s}: {count:5d}")

    print("\nBy Bond Type:")
    for bt, count in sorted(stats["by_bond_type"].items(), key=lambda x: -x[1]):
        print(f"  {bt:20s}: {count:5d}")

    print("\nTop Actions:")
    for action, count in list(stats["top_actions"].items())[:10]:
        print(f"  {action:20s}: {count:5d}")

    print("=" * 60)
