"""
Extraction Metrics - Validate moral bond extraction quality.

Compares extracted bonds against gold standard annotations to measure
precision, recall, F1, and field-level accuracy.
"""

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import yaml

from .moral_structure import (
    ActionCategory,
    BondType,
    ContextType,
    ModalStrength,
    MoralBond,
    RoleType,
)

# =============================================================================
# GOLD STANDARD LOADING
# =============================================================================


@dataclass
class GoldStandardEntry:
    """A gold standard annotated passage."""

    id: str
    text: str
    language: str
    tradition: str
    source: str
    expected_bonds: list[MoralBond]


def load_gold_standard(path: Path) -> list[GoldStandardEntry]:
    """Load gold standard annotations from YAML file."""
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    entries = []
    for item in data:
        bonds = []
        for bond_data in item.get("expected_bonds", []):
            bond = MoralBond(
                bond_type=BondType(bond_data["bond_type"]),
                agent_role=RoleType(bond_data["agent_role"]),
                patient_role=RoleType(bond_data["patient_role"]),
                action=ActionCategory(bond_data["action"]),
                modal_strength=ModalStrength(bond_data.get("modal_strength", "must")),
                context=ContextType(bond_data.get("context", "generic")),
                condition=bond_data.get("condition"),
                exception=bond_data.get("exception"),
                confidence=bond_data.get("confidence", 1.0),
                source_language=item["language"],
                source_tradition=item["tradition"],
            )
            bonds.append(bond)

        entry = GoldStandardEntry(
            id=item["id"],
            text=item["text"],
            language=item["language"],
            tradition=item["tradition"],
            source=item["source"],
            expected_bonds=bonds,
        )
        entries.append(entry)

    return entries


# =============================================================================
# MATCHING FUNCTIONS
# =============================================================================


def bond_exact_match(pred: MoralBond, gold: MoralBond) -> bool:
    """Check if two bonds match exactly on core fields."""
    return (
        pred.bond_type == gold.bond_type
        and pred.agent_role == gold.agent_role
        and pred.patient_role == gold.patient_role
        and pred.action == gold.action
    )


def bond_partial_match(pred: MoralBond, gold: MoralBond) -> float:
    """
    Compute partial match score between two bonds.

    Returns a score from 0.0 to 1.0 based on field agreement.
    """
    score = 0.0
    weights = {
        "bond_type": 0.4,
        "agent_role": 0.2,
        "patient_role": 0.2,
        "action": 0.2,
    }

    if pred.bond_type == gold.bond_type:
        score += weights["bond_type"]
    if pred.agent_role == gold.agent_role:
        score += weights["agent_role"]
    if pred.patient_role == gold.patient_role:
        score += weights["patient_role"]
    if pred.action == gold.action:
        score += weights["action"]

    return score


def find_best_match(
    pred: MoralBond, gold_bonds: list[MoralBond], threshold: float = 0.5
) -> tuple[MoralBond, float] | None:
    """
    Find the best matching gold bond for a prediction.

    Returns (matched_bond, score) or None if no match above threshold.
    """
    best_match = None
    best_score = 0.0

    for gold in gold_bonds:
        # Check exact match first
        if bond_exact_match(pred, gold):
            return (gold, 1.0)

        # Check partial match
        score = bond_partial_match(pred, gold)
        if score > best_score:
            best_score = score
            best_match = gold

    if best_score >= threshold:
        return (best_match, best_score)
    return None


# =============================================================================
# METRICS COMPUTATION
# =============================================================================


@dataclass
class ExtractionMetrics:
    """Metrics for extraction evaluation."""

    # Overall metrics
    precision: float
    recall: float
    f1: float

    # Exact match metrics
    exact_precision: float
    exact_recall: float
    exact_f1: float

    # Per-field accuracy
    bond_type_accuracy: float
    agent_role_accuracy: float
    patient_role_accuracy: float
    action_accuracy: float

    # Counts
    total_predicted: int
    total_gold: int
    true_positives: int
    false_positives: int
    false_negatives: int

    # Per-tradition breakdown
    by_tradition: dict


def compute_extraction_metrics(
    predicted: list[MoralBond],
    gold: list[MoralBond],
    partial_threshold: float = 0.5,
) -> ExtractionMetrics:
    """
    Compute comprehensive extraction metrics.

    Args:
        predicted: List of predicted MoralBond objects
        gold: List of gold standard MoralBond objects
        partial_threshold: Minimum score for partial match

    Returns:
        ExtractionMetrics with precision, recall, F1, and per-field accuracy
    """
    # Track matches
    matched_gold = set()
    matched_predictions = []
    exact_matches = 0
    partial_matches = 0

    # Field-level tracking
    field_correct = {
        "bond_type": 0,
        "agent_role": 0,
        "patient_role": 0,
        "action": 0,
    }
    field_total = 0

    # Per-tradition tracking
    tradition_stats = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

    # Find matches for each prediction
    for pred in predicted:
        match = find_best_match(
            pred, [g for i, g in enumerate(gold) if i not in matched_gold], partial_threshold
        )

        tradition = pred.source_tradition or "unknown"

        if match:
            gold_bond, score = match
            gold_idx = gold.index(gold_bond)
            matched_gold.add(gold_idx)
            matched_predictions.append((pred, gold_bond, score))

            if score == 1.0:
                exact_matches += 1
            else:
                partial_matches += 1

            tradition_stats[tradition]["tp"] += 1

            # Field-level accuracy
            field_total += 1
            if pred.bond_type == gold_bond.bond_type:
                field_correct["bond_type"] += 1
            if pred.agent_role == gold_bond.agent_role:
                field_correct["agent_role"] += 1
            if pred.patient_role == gold_bond.patient_role:
                field_correct["patient_role"] += 1
            if pred.action == gold_bond.action:
                field_correct["action"] += 1
        else:
            tradition_stats[tradition]["fp"] += 1

    # Count false negatives (unmatched gold)
    for i, gold_bond in enumerate(gold):
        if i not in matched_gold:
            tradition = gold_bond.source_tradition or "unknown"
            tradition_stats[tradition]["fn"] += 1

    # Compute metrics
    tp = len(matched_predictions)
    fp = len(predicted) - tp
    fn = len(gold) - tp

    precision = tp / len(predicted) if predicted else 0.0
    recall = tp / len(gold) if gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    exact_precision = exact_matches / len(predicted) if predicted else 0.0
    exact_recall = exact_matches / len(gold) if gold else 0.0
    exact_f1 = (
        2 * exact_precision * exact_recall / (exact_precision + exact_recall)
        if (exact_precision + exact_recall) > 0
        else 0.0
    )

    # Per-tradition metrics
    by_tradition = {}
    for tradition, counts in tradition_stats.items():
        t_tp = counts["tp"]
        t_fp = counts["fp"]
        t_fn = counts["fn"]
        t_precision = t_tp / (t_tp + t_fp) if (t_tp + t_fp) > 0 else 0.0
        t_recall = t_tp / (t_tp + t_fn) if (t_tp + t_fn) > 0 else 0.0
        t_f1 = (
            2 * t_precision * t_recall / (t_precision + t_recall)
            if (t_precision + t_recall) > 0
            else 0.0
        )
        by_tradition[tradition] = {
            "precision": t_precision,
            "recall": t_recall,
            "f1": t_f1,
            "tp": t_tp,
            "fp": t_fp,
            "fn": t_fn,
        }

    return ExtractionMetrics(
        precision=precision,
        recall=recall,
        f1=f1,
        exact_precision=exact_precision,
        exact_recall=exact_recall,
        exact_f1=exact_f1,
        bond_type_accuracy=field_correct["bond_type"] / field_total if field_total > 0 else 0.0,
        agent_role_accuracy=field_correct["agent_role"] / field_total if field_total > 0 else 0.0,
        patient_role_accuracy=(
            field_correct["patient_role"] / field_total if field_total > 0 else 0.0
        ),
        action_accuracy=field_correct["action"] / field_total if field_total > 0 else 0.0,
        total_predicted=len(predicted),
        total_gold=len(gold),
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        by_tradition=by_tradition,
    )


def compute_confusion_matrix(
    predicted: list[MoralBond], gold: list[MoralBond]
) -> dict[str, dict[str, int]]:
    """
    Compute confusion matrix for bond type predictions.

    Returns dict[gold_type][pred_type] = count
    """
    matrix = defaultdict(lambda: defaultdict(int))

    # Match predictions to gold
    matched_gold = set()
    for pred in predicted:
        match = find_best_match(
            pred, [g for i, g in enumerate(gold) if i not in matched_gold], threshold=0.3
        )
        if match:
            gold_bond, _ = match
            gold_idx = gold.index(gold_bond)
            matched_gold.add(gold_idx)
            matrix[gold_bond.bond_type.value][pred.bond_type.value] += 1

    return {k: dict(v) for k, v in matrix.items()}


# =============================================================================
# REPORTING
# =============================================================================


def print_metrics_report(metrics: ExtractionMetrics) -> None:
    """Print a formatted metrics report."""
    print("=" * 60)
    print("EXTRACTION METRICS REPORT")
    print("=" * 60)

    print("\n## Overall Metrics (Partial Match)")
    print(f"  Precision: {metrics.precision:.3f}")
    print(f"  Recall:    {metrics.recall:.3f}")
    print(f"  F1 Score:  {metrics.f1:.3f}")

    print("\n## Exact Match Metrics")
    print(f"  Precision: {metrics.exact_precision:.3f}")
    print(f"  Recall:    {metrics.exact_recall:.3f}")
    print(f"  F1 Score:  {metrics.exact_f1:.3f}")

    print("\n## Per-Field Accuracy")
    print(f"  Bond Type:    {metrics.bond_type_accuracy:.3f}")
    print(f"  Agent Role:   {metrics.agent_role_accuracy:.3f}")
    print(f"  Patient Role: {metrics.patient_role_accuracy:.3f}")
    print(f"  Action:       {metrics.action_accuracy:.3f}")

    print("\n## Counts")
    print(f"  Total Predicted: {metrics.total_predicted}")
    print(f"  Total Gold:      {metrics.total_gold}")
    print(f"  True Positives:  {metrics.true_positives}")
    print(f"  False Positives: {metrics.false_positives}")
    print(f"  False Negatives: {metrics.false_negatives}")

    if metrics.by_tradition:
        print("\n## By Tradition")
        for tradition, stats in sorted(metrics.by_tradition.items()):
            print(f"  {tradition}:")
            print(f"    P={stats['precision']:.2f} R={stats['recall']:.2f} F1={stats['f1']:.2f}")
            print(f"    TP={stats['tp']} FP={stats['fp']} FN={stats['fn']}")

    print("=" * 60)


def evaluate_on_gold_standard(
    extractor_fn,
    gold_standard_path: Path,
    verbose: bool = True,
) -> ExtractionMetrics:
    """
    Evaluate an extractor function on the gold standard dataset.

    Args:
        extractor_fn: Function that takes (text, language, tradition) and returns list[MoralBond]
        gold_standard_path: Path to gold standard YAML file
        verbose: Print progress and results

    Returns:
        ExtractionMetrics for the evaluation
    """
    entries = load_gold_standard(gold_standard_path)

    all_predicted = []
    all_gold = []

    for entry in entries:
        if verbose:
            print(f"Evaluating: {entry.id}")

        predicted = extractor_fn(entry.text, entry.language, entry.tradition)
        all_predicted.extend(predicted)
        all_gold.extend(entry.expected_bonds)

    metrics = compute_extraction_metrics(all_predicted, all_gold)

    if verbose:
        print_metrics_report(metrics)

    return metrics
