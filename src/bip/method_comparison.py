"""
Method Comparison - Compare embedding-based vs structured extraction.

Provides tools to evaluate agreement between:
1. BIP model embedding predictions (z_bond -> bond_type)
2. Structured LLM extraction (text -> MoralBond)
"""

from collections import Counter, defaultdict
from dataclasses import dataclass

from .moral_structure import BondType, MoralBond

# =============================================================================
# COMPARISON DATA STRUCTURES
# =============================================================================


@dataclass
class ComparisonResult:
    """Result of comparing two extraction methods."""

    # Overall agreement
    agreement_rate: float
    total_compared: int

    # By bond type
    bond_type_agreement: dict[str, float]

    # Disagreement analysis
    disagreements: list[dict]

    # Per-tradition breakdown
    by_tradition: dict[str, dict]

    # Which method is more consistent
    consistency_scores: dict[str, float]


# =============================================================================
# EMBEDDING PREDICTION NORMALIZATION
# =============================================================================

# Map BIP model bond type indices to our BondType enum
BIP_BOND_TYPE_MAP = {
    0: BondType.PROHIBITION,  # HARM_PREVENTION -> prohibition
    1: BondType.OBLIGATION,  # RECIPROCITY -> obligation (mutual duties)
    2: BondType.LIBERTY,  # AUTONOMY -> liberty
    3: BondType.CLAIM,  # PROPERTY -> claim (rights)
    4: BondType.OBLIGATION,  # FAMILY -> obligation
    5: BondType.OBLIGATION,  # AUTHORITY -> obligation
    6: BondType.PERMISSION,  # EMERGENCY -> permission (exception)
    7: BondType.OBLIGATION,  # CONTRACT -> obligation
    8: BondType.OBLIGATION,  # CARE -> obligation
    9: BondType.OBLIGATION,  # FAIRNESS -> obligation
}


def normalize_embedding_prediction(pred: dict) -> dict:
    """
    Normalize embedding model prediction to comparable format.

    Args:
        pred: Dict with 'text', 'bond_type' (index or name), 'language', etc.

    Returns:
        Normalized dict with BondType enum
    """
    result = pred.copy()

    # Convert bond_type index to enum
    bt = pred.get("bond_type")
    if isinstance(bt, int):
        result["bond_type"] = BIP_BOND_TYPE_MAP.get(bt, BondType.NO_BOND)
    elif isinstance(bt, str):
        try:
            result["bond_type"] = BondType(bt.lower())
        except ValueError:
            result["bond_type"] = BondType.NO_BOND

    return result


# =============================================================================
# COMPARISON FUNCTIONS
# =============================================================================


def match_bond_to_prediction(
    bond: MoralBond,
    predictions: list[dict],
    text_key: str = "text",
) -> dict | None:
    """
    Find the embedding prediction that matches a structured bond.

    Matches by source_text_id or text similarity.
    """
    # Try exact ID match
    if bond.source_text_id:
        for pred in predictions:
            if pred.get("id") == bond.source_text_id:
                return pred

    # Fallback: could add text similarity matching here
    return None


def compare_bond_types(
    structured_bond: MoralBond,
    embedding_pred: dict,
) -> dict:
    """
    Compare bond type between structured extraction and embedding prediction.

    Returns dict with match status and details.
    """
    struct_type = structured_bond.bond_type
    embed_type = embedding_pred.get("bond_type")

    if isinstance(embed_type, int):
        embed_type = BIP_BOND_TYPE_MAP.get(embed_type, BondType.NO_BOND)
    elif isinstance(embed_type, str):
        try:
            embed_type = BondType(embed_type.lower())
        except ValueError:
            embed_type = BondType.NO_BOND

    exact_match = struct_type == embed_type

    # Check if they're in the same "family"
    # (obligation, prohibition) vs (permission, liberty) vs (claim, power)
    obligation_family = {BondType.OBLIGATION, BondType.PROHIBITION}
    permission_family = {BondType.PERMISSION, BondType.LIBERTY}
    rights_family = {BondType.CLAIM, BondType.POWER, BondType.IMMUNITY}

    same_family = any(
        struct_type in family and embed_type in family
        for family in [obligation_family, permission_family, rights_family]
    )

    return {
        "exact_match": exact_match,
        "same_family": same_family,
        "structured_type": struct_type.value,
        "embedding_type": embed_type.value if isinstance(embed_type, BondType) else str(embed_type),
        "structured_confidence": structured_bond.confidence,
        "embedding_confidence": embedding_pred.get("confidence", 1.0),
    }


def compare_extraction_methods(
    structured_bonds: list[MoralBond],
    embedding_predictions: list[dict],
    match_by_text: bool = False,
) -> ComparisonResult:
    """
    Compare structured extraction vs embedding predictions.

    Args:
        structured_bonds: Bonds from LLM structured extraction
        embedding_predictions: Predictions from embedding model
        match_by_text: Whether to match by text content (slower)

    Returns:
        ComparisonResult with agreement metrics
    """
    # Normalize embedding predictions
    normalized_preds = [normalize_embedding_prediction(p) for p in embedding_predictions]

    # Build lookup by ID
    pred_by_id = {p.get("id"): p for p in normalized_preds if p.get("id")}

    # Track comparisons
    comparisons = []
    disagreements = []
    tradition_stats = defaultdict(lambda: {"matches": 0, "total": 0})
    bond_type_stats = defaultdict(lambda: {"matches": 0, "total": 0})

    for bond in structured_bonds:
        # Find matching prediction
        pred = None
        if bond.source_text_id and bond.source_text_id in pred_by_id:
            pred = pred_by_id[bond.source_text_id]

        if not pred:
            continue  # No matching prediction

        # Compare
        comparison = compare_bond_types(bond, pred)
        comparisons.append(comparison)

        # Track by tradition
        tradition = bond.source_tradition or "unknown"
        tradition_stats[tradition]["total"] += 1
        if comparison["exact_match"]:
            tradition_stats[tradition]["matches"] += 1

        # Track by bond type
        bt = comparison["structured_type"]
        bond_type_stats[bt]["total"] += 1
        if comparison["exact_match"]:
            bond_type_stats[bt]["matches"] += 1

        # Track disagreements
        if not comparison["exact_match"]:
            disagreements.append(
                {
                    "text_id": bond.source_text_id,
                    "tradition": tradition,
                    "structured": comparison["structured_type"],
                    "embedding": comparison["embedding_type"],
                    "same_family": comparison["same_family"],
                }
            )

    # Compute agreement rate
    total = len(comparisons)
    matches = sum(1 for c in comparisons if c["exact_match"])
    agreement_rate = matches / total if total > 0 else 0.0

    # Per-bond-type agreement
    bond_type_agreement = {}
    for bt, stats in bond_type_stats.items():
        if stats["total"] > 0:
            bond_type_agreement[bt] = stats["matches"] / stats["total"]

    # Per-tradition breakdown
    by_tradition = {}
    for tradition, stats in tradition_stats.items():
        if stats["total"] > 0:
            by_tradition[tradition] = {
                "agreement": stats["matches"] / stats["total"],
                "total": stats["total"],
                "matches": stats["matches"],
            }

    # Consistency scores (how often each method gives same answer for similar inputs)
    struct_consistency = compute_consistency(structured_bonds)
    embed_consistency = compute_consistency_from_preds(normalized_preds)

    return ComparisonResult(
        agreement_rate=agreement_rate,
        total_compared=total,
        bond_type_agreement=bond_type_agreement,
        disagreements=disagreements[:50],  # Limit to top 50
        by_tradition=by_tradition,
        consistency_scores={
            "structured": struct_consistency,
            "embedding": embed_consistency,
        },
    )


def compute_consistency(bonds: list[MoralBond]) -> float:
    """
    Compute how consistently the structured method assigns bond types.

    Measures: for similar canonical tuples (same agent/patient/action),
    how often is the same bond_type assigned?
    """
    # Group by (agent, patient, action)
    groups = defaultdict(list)
    for bond in bonds:
        key = (bond.agent_role, bond.patient_role, bond.action)
        groups[key].append(bond.bond_type)

    # Compute consistency per group
    consistencies = []
    for key, bond_types in groups.items():
        if len(bond_types) > 1:
            # Most common bond type
            counter = Counter(bond_types)
            most_common_count = counter.most_common(1)[0][1]
            consistency = most_common_count / len(bond_types)
            consistencies.append(consistency)

    return sum(consistencies) / len(consistencies) if consistencies else 1.0


def compute_consistency_from_preds(predictions: list[dict]) -> float:
    """Compute consistency from embedding predictions."""
    # Group by text hash (approximation)
    groups = defaultdict(list)
    for pred in predictions:
        text = pred.get("text", "")
        key = hash(text[:50])  # Use first 50 chars as key
        bt = pred.get("bond_type")
        if bt is not None:
            groups[key].append(bt)

    consistencies = []
    for key, bond_types in groups.items():
        if len(bond_types) > 1:
            counter = Counter(bond_types)
            most_common_count = counter.most_common(1)[0][1]
            consistency = most_common_count / len(bond_types)
            consistencies.append(consistency)

    return sum(consistencies) / len(consistencies) if consistencies else 1.0


# =============================================================================
# REPORTING
# =============================================================================


def print_comparison_report(result: ComparisonResult) -> None:
    """Print formatted comparison report."""
    print("=" * 60)
    print("METHOD COMPARISON: Structured vs Embedding")
    print("=" * 60)

    print("\n## Overall Agreement")
    print(f"  Agreement rate: {result.agreement_rate:.1%}")
    print(f"  Total compared: {result.total_compared}")

    print("\n## Consistency Scores")
    print(f"  Structured extraction: {result.consistency_scores.get('structured', 0):.1%}")
    print(f"  Embedding prediction: {result.consistency_scores.get('embedding', 0):.1%}")

    if result.bond_type_agreement:
        print("\n## Agreement by Bond Type")
        for bt, rate in sorted(result.bond_type_agreement.items(), key=lambda x: -x[1]):
            print(f"  {bt:20s}: {rate:.1%}")

    if result.by_tradition:
        print("\n## Agreement by Tradition")
        for tradition, stats in sorted(
            result.by_tradition.items(), key=lambda x: -x[1]["agreement"]
        ):
            print(
                f"  {tradition:20s}: {stats['agreement']:.1%} ({stats['matches']}/{stats['total']})"
            )

    if result.disagreements:
        print("\n## Sample Disagreements (top 10)")
        for d in result.disagreements[:10]:
            family_note = " (same family)" if d["same_family"] else ""
            print(f"  {d['tradition']}: {d['structured']} vs {d['embedding']}{family_note}")

    print("=" * 60)


def analyze_systematic_disagreements(result: ComparisonResult) -> dict:
    """
    Analyze patterns in disagreements between methods.

    Returns dict with systematic patterns found.
    """
    if not result.disagreements:
        return {"patterns": []}

    # Count transition patterns
    transitions = Counter()
    for d in result.disagreements:
        key = (d["structured"], d["embedding"])
        transitions[key] += 1

    # Find systematic patterns (appear > 3 times)
    patterns = []
    for (struct_type, embed_type), count in transitions.most_common(10):
        if count >= 3:
            patterns.append(
                {
                    "structured_predicts": struct_type,
                    "embedding_predicts": embed_type,
                    "count": count,
                    "percentage": count / len(result.disagreements) * 100,
                }
            )

    # Tradition-specific patterns
    tradition_patterns = defaultdict(lambda: defaultdict(int))
    for d in result.disagreements:
        key = (d["structured"], d["embedding"])
        tradition_patterns[d["tradition"]][key] += 1

    return {
        "patterns": patterns,
        "tradition_patterns": {k: dict(v) for k, v in tradition_patterns.items()},
        "total_disagreements": len(result.disagreements),
    }
