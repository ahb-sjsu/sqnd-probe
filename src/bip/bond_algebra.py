"""
Bond Algebra - Mathematical analysis of moral bond structures.

Computes adjacency matrices, similarity metrics, and algebraic properties
of moral bond collections.
"""

from collections import Counter, defaultdict
from typing import Optional

import numpy as np

from .moral_structure import (
    MoralBond,
    BondType,
    RoleType,
    ActionCategory,
)


# =============================================================================
# MATRIX COMPUTATIONS
# =============================================================================


def get_role_index_map() -> dict[str, int]:
    """Get mapping from role names to indices."""
    roles = [r.value for r in RoleType]
    return {role: i for i, role in enumerate(roles)}


def get_action_index_map() -> dict[str, int]:
    """Get mapping from action names to indices."""
    actions = [a.value for a in ActionCategory]
    return {action: i for i, action in enumerate(actions)}


def get_bond_type_index_map() -> dict[str, int]:
    """Get mapping from bond type names to indices."""
    types = [t.value for t in BondType]
    return {bt: i for i, bt in enumerate(types)}


def compute_role_transition_matrix(bonds: list[MoralBond]) -> tuple[np.ndarray, list[str]]:
    """
    Compute role transition matrix.

    Matrix[i,j] = count of bonds where agent=role_i and patient=role_j

    Returns:
        (matrix, role_names) tuple
    """
    role_map = get_role_index_map()
    n_roles = len(role_map)
    matrix = np.zeros((n_roles, n_roles), dtype=np.float64)

    for bond in bonds:
        i = role_map.get(bond.agent_role.value, -1)
        j = role_map.get(bond.patient_role.value, -1)
        if i >= 0 and j >= 0:
            matrix[i, j] += 1

    # Normalize rows to get transition probabilities
    row_sums = matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    matrix_normalized = matrix / row_sums

    role_names = list(role_map.keys())
    return matrix_normalized, role_names


def compute_action_bond_matrix(bonds: list[MoralBond]) -> tuple[np.ndarray, list[str], list[str]]:
    """
    Compute action-bond type co-occurrence matrix.

    Matrix[i,j] = count of action_i with bond_type_j

    Returns:
        (matrix, action_names, bond_type_names) tuple
    """
    action_map = get_action_index_map()
    bond_map = get_bond_type_index_map()

    n_actions = len(action_map)
    n_types = len(bond_map)
    matrix = np.zeros((n_actions, n_types), dtype=np.float64)

    for bond in bonds:
        i = action_map.get(bond.action.value, -1)
        j = bond_map.get(bond.bond_type.value, -1)
        if i >= 0 and j >= 0:
            matrix[i, j] += 1

    action_names = list(action_map.keys())
    bond_names = list(bond_map.keys())
    return matrix, action_names, bond_names


def compute_tradition_vectors(bonds: list[MoralBond]) -> tuple[np.ndarray, list[str], list[str]]:
    """
    Compute tradition vectors in canonical bond space.

    Each tradition is represented as a vector of canonical pattern frequencies.

    Returns:
        (matrix, tradition_names, pattern_names) tuple
        matrix[i, j] = frequency of pattern_j in tradition_i
    """
    # Get all unique patterns and traditions
    patterns = set()
    traditions = set()

    for bond in bonds:
        patterns.add(bond.to_canonical_tuple())
        if bond.source_tradition:
            traditions.add(bond.source_tradition)

    pattern_list = sorted(list(patterns), key=str)
    tradition_list = sorted(list(traditions))

    pattern_map = {p: i for i, p in enumerate(pattern_list)}
    tradition_map = {t: i for i, t in enumerate(tradition_list)}

    # Build count matrix
    matrix = np.zeros((len(tradition_list), len(pattern_list)), dtype=np.float64)

    for bond in bonds:
        if bond.source_tradition:
            i = tradition_map[bond.source_tradition]
            j = pattern_map[bond.to_canonical_tuple()]
            matrix[i, j] += 1

    # Normalize rows (L2 norm for cosine similarity)
    row_norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    row_norms[row_norms == 0] = 1
    matrix_normalized = matrix / row_norms

    pattern_names = [str(p) for p in pattern_list]
    return matrix_normalized, tradition_list, pattern_names


def compute_tradition_similarity(bonds: list[MoralBond]) -> tuple[np.ndarray, list[str]]:
    """
    Compute pairwise cosine similarity between traditions.

    Returns:
        (similarity_matrix, tradition_names) tuple
    """
    vectors, tradition_names, _ = compute_tradition_vectors(bonds)

    # Cosine similarity = dot product of normalized vectors
    similarity = np.dot(vectors, vectors.T)

    return similarity, tradition_names


# =============================================================================
# ALGEBRAIC STRUCTURE ANALYSIS
# =============================================================================


def compute_bond_composition(bonds: list[MoralBond]) -> dict:
    """
    Analyze composition structure of bonds.

    Checks if bonds can be "composed" (A→B, B→C implies A→C).

    Returns dict with composition chains found.
    """
    # Build lookup by (agent_role, patient_role)
    role_pairs = defaultdict(list)
    for bond in bonds:
        key = (bond.agent_role, bond.patient_role)
        role_pairs[key].append(bond)

    # Find transitive chains
    chains = []
    for (a1, b1), bonds1 in role_pairs.items():
        for (a2, b2), bonds2 in role_pairs.items():
            if b1 == a2:  # Composable: A→B and B→C
                # Check if A→C exists
                if (a1, b2) in role_pairs:
                    chains.append(
                        {
                            "first": (a1.value, b1.value),
                            "second": (a2.value, b2.value),
                            "composed": (a1.value, b2.value),
                            "count": len(bonds1) * len(bonds2),
                        }
                    )

    return {
        "transitive_chains": chains,
        "total_chains": len(chains),
    }


def compute_symmetry_structure(bonds: list[MoralBond]) -> dict:
    """
    Analyze symmetry properties of the bond collection.

    Returns dict with symmetry statistics.
    """
    # Build lookup
    role_action_bonds = defaultdict(list)
    for bond in bonds:
        key = (bond.agent_role, bond.patient_role, bond.action)
        role_action_bonds[key].append(bond)

    # Count reciprocal pairs (A→B, B→A with same action)
    reciprocal = 0
    for (agent, patient, action), bond_list in role_action_bonds.items():
        inverse_key = (patient, agent, action)
        if inverse_key in role_action_bonds:
            reciprocal += 1

    # Count reflexive bonds (A→A)
    reflexive = sum(1 for b in bonds if b.agent_role == b.patient_role)

    # Count obligation/prohibition pairs
    obligation_prohibition_pairs = 0
    for bond in bonds:
        if bond.bond_type == BondType.OBLIGATION:
            # Look for matching prohibition
            for other in bonds:
                if (
                    other.bond_type == BondType.PROHIBITION
                    and other.agent_role == bond.agent_role
                    and other.patient_role == bond.patient_role
                    and other.action == bond.action
                ):
                    obligation_prohibition_pairs += 1
                    break

    return {
        "reciprocal_pairs": reciprocal // 2,  # Divide by 2 since we count both directions
        "reflexive_bonds": reflexive,
        "obligation_prohibition_pairs": obligation_prohibition_pairs,
        "total_bonds": len(bonds),
        "symmetry_ratio": reciprocal / (2 * len(bonds)) if bonds else 0,
    }


def compute_group_structure(bonds: list[MoralBond]) -> dict:
    """
    Analyze if bonds form an algebraic structure.

    Checks for:
    - Closure under composition
    - Identity element
    - Inverse elements

    Returns dict with algebraic properties.
    """
    # Get unique role pairs
    role_pairs = set((b.agent_role, b.patient_role) for b in bonds)

    # Check for identity (A→A for each A)
    agents = set(b.agent_role for b in bonds)
    patients = set(b.patient_role for b in bonds)
    all_roles = agents | patients

    identity_present = all((r, r) in role_pairs for r in all_roles)

    # Check for inverses (A→B implies B→A)
    has_inverse = all((p, a) in role_pairs for (a, p) in role_pairs)

    # Check closure under composition
    closure_violations = 0
    for a1, b1 in role_pairs:
        for a2, b2 in role_pairs:
            if b1 == a2:  # Composable
                if (a1, b2) not in role_pairs:
                    closure_violations += 1

    return {
        "is_closed": closure_violations == 0,
        "closure_violations": closure_violations,
        "has_identity": identity_present,
        "has_inverses": has_inverse,
        "is_group": closure_violations == 0 and identity_present and has_inverse,
        "is_monoid": closure_violations == 0 and identity_present,
        "unique_role_pairs": len(role_pairs),
    }


# =============================================================================
# ISOMORPHISM DETECTION
# =============================================================================


def find_moral_isomorphisms(
    bonds1: list[MoralBond],
    bonds2: list[MoralBond],
    threshold: float = 0.8,
) -> list[dict]:
    """
    Find structure-preserving mappings between two bond collections.

    An isomorphism maps canonical tuples from one collection to another
    while preserving relationships.

    Args:
        bonds1: First collection of bonds
        bonds2: Second collection of bonds
        threshold: Minimum overlap for isomorphism detection

    Returns:
        List of potential isomorphisms with their strength scores
    """
    # Get canonical patterns for each collection
    patterns1 = Counter(b.to_canonical_tuple() for b in bonds1)
    patterns2 = Counter(b.to_canonical_tuple() for b in bonds2)

    # Find shared patterns
    shared = set(patterns1.keys()) & set(patterns2.keys())
    total = set(patterns1.keys()) | set(patterns2.keys())

    if not total:
        return []

    # Jaccard similarity
    jaccard = len(shared) / len(total)

    # Pattern-level matches
    matches = []
    for pattern in shared:
        matches.append(
            {
                "pattern": str(pattern),
                "count1": patterns1[pattern],
                "count2": patterns2[pattern],
                "frequency_ratio": min(patterns1[pattern], patterns2[pattern])
                / max(patterns1[pattern], patterns2[pattern]),
            }
        )

    # Sort by frequency ratio
    matches.sort(key=lambda x: -x["frequency_ratio"])

    if jaccard >= threshold:
        return [
            {
                "jaccard_similarity": jaccard,
                "shared_patterns": len(shared),
                "total_patterns": len(total),
                "top_matches": matches[:10],
            }
        ]

    return []


# =============================================================================
# REPORTING
# =============================================================================


def print_algebra_report(bonds: list[MoralBond]) -> None:
    """Print comprehensive algebraic analysis report."""
    print("=" * 60)
    print("BOND ALGEBRA ANALYSIS")
    print("=" * 60)

    # Symmetry structure
    print("\n## Symmetry Structure")
    symmetry = compute_symmetry_structure(bonds)
    print(f"  Total bonds: {symmetry['total_bonds']}")
    print(f"  Reciprocal pairs: {symmetry['reciprocal_pairs']}")
    print(f"  Reflexive bonds: {symmetry['reflexive_bonds']}")
    print(f"  Obligation/Prohibition pairs: {symmetry['obligation_prohibition_pairs']}")
    print(f"  Symmetry ratio: {symmetry['symmetry_ratio']:.3f}")

    # Group structure
    print("\n## Algebraic Structure")
    group = compute_group_structure(bonds)
    print(f"  Unique role pairs: {group['unique_role_pairs']}")
    print(f"  Closed under composition: {group['is_closed']}")
    print(f"  Has identity: {group['has_identity']}")
    print(f"  Has inverses: {group['has_inverses']}")
    print(f"  Is monoid: {group['is_monoid']}")
    print(f"  Is group: {group['is_group']}")

    # Composition chains
    print("\n## Composition Structure")
    composition = compute_bond_composition(bonds)
    print(f"  Transitive chains found: {composition['total_chains']}")
    if composition["transitive_chains"]:
        print("  Top chains:")
        for chain in composition["transitive_chains"][:5]:
            print(f"    {chain['first']} → {chain['second']} = {chain['composed']}")

    # Tradition similarity
    print("\n## Tradition Similarity")
    try:
        similarity, traditions = compute_tradition_similarity(bonds)
        if len(traditions) > 1:
            print("  Pairwise cosine similarities:")
            for i, t1 in enumerate(traditions):
                for j, t2 in enumerate(traditions):
                    if i < j:
                        print(f"    {t1} <-> {t2}: {similarity[i, j]:.3f}")
    except Exception as e:
        print(f"  Could not compute tradition similarity: {e}")

    print("=" * 60)
