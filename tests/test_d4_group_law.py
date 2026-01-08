# Copyright (c) 2026 Andrew H. Bond and Claude Opus 4.5
# Department of Computer Engineering, San Jose State University
# Licensed under the MIT License. See LICENSE file for details.

"""
Tests verifying D4 group law and semantic mappings.

This test suite addresses the critique that the D4 structure must be
rigorously verified. We test:
1. The defining relations of D4 (r⁴=e, s²=e, srs=r⁻¹)
2. Correct identification of semantic operations with group elements
3. Non-abelian structure (rs ≠ sr)
"""

import pytest

from dear_ethicist.models import (
    D4Element,
    HohfeldianState,
    d4_apply_to_state,
    d4_inverse,
    d4_multiply,
    correlative,
)


class TestD4DefiningRelations:
    """Test the defining relations of the dihedral group D4."""

    def test_r_fourth_power_is_identity(self):
        """r⁴ = e: Four rotations return to identity."""
        r = D4Element.R
        result = d4_multiply(r, d4_multiply(r, d4_multiply(r, r)))
        assert result == D4Element.E

        # Also verify on states
        for state in HohfeldianState:
            final = d4_apply_to_state(D4Element.R, state)
            final = d4_apply_to_state(D4Element.R, final)
            final = d4_apply_to_state(D4Element.R, final)
            final = d4_apply_to_state(D4Element.R, final)
            assert final == state, f"r⁴({state}) should equal {state}"

    def test_s_squared_is_identity(self):
        """s² = e: Two reflections return to identity."""
        s = D4Element.S
        result = d4_multiply(s, s)
        assert result == D4Element.E

        # Also verify on states
        for state in HohfeldianState:
            final = d4_apply_to_state(D4Element.S, state)
            final = d4_apply_to_state(D4Element.S, final)
            assert final == state, f"s²({state}) should equal {state}"

    def test_srs_equals_r_inverse(self):
        """srs = r⁻¹: The key non-abelian relation."""
        s = D4Element.S
        r = D4Element.R

        # Compute srs
        srs = d4_multiply(s, d4_multiply(r, s))

        # Compute r⁻¹ = r³
        r_inv = d4_inverse(r)

        assert srs == r_inv, f"srs = {srs}, but r⁻¹ = {r_inv}"
        assert r_inv == D4Element.R3, "r⁻¹ should be r³"

    def test_all_reflections_are_involutions(self):
        """All reflection elements (s, sr, sr², sr³) are self-inverse."""
        reflections = [D4Element.S, D4Element.SR, D4Element.SR2, D4Element.SR3]
        for refl in reflections:
            result = d4_multiply(refl, refl)
            assert result == D4Element.E, f"{refl}² should be identity"


class TestNonAbelianStructure:
    """Test that D4 is non-abelian (rs ≠ sr)."""

    def test_rs_not_equal_sr(self):
        """rs ≠ sr: Rotation and reflection do not commute."""
        r = D4Element.R
        s = D4Element.S

        rs = d4_multiply(r, s)
        sr = d4_multiply(s, r)

        assert rs != sr, f"D4 should be non-abelian, but rs = sr = {rs}"

    def test_rs_is_sr3(self):
        """rs = sr³ (specific relation)."""
        r = D4Element.R
        s = D4Element.S

        rs = d4_multiply(r, s)
        assert rs == D4Element.SR3, f"rs should be sr³, got {rs}"

    def test_sr_composition(self):
        """sr is a distinct element."""
        s = D4Element.S
        r = D4Element.R

        sr = d4_multiply(s, r)
        assert sr == D4Element.SR


class TestSemanticMappings:
    """
    Test that our semantic interpretations are consistent with D4.

    Conventions:
    - r (rotation): O → C → L → N → O
    - s (reflection/correlative): O ↔ C, L ↔ N
    - r² (negation): O ↔ L, C ↔ N
    """

    def test_rotation_cycle(self):
        """r: O → C → L → N → O."""
        assert d4_apply_to_state(D4Element.R, HohfeldianState.O) == HohfeldianState.C
        assert d4_apply_to_state(D4Element.R, HohfeldianState.C) == HohfeldianState.L
        assert d4_apply_to_state(D4Element.R, HohfeldianState.L) == HohfeldianState.N
        assert d4_apply_to_state(D4Element.R, HohfeldianState.N) == HohfeldianState.O

    def test_correlative_is_s(self):
        """s (correlative): O ↔ C, L ↔ N."""
        # Test via correlative function
        assert correlative(HohfeldianState.O) == HohfeldianState.C
        assert correlative(HohfeldianState.C) == HohfeldianState.O
        assert correlative(HohfeldianState.L) == HohfeldianState.N
        assert correlative(HohfeldianState.N) == HohfeldianState.L

        # Test via d4_apply_to_state with S element
        assert d4_apply_to_state(D4Element.S, HohfeldianState.O) == HohfeldianState.C
        assert d4_apply_to_state(D4Element.S, HohfeldianState.C) == HohfeldianState.O
        assert d4_apply_to_state(D4Element.S, HohfeldianState.L) == HohfeldianState.N
        assert d4_apply_to_state(D4Element.S, HohfeldianState.N) == HohfeldianState.L

    def test_negation_is_r2(self):
        """r² (negation): O ↔ L, C ↔ N."""
        assert d4_apply_to_state(D4Element.R2, HohfeldianState.O) == HohfeldianState.L
        assert d4_apply_to_state(D4Element.R2, HohfeldianState.L) == HohfeldianState.O
        assert d4_apply_to_state(D4Element.R2, HohfeldianState.C) == HohfeldianState.N
        assert d4_apply_to_state(D4Element.R2, HohfeldianState.N) == HohfeldianState.C

    def test_semantic_gate_consistency(self):
        """
        Verify semantic gates map to correct D4 elements.

        - "only if convenient" (obligation release): O → L requires r²
        - "I promise" (liberty binding): L → O requires r²
        - Perspective shift A↔B: requires s
        """
        # Obligation release: O → L
        assert d4_apply_to_state(D4Element.R2, HohfeldianState.O) == HohfeldianState.L

        # Liberty binding: L → O
        assert d4_apply_to_state(D4Element.R2, HohfeldianState.L) == HohfeldianState.O

        # Perspective shift (correlative)
        assert d4_apply_to_state(D4Element.S, HohfeldianState.O) == HohfeldianState.C

    def test_quarter_turn_semantics(self):
        """
        r (quarter-turn) operations for non-abelian verification.

        - "you have every right to refuse" might shift L → C (one rotation)
        - "they cannot demand" might shift C → N (one rotation)

        These operations, if empirically verified, prove non-abelian structure.
        """
        # L → C via r? Let's check
        # r: O→C, C→L, L→N, N→O
        # So L → N via r, not L → C

        # Actually, r³: L → C
        assert d4_apply_to_state(D4Element.R3, HohfeldianState.L) == HohfeldianState.C

        # And r: C → L, not C → N
        # r: C → L
        assert d4_apply_to_state(D4Element.R, HohfeldianState.C) == HohfeldianState.L


class TestAbelianSubgroup:
    """
    Test that {e, r², s, sr²} forms an abelian subgroup (Klein four-group V₄).

    This addresses the critique that if only r² and s are empirically observed,
    we have only demonstrated an abelian subgroup.
    """

    def test_klein_four_subgroup(self):
        """The subgroup {e, r², s, sr²} is abelian."""
        elements = [D4Element.E, D4Element.R2, D4Element.S, D4Element.SR2]

        # All pairs should commute
        for a in elements:
            for b in elements:
                ab = d4_multiply(a, b)
                ba = d4_multiply(b, a)
                assert ab == ba, f"{a} and {b} should commute in V₄ subgroup"

    def test_subgroup_closed(self):
        """The subgroup {e, r², s, sr²} is closed under multiplication."""
        elements = {D4Element.E, D4Element.R2, D4Element.S, D4Element.SR2}

        for a in elements:
            for b in elements:
                product = d4_multiply(a, b)
                assert product in elements, f"{a} * {b} = {product} not in subgroup"

    def test_escaping_abelian_subgroup(self):
        """
        To prove non-abelian structure, we need operations outside {e, r², s, sr²}.

        Specifically, we need r, r³, sr, or sr³.
        """
        abelian_subgroup = {D4Element.E, D4Element.R2, D4Element.S, D4Element.SR2}
        full_group = set(D4Element)

        non_abelian_elements = full_group - abelian_subgroup

        # These elements don't commute with s
        for elem in [D4Element.R, D4Element.R3]:
            rs = d4_multiply(elem, D4Element.S)
            sr = d4_multiply(D4Element.S, elem)
            assert rs != sr, f"{elem} should not commute with s"


class TestGroupLawVerificationProtocol:
    """
    Tests that would be run in the empirical group law verification experiment.

    These simulate what we would measure with LLM subjects.
    """

    def test_four_rotations_return_to_start(self):
        """
        Experimental prediction: Applying a 'quarter-turn' gate 4 times
        should return the state distribution to the original.
        """
        for initial in HohfeldianState:
            state = initial
            for _ in range(4):
                state = d4_apply_to_state(D4Element.R, state)
            assert state == initial

    def test_double_perspective_swap_returns(self):
        """
        Experimental prediction: Two perspective swaps return to original.
        """
        for initial in HohfeldianState:
            state = initial
            state = d4_apply_to_state(D4Element.S, state)  # First swap
            state = d4_apply_to_state(D4Element.S, state)  # Second swap
            assert state == initial

    def test_srs_matches_r_inverse(self):
        """
        Experimental prediction: The sequence swap-rotate-swap should
        produce the same final state as three reverse rotations.
        """
        for initial in HohfeldianState:
            # Path 1: srs
            state1 = initial
            state1 = d4_apply_to_state(D4Element.S, state1)
            state1 = d4_apply_to_state(D4Element.R, state1)
            state1 = d4_apply_to_state(D4Element.S, state1)

            # Path 2: r³ = r⁻¹
            state2 = d4_apply_to_state(D4Element.R3, initial)

            assert state1 == state2, f"srs({initial}) = {state1}, r³({initial}) = {state2}"

    def test_path_dependence_signature(self):
        """
        Experimental prediction: rs and sr paths produce different results.
        This is THE signature of non-abelian structure.
        """
        for initial in HohfeldianState:
            # Path 1: rs (rotate then swap)
            state_rs = d4_apply_to_state(D4Element.R, initial)
            state_rs = d4_apply_to_state(D4Element.S, state_rs)

            # Path 2: sr (swap then rotate)
            state_sr = d4_apply_to_state(D4Element.S, initial)
            state_sr = d4_apply_to_state(D4Element.R, state_sr)

            # These should differ for at least some initial states
            # (they differ for all states in D4)

        # Verify rs ≠ sr as group elements (already tested above)
        rs = d4_multiply(D4Element.R, D4Element.S)
        sr = d4_multiply(D4Element.S, D4Element.R)
        assert rs != sr
