# Copyright (c) 2026 Andrew H. Bond and Claude Opus 4.5
# Department of Computer Engineering, San Jose State University
# Licensed under the MIT License. See LICENSE file for details.

"""Tests for Dear Ethicist data models."""

from dear_ethicist.models import (
    D4Element,
    HohfeldianState,
    Verdict,
    correlative,
    d4_apply_to_state,
    d4_inverse,
    d4_multiply,
)


class TestHohfeldianState:
    """Tests for HohfeldianState enum."""

    def test_four_states_exist(self):
        """Verify all four Hohfeldian states are defined."""
        assert HohfeldianState.O.value == "O"
        assert HohfeldianState.C.value == "C"
        assert HohfeldianState.L.value == "L"
        assert HohfeldianState.N.value == "N"


class TestD4Operations:
    """Tests for D4 group operations."""

    def test_identity(self):
        """e * x = x for all x."""
        for element in D4Element:
            assert d4_multiply(D4Element.E, element) == element

    def test_r_fourth_power(self):
        """r^4 = e."""
        result = D4Element.E
        for _ in range(4):
            result = d4_multiply(result, D4Element.R)
        assert result == D4Element.E

    def test_s_squared(self):
        """s^2 = e."""
        assert d4_multiply(D4Element.S, D4Element.S) == D4Element.E

    def test_inverses(self):
        """x * x^-1 = e for all x."""
        for element in D4Element:
            inv = d4_inverse(element)
            assert d4_multiply(element, inv) == D4Element.E

    def test_non_abelian(self):
        """D4 is non-abelian: sr != rs."""
        sr = d4_multiply(D4Element.S, D4Element.R)
        rs = d4_multiply(D4Element.R, D4Element.S)
        assert sr != rs

    def test_rotation_cycles_states(self):
        """r: O -> C -> L -> N -> O."""
        assert d4_apply_to_state(D4Element.R, HohfeldianState.O) == HohfeldianState.C
        assert d4_apply_to_state(D4Element.R, HohfeldianState.C) == HohfeldianState.L
        assert d4_apply_to_state(D4Element.R, HohfeldianState.L) == HohfeldianState.N
        assert d4_apply_to_state(D4Element.R, HohfeldianState.N) == HohfeldianState.O

    def test_reflection_correlatives(self):
        """s: O <-> C, L <-> N."""
        assert d4_apply_to_state(D4Element.S, HohfeldianState.O) == HohfeldianState.C
        assert d4_apply_to_state(D4Element.S, HohfeldianState.C) == HohfeldianState.O
        assert d4_apply_to_state(D4Element.S, HohfeldianState.L) == HohfeldianState.N
        assert d4_apply_to_state(D4Element.S, HohfeldianState.N) == HohfeldianState.L


class TestCorrelative:
    """Tests for correlative function."""

    def test_o_c_correlative(self):
        """O and C are correlatives."""
        assert correlative(HohfeldianState.O) == HohfeldianState.C
        assert correlative(HohfeldianState.C) == HohfeldianState.O

    def test_l_n_correlative(self):
        """L and N are correlatives."""
        assert correlative(HohfeldianState.L) == HohfeldianState.N
        assert correlative(HohfeldianState.N) == HohfeldianState.L


class TestLetter:
    """Tests for Letter model."""

    def test_get_writer(self, sample_letter):
        """Test getting the letter writer."""
        writer = sample_letter.get_writer()
        assert writer is not None
        assert writer.name == "You"
        assert writer.is_writer is True

    def test_get_other_party(self, sample_letter):
        """Test getting the other party."""
        other = sample_letter.get_other_party()
        assert other is not None
        assert other.name == "Morgan"
        assert other.is_writer is False


class TestVerdict:
    """Tests for Verdict model."""

    def test_is_correct_when_matching(self):
        """Verdict is correct when state matches expected."""
        verdict = Verdict(
            party_name="Test",
            state=HohfeldianState.O,
            expected=HohfeldianState.O,
        )
        assert verdict.is_correct is True

    def test_is_correct_when_not_matching(self):
        """Verdict is incorrect when state doesn't match expected."""
        verdict = Verdict(
            party_name="Test",
            state=HohfeldianState.O,
            expected=HohfeldianState.L,
        )
        assert verdict.is_correct is False

    def test_is_correct_when_no_expected(self):
        """Verdict correctness is None when no expected value."""
        verdict = Verdict(
            party_name="Test",
            state=HohfeldianState.O,
            expected=None,
        )
        assert verdict.is_correct is None
