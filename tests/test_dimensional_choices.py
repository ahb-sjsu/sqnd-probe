# Copyright (c) 2026 Andrew H. Bond
# Tests for dimensional choice generation

"""Tests for the dimensional justification system."""

import pytest

from dear_ethicist.dimensional_choices import (
    CLAIM_JUSTIFICATIONS,
    LIBERTY_JUSTIFICATIONS,
    NO_CLAIM_JUSTIFICATIONS,
    OBLIGATION_JUSTIFICATIONS,
    DimensionalSignature,
    DimensionalVerdict,
    EthicalDimension,
    Justification,
    detect_scenario_type,
    get_justifications_for_letter,
    get_justifications_for_state,
)
from dear_ethicist.models import HohfeldianState, Letter, Party, Protocol


class TestEthicalDimension:
    """Tests for EthicalDimension enum."""

    def test_all_nine_dimensions_exist(self):
        """Verify all 9 DEME dimensions are defined."""
        assert len(EthicalDimension) == 9
        expected = {
            "harm",
            "rights",
            "fairness",
            "autonomy",
            "legitimacy",
            "epistemic",
            "privacy",
            "societal",
            "procedural",
        }
        actual = {d.value for d in EthicalDimension}
        assert actual == expected


class TestDimensionalSignature:
    """Tests for DimensionalSignature model."""

    def test_default_signature_is_zero(self):
        """Default signature should have all zeros."""
        sig = DimensionalSignature()
        assert sig.harm == 0.0
        assert sig.autonomy == 0.0
        assert sig.legitimacy == 0.0

    def test_dominant_dimension(self):
        """Should return dimension with highest weight."""
        sig = DimensionalSignature(legitimacy=0.8, harm=0.2)
        assert sig.dominant() == EthicalDimension.LEGITIMACY

    def test_dominant_with_tie_returns_first(self):
        """When tied, returns one of the highest (deterministic)."""
        sig = DimensionalSignature(harm=0.5, fairness=0.5)
        # Just verify it returns one of the tied values
        assert sig.dominant() in {EthicalDimension.HARM, EthicalDimension.FAIRNESS}


class TestJustification:
    """Tests for Justification model."""

    def test_justification_has_label_and_dimensions(self):
        """Justification should store label and dimensional signature."""
        j = Justification(label="Test justification", dimensions=DimensionalSignature(autonomy=0.9))
        assert j.label == "Test justification"
        assert j.dimensions.autonomy == 0.9


class TestDetectScenarioType:
    """Tests for scenario detection from letters."""

    def test_detect_promise_from_protocol_params(self):
        """Should detect promise scenario from series param."""
        letter = Letter(
            letter_id="test",
            protocol=Protocol.GATE_DETECTION,
            protocol_params={"series": "promise_release"},
            signoff="TEST",
            subject="Test",
            body="Some body text",
        )
        assert detect_scenario_type(letter) == "promise"

    def test_detect_debt_from_protocol_params(self):
        """Should detect debt scenario from series param."""
        letter = Letter(
            letter_id="test",
            protocol=Protocol.GATE_DETECTION,
            protocol_params={"series": "loan_repayment"},
            signoff="TEST",
            subject="Test",
            body="Some body text",
        )
        assert detect_scenario_type(letter) == "debt"

    def test_detect_secret_from_scenario_param(self):
        """Should detect secret scenario from scenario param."""
        letter = Letter(
            letter_id="test",
            protocol=Protocol.CORRELATIVE,
            protocol_params={"scenario": "secret"},
            signoff="TEST",
            subject="Test",
            body="Some body text",
        )
        assert detect_scenario_type(letter) == "secret"

    def test_detect_promise_from_body_text(self):
        """Should detect promise scenario from body keywords."""
        letter = Letter(
            letter_id="test",
            protocol=Protocol.CORRELATIVE,
            signoff="TEST",
            subject="Test",
            body="My friend promised to help me move.",
        )
        assert detect_scenario_type(letter) == "promise"

    def test_detect_debt_from_body_text(self):
        """Should detect debt scenario from body keywords."""
        letter = Letter(
            letter_id="test",
            protocol=Protocol.CORRELATIVE,
            signoff="TEST",
            subject="Test",
            body="I lent them $500 last month.",
        )
        assert detect_scenario_type(letter) == "debt"

    def test_detect_secret_from_body_text(self):
        """Should detect secret scenario from body keywords."""
        letter = Letter(
            letter_id="test",
            protocol=Protocol.CORRELATIVE,
            signoff="TEST",
            subject="Test",
            body="They told me a secret about their marriage.",
        )
        assert detect_scenario_type(letter) == "secret"

    def test_falls_back_to_generic(self):
        """Should return generic when no pattern matches."""
        letter = Letter(
            letter_id="test",
            protocol=Protocol.CORRELATIVE,
            signoff="TEST",
            subject="Test",
            body="Something happened and I don't know what to do.",
        )
        assert detect_scenario_type(letter) == "generic"


class TestGetJustificationsForState:
    """Tests for getting justifications by Hohfeldian state."""

    def test_obligation_justifications_exist(self):
        """Should return justifications for O state."""
        justifications = get_justifications_for_state(HohfeldianState.O, "promise")
        assert len(justifications) > 0
        assert all(isinstance(j, Justification) for j in justifications)

    def test_liberty_justifications_exist(self):
        """Should return justifications for L state."""
        justifications = get_justifications_for_state(HohfeldianState.L, "promise")
        assert len(justifications) > 0

    def test_claim_justifications_exist(self):
        """Should return justifications for C state."""
        justifications = get_justifications_for_state(HohfeldianState.C, "promise")
        assert len(justifications) > 0

    def test_no_claim_justifications_exist(self):
        """Should return justifications for N state."""
        justifications = get_justifications_for_state(HohfeldianState.N, "promise")
        assert len(justifications) > 0

    def test_falls_back_to_generic_scenario(self):
        """Should use generic justifications for unknown scenario."""
        justifications = get_justifications_for_state(HohfeldianState.O, "unknown_scenario")
        assert len(justifications) > 0
        # Should match generic justifications
        generic = OBLIGATION_JUSTIFICATIONS["generic"]
        assert justifications == generic


class TestGetJustificationsForLetter:
    """Tests for getting justifications from a letter."""

    def test_returns_justifications_for_promise_letter(self):
        """Should detect scenario and return appropriate justifications."""
        letter = Letter(
            letter_id="test",
            protocol=Protocol.GATE_DETECTION,
            protocol_params={"series": "promise_release"},
            signoff="TEST",
            subject="Test",
            body="They promised to help.",
            parties=[
                Party(name="You", role="writer", is_writer=True),
                Party(name="Friend", role="friend"),
            ],
        )
        justifications = get_justifications_for_letter(letter, "Friend", HohfeldianState.O)
        assert len(justifications) > 0
        # Should be promise-specific justifications
        assert justifications == OBLIGATION_JUSTIFICATIONS["promise"]

    def test_returns_liberty_justifications(self):
        """Should return L justifications when L is chosen."""
        letter = Letter(
            letter_id="test",
            protocol=Protocol.GATE_DETECTION,
            signoff="TEST",
            subject="Test",
            body="They promised to help.",
        )
        justifications = get_justifications_for_letter(letter, "Friend", HohfeldianState.L)
        assert len(justifications) > 0
        assert justifications == LIBERTY_JUSTIFICATIONS["promise"]


class TestDimensionalVerdict:
    """Tests for DimensionalVerdict model."""

    def test_verdict_stores_all_fields(self):
        """Verdict should store party, state, justification, and dimensions."""
        verdict = DimensionalVerdict(
            party_name="Taylor",
            state=HohfeldianState.O,
            justification="They looked you in the eye and said they'd be there",
            dimensions=DimensionalSignature(legitimacy=0.8, societal=0.2),
            expected=HohfeldianState.O,
        )
        assert verdict.party_name == "Taylor"
        assert verdict.state == HohfeldianState.O
        assert verdict.justification == "They looked you in the eye and said they'd be there"
        assert verdict.dimensions.legitimacy == 0.8

    def test_is_hohfeld_correct_when_matching(self):
        """Should return True when state matches expected."""
        verdict = DimensionalVerdict(
            party_name="Taylor",
            state=HohfeldianState.O,
            justification="Test",
            dimensions=DimensionalSignature(),
            expected=HohfeldianState.O,
        )
        assert verdict.is_hohfeld_correct is True

    def test_is_hohfeld_correct_when_not_matching(self):
        """Should return False when state doesn't match expected."""
        verdict = DimensionalVerdict(
            party_name="Taylor",
            state=HohfeldianState.L,
            justification="Test",
            dimensions=DimensionalSignature(),
            expected=HohfeldianState.O,
        )
        assert verdict.is_hohfeld_correct is False

    def test_is_hohfeld_correct_when_no_expected(self):
        """Should return None when no expected state is set."""
        verdict = DimensionalVerdict(
            party_name="Taylor",
            state=HohfeldianState.O,
            justification="Test",
            dimensions=DimensionalSignature(),
        )
        assert verdict.is_hohfeld_correct is None


class TestJustificationQuality:
    """Tests ensuring justifications are well-formed."""

    @pytest.mark.parametrize("scenario", ["promise", "debt", "secret", "favor", "generic"])
    def test_obligation_justifications_have_content(self, scenario):
        """Each O justification should have non-empty label and non-zero dimensions."""
        justifications = OBLIGATION_JUSTIFICATIONS[scenario]
        for j in justifications:
            assert len(j.label) > 10, f"Label too short: {j.label}"
            # At least one dimension should be non-zero
            dims = j.dimensions
            total = sum(
                [
                    dims.harm,
                    dims.rights,
                    dims.fairness,
                    dims.autonomy,
                    dims.legitimacy,
                    dims.epistemic,
                    dims.privacy,
                    dims.societal,
                    dims.procedural,
                ]
            )
            assert total > 0, f"No dimensions set for: {j.label}"

    @pytest.mark.parametrize("scenario", ["promise", "debt", "secret", "favor", "generic"])
    def test_liberty_justifications_have_content(self, scenario):
        """Each L justification should have non-empty label and non-zero dimensions."""
        justifications = LIBERTY_JUSTIFICATIONS[scenario]
        for j in justifications:
            assert len(j.label) > 10, f"Label too short: {j.label}"
            dims = j.dimensions
            total = sum(
                [
                    dims.harm,
                    dims.rights,
                    dims.fairness,
                    dims.autonomy,
                    dims.legitimacy,
                    dims.epistemic,
                    dims.privacy,
                    dims.societal,
                    dims.procedural,
                ]
            )
            assert total > 0, f"No dimensions set for: {j.label}"

    @pytest.mark.parametrize("scenario", ["promise", "debt", "secret", "favor", "generic"])
    def test_claim_justifications_have_content(self, scenario):
        """Each C justification should have non-empty label and non-zero dimensions."""
        justifications = CLAIM_JUSTIFICATIONS[scenario]
        for j in justifications:
            assert len(j.label) > 10, f"Label too short: {j.label}"
            dims = j.dimensions
            total = sum(
                [
                    dims.harm,
                    dims.rights,
                    dims.fairness,
                    dims.autonomy,
                    dims.legitimacy,
                    dims.epistemic,
                    dims.privacy,
                    dims.societal,
                    dims.procedural,
                ]
            )
            assert total > 0, f"No dimensions set for: {j.label}"

    @pytest.mark.parametrize("scenario", ["promise", "debt", "secret", "favor", "generic"])
    def test_no_claim_justifications_have_content(self, scenario):
        """Each N justification should have non-empty label and non-zero dimensions."""
        justifications = NO_CLAIM_JUSTIFICATIONS[scenario]
        for j in justifications:
            assert len(j.label) > 10, f"Label too short: {j.label}"
            dims = j.dimensions
            total = sum(
                [
                    dims.harm,
                    dims.rights,
                    dims.fairness,
                    dims.autonomy,
                    dims.legitimacy,
                    dims.epistemic,
                    dims.privacy,
                    dims.societal,
                    dims.procedural,
                ]
            )
            assert total > 0, f"No dimensions set for: {j.label}"

    def test_each_scenario_has_multiple_justifications(self):
        """Each scenario should have at least 2 justifications per state."""
        for scenario in ["promise", "debt", "secret", "favor", "generic"]:
            assert len(OBLIGATION_JUSTIFICATIONS[scenario]) >= 2
            assert len(LIBERTY_JUSTIFICATIONS[scenario]) >= 2
            assert len(CLAIM_JUSTIFICATIONS[scenario]) >= 2
            assert len(NO_CLAIM_JUSTIFICATIONS[scenario]) >= 2
