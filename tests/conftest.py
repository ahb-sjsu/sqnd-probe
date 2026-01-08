"""Pytest configuration and fixtures for Dear Ethicist tests."""

import pytest

from dear_ethicist.models import (
    HohfeldianState,
    Letter,
    LetterVoice,
    Party,
    Protocol,
    Response,
    Verdict,
)


@pytest.fixture
def sample_letter() -> Letter:
    """Create a sample letter for testing."""
    return Letter(
        letter_id="test_letter",
        protocol=Protocol.GATE_DETECTION,
        protocol_params={"level": 5},
        signoff="TEST WRITER",
        subject="Test subject",
        body="Dear Ethicist,\n\nTest body.\n\n— TEST WRITER",
        voice=LetterVoice.GENUINELY_STUCK,
        parties=[
            Party(name="You", role="writer", is_writer=True, expected_state=HohfeldianState.C),
            Party(name="Morgan", role="friend", expected_state=HohfeldianState.O),
        ],
        questions=["Test question?"],
    )


@pytest.fixture
def sample_verdict() -> Verdict:
    """Create a sample verdict for testing."""
    return Verdict(
        party_name="Morgan",
        state=HohfeldianState.L,
        expected=HohfeldianState.L,
    )


@pytest.fixture
def sample_response(sample_verdict: Verdict) -> Response:
    """Create a sample response for testing."""
    return Response(
        letter_id="test_letter",
        verdicts=[sample_verdict],
    )
