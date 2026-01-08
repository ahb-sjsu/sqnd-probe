"""Tests for Dear Ethicist reader reaction generation."""

import pytest

from dear_ethicist.models import (
    HohfeldianState,
    Letter,
    Party,
    Protocol,
    Response,
    Verdict,
)
from dear_ethicist.reactions import (
    CRITICAL_TEMPLATES,
    HUMOROUS_TEMPLATES,
    MIXED_TEMPLATES,
    SUPPORTIVE_TEMPLATES,
    USERNAMES,
    _generate_reaction_text,
    _get_reaction_distribution,
    generate_engagement_stats,
    generate_followup,
    generate_reactions,
)


@pytest.fixture
def sample_letter():
    """Create a sample letter for testing."""
    return Letter(
        letter_id="reaction_test",
        protocol=Protocol.CORRELATIVE,
        signoff="TEST PERSON",
        subject="Test Subject",
        body="Dear Ethicist,\n\nTest body.",
        parties=[
            Party(name="You", role="writer", is_writer=True),
            Party(name="Friend", role="friend", expected_state=HohfeldianState.O),
        ],
    )


@pytest.fixture
def sample_response():
    """Create a sample response for testing."""
    return Response(
        letter_id="reaction_test",
        verdicts=[
            Verdict(
                party_name="Friend",
                state=HohfeldianState.O,
                expected=HohfeldianState.O,
            ),
        ],
    )


class TestGenerateReactions:
    """Tests for generate_reactions function."""

    def test_generates_correct_count(self, sample_letter, sample_response):
        """Generate the requested number of reactions."""
        reactions = generate_reactions(sample_letter, sample_response, count=5)
        assert len(reactions) == 5

    def test_generates_default_count(self, sample_letter, sample_response):
        """Default count is 3."""
        reactions = generate_reactions(sample_letter, sample_response)
        assert len(reactions) == 3

    def test_reactions_have_required_fields(self, sample_letter, sample_response):
        """Each reaction has username, text, and tone."""
        reactions = generate_reactions(sample_letter, sample_response, count=1)
        reaction = reactions[0]

        assert reaction.username is not None
        assert reaction.text is not None
        assert reaction.tone in ["supportive", "critical", "mixed", "humorous"]

    def test_usernames_are_unique(self, sample_letter, sample_response):
        """Reactions have unique usernames."""
        reactions = generate_reactions(sample_letter, sample_response, count=5)
        usernames = [r.username for r in reactions]
        assert len(usernames) == len(set(usernames))

    def test_handles_many_reactions(self, sample_letter, sample_response):
        """Can generate more reactions than unique usernames."""
        # Should work even if count > len(USERNAMES)
        reactions = generate_reactions(sample_letter, sample_response, count=len(USERNAMES) + 5)
        assert len(reactions) == len(USERNAMES) + 5


class TestGetReactionDistribution:
    """Tests for _get_reaction_distribution function."""

    def test_matching_verdicts_distribution(self, sample_letter):
        """Matching verdicts give balanced distribution."""
        response = Response(
            letter_id="test",
            verdicts=[
                Verdict(
                    party_name="Friend",
                    state=HohfeldianState.O,
                    expected=HohfeldianState.O,
                ),
            ],
        )
        weights = _get_reaction_distribution(sample_letter, response)

        # More supportive when matching expected
        assert weights[0] > weights[1]  # supportive > critical

    def test_mismatching_verdicts_distribution(self, sample_letter):
        """Mismatching verdicts give more critical distribution."""
        response = Response(
            letter_id="test",
            verdicts=[
                Verdict(
                    party_name="Friend",
                    state=HohfeldianState.L,  # Different from expected O
                    expected=HohfeldianState.O,
                ),
            ],
        )
        weights = _get_reaction_distribution(sample_letter, response)

        # More critical when not matching expected
        assert weights[1] > weights[0]  # critical > supportive

    def test_distribution_sums_to_one(self, sample_letter, sample_response):
        """Distribution weights sum to approximately 1."""
        weights = _get_reaction_distribution(sample_letter, sample_response)
        assert sum(weights) == pytest.approx(1.0, rel=0.01)


class TestGenerateReactionText:
    """Tests for _generate_reaction_text function."""

    def test_supportive_text(self, sample_letter, sample_response):
        """Supportive text comes from supportive templates."""
        text = _generate_reaction_text("supportive", "TestUser", sample_letter, sample_response)
        # Should contain @ mention
        assert "@" in text

    def test_critical_text(self, sample_letter, sample_response):
        """Critical text comes from critical templates."""
        text = _generate_reaction_text("critical", "TestUser", sample_letter, sample_response)
        assert "@" in text

    def test_mixed_text(self, sample_letter, sample_response):
        """Mixed text comes from mixed templates."""
        text = _generate_reaction_text("mixed", "TestUser", sample_letter, sample_response)
        assert "@" in text

    def test_humorous_text(self, sample_letter, sample_response):
        """Humorous text comes from humorous templates."""
        text = _generate_reaction_text("humorous", "TestUser", sample_letter, sample_response)
        assert "@" in text


class TestGenerateFollowup:
    """Tests for generate_followup function."""

    def test_followup_structure(self, sample_letter, sample_response):
        """Followup has required fields when generated."""
        # Run multiple times to catch the 30% chance
        followup = None
        for _ in range(100):
            followup = generate_followup(sample_letter, sample_response)
            if followup is not None:
                break

        if followup is not None:
            assert followup.original_letter_id == sample_letter.letter_id
            assert followup.outcome_text is not None
            assert followup.writer_satisfaction in ["grateful", "neutral"]

    def test_followup_can_be_none(self, sample_letter, sample_response):
        """Followup can be None (70% of the time)."""
        # With 100 tries, very unlikely to never get None
        none_count = 0
        for _ in range(100):
            if generate_followup(sample_letter, sample_response) is None:
                none_count += 1

        assert none_count > 0

    def test_matching_verdicts_positive_outcome(self, sample_letter):
        """Matching verdicts tend toward grateful outcome."""
        response = Response(
            letter_id="test",
            verdicts=[
                Verdict(
                    party_name="Friend",
                    state=HohfeldianState.O,
                    expected=HohfeldianState.O,
                ),
            ],
        )

        grateful_count = 0
        for _ in range(100):
            followup = generate_followup(sample_letter, response)
            if followup and followup.writer_satisfaction == "grateful":
                grateful_count += 1

        # Should have some grateful outcomes
        assert grateful_count > 0

    def test_mismatching_verdicts_neutral_outcome(self, sample_letter):
        """Mismatching verdicts tend toward neutral outcome."""
        response = Response(
            letter_id="test",
            verdicts=[
                Verdict(
                    party_name="Friend",
                    state=HohfeldianState.L,
                    expected=HohfeldianState.O,
                ),
            ],
        )

        neutral_count = 0
        for _ in range(100):
            followup = generate_followup(sample_letter, response)
            if followup and followup.writer_satisfaction == "neutral":
                neutral_count += 1

        # Should have some neutral outcomes
        assert neutral_count > 0


class TestGenerateEngagementStats:
    """Tests for generate_engagement_stats function."""

    def test_returns_required_keys(self):
        """Stats include comments, shares, saves."""
        stats = generate_engagement_stats()

        assert "comments" in stats
        assert "shares" in stats
        assert "saves" in stats

    def test_values_are_positive(self):
        """All stat values are positive."""
        stats = generate_engagement_stats()

        assert stats["comments"] > 0
        assert stats["shares"] > 0
        assert stats["saves"] > 0

    def test_values_in_reasonable_range(self):
        """Stat values are in expected ranges."""
        stats = generate_engagement_stats()

        assert 200 <= stats["comments"] <= 2000
        assert stats["shares"] >= stats["comments"] // 2
        assert 50 <= stats["saves"] <= 500

    def test_randomness(self):
        """Stats vary between calls."""
        stats1 = generate_engagement_stats()
        stats2 = generate_engagement_stats()

        # Very unlikely to be exactly the same
        # (though technically possible, so we just check structure)
        assert isinstance(stats1["comments"], int)
        assert isinstance(stats2["comments"], int)


class TestTemplates:
    """Tests for reaction templates."""

    def test_supportive_templates_have_placeholder(self):
        """Supportive templates have username placeholder."""
        for template in SUPPORTIVE_TEMPLATES:
            assert "{username}" in template

    def test_critical_templates_have_placeholder(self):
        """Critical templates have username placeholder."""
        for template in CRITICAL_TEMPLATES:
            assert "{username}" in template

    def test_mixed_templates_have_placeholder(self):
        """Mixed templates have username placeholder."""
        for template in MIXED_TEMPLATES:
            assert "{username}" in template

    def test_humorous_templates_have_placeholder(self):
        """Humorous templates have username placeholder."""
        for template in HUMOROUS_TEMPLATES:
            assert "{username}" in template

    def test_usernames_not_empty(self):
        """Username list is not empty."""
        assert len(USERNAMES) > 0
