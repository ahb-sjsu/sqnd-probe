"""
Reader reaction generation for Dear Ethicist.

Generates realistic "reader comments" based on the player's verdicts.
These provide feedback without indicating right/wrong answers.
"""

import random

from dear_ethicist.models import (
    FollowUp,
    Letter,
    ReaderReaction,
    Response,
)

# =============================================================================
# REACTION TEMPLATES
# =============================================================================

SUPPORTIVE_TEMPLATES = [
    "@{username}: Finally, someone with common sense!",
    "@{username}: This is exactly what I needed to hear.",
    "@{username}: You nailed it. Forwarding to my {relation}.",
    "@{username}: 100% agree. Clear and fair.",
    "@{username}: The Ethicist speaks truth once again.",
]

CRITICAL_TEMPLATES = [
    "@{username}: Eh, I think you're being too harsh here.",
    "@{username}: Disagree. Life is more complicated than this.",
    "@{username}: Easy to say from the outside...",
    "@{username}: This advice would NOT work in the real world.",
    "@{username}: I expected more nuance from this column.",
]

MIXED_TEMPLATES = [
    "@{username}: I see both sides, but lean toward the opposite view.",
    "@{username}: Good points, though I'm not fully convinced.",
    "@{username}: Interesting take. My therapist would disagree.",
    "@{username}: Right answer, wrong reasoning?",
]

HUMOROUS_TEMPLATES = [
    "@{username}: Morgan/Jamie/whoever sounds like my ex lol",
    "@{username}: Can you answer my letter about whether I owe my cat an apology?",
    "@{username}: *takes notes for next family dinner*",
    "@{username}: The real question is why anyone promises anything ever",
]

USERNAMES = [
    "MidwestMom47",
    "DevilsAdvocate",
    "RealistRick",
    "HopefulHannah",
    "SkepticalSam",
    "WisdomSeeker",
    "TruthTeller99",
    "JustMyOpinion",
    "BeenThereDoneThat",
    "NeutralNancy",
    "FairIsFair",
    "CommonSense101",
    "PhilosophyMajor",
    "RealTalk",
    "ThinkingItThrough",
    "VoiceOfReason",
]

RELATIONS = ["sister", "brother", "friend", "coworker", "neighbor", "roommate"]


def generate_reactions(letter: Letter, response: Response, count: int = 3) -> list[ReaderReaction]:
    """
    Generate reader reactions to a published response.

    Mix of supportive, critical, mixed, and humorous reactions.
    Does NOT indicate whether the player was "right."
    """
    reactions = []
    used_usernames = set()

    # Determine reaction distribution based on verdict confidence
    # More controversial verdicts get more mixed reactions
    distribution = _get_reaction_distribution(letter, response)

    for _ in range(count):
        tone = random.choices(
            ["supportive", "critical", "mixed", "humorous"], weights=distribution, k=1
        )[0]

        # Pick unused username
        available = [u for u in USERNAMES if u not in used_usernames]
        if not available:
            available = USERNAMES
        username = random.choice(available)
        used_usernames.add(username)

        # Generate reaction text
        text = _generate_reaction_text(tone, username, letter, response)

        reactions.append(
            ReaderReaction(
                username=username,
                text=text,
                tone=tone,
            )
        )

    return reactions


def _get_reaction_distribution(_letter: Letter, response: Response) -> list[float]:
    """
    Get probability distribution for reaction tones.

    Returns weights for [supportive, critical, mixed, humorous].
    """
    # Default: balanced mix
    weights = [0.35, 0.25, 0.25, 0.15]

    # Check if verdict matches expected (if known)
    verdicts_match = True
    for verdict in response.verdicts:
        if verdict.expected and verdict.state != verdict.expected:
            verdicts_match = False
            break

    if not verdicts_match:
        # More critical and mixed reactions for "unexpected" verdicts
        weights = [0.20, 0.35, 0.30, 0.15]

    return weights


def _generate_reaction_text(
    tone: str, username: str, _letter: Letter, _response: Response
) -> str:
    """Generate reaction text based on tone."""
    if tone == "supportive":
        template = random.choice(SUPPORTIVE_TEMPLATES)
    elif tone == "critical":
        template = random.choice(CRITICAL_TEMPLATES)
    elif tone == "mixed":
        template = random.choice(MIXED_TEMPLATES)
    else:
        template = random.choice(HUMOROUS_TEMPLATES)

    return template.format(
        username=username,
        relation=random.choice(RELATIONS),
    )


def generate_followup(letter: Letter, response: Response) -> FollowUp | None:
    """
    Generate a follow-up letter from the original writer.

    Only some letters get follow-ups (adds engagement).
    """
    # 30% chance of follow-up
    if random.random() > 0.3:
        return None

    # Determine outcome based on verdict alignment
    verdicts_match_expected = all(
        v.expected is None or v.state == v.expected for v in response.verdicts
    )

    if verdicts_match_expected:
        # Positive outcome
        outcomes = [
            f"I took your advice and talked to {letter.get_other_party().name if letter.get_other_party() else 'them'}. We worked it out!",
            "Thank you! Your clarity helped me see the situation more clearly.",
            "Update: Everything resolved itself. You were right about the core issue.",
        ]
        satisfaction = "grateful"
    else:
        # Mixed outcome - not wrong, just different
        outcomes = [
            "I tried your approach. Results were... mixed. But I learned something.",
            "Update: The situation is still complicated. But your perspective helped.",
            "Things didn't go as planned, but I don't regret how I handled it.",
        ]
        satisfaction = "neutral"

    return FollowUp(
        original_letter_id=letter.letter_id,
        outcome_text=random.choice(outcomes),
        writer_satisfaction=satisfaction,
    )


def generate_engagement_stats() -> dict:
    """Generate fake engagement statistics."""
    comments = random.randint(200, 2000)
    shares = random.randint(comments // 2, comments * 3)

    return {
        "comments": comments,
        "shares": shares,
        "saves": random.randint(50, 500),
    }
