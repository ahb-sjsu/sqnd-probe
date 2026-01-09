# Copyright (c) 2026 Andrew H. Bond
# Extended choice generation for multi-dimensional measurement

"""
Dimensional Justification System

Keep the binary Hohfeldian choice (O vs L, C vs N).
Add a follow-up: "Why?" with justifications that sound like real reasoning.

Design principles:
- Sound like things people actually say, not ethics class answers
- Reveal reasoning style without telegraphing the dimension
- Each justification should feel like a genuine perspective
- Avoid pedantic or obvious framings
"""

from enum import Enum

from pydantic import BaseModel

from dear_ethicist.models import HohfeldianState, Letter


class EthicalDimension(str, Enum):
    """The 9 DEME ethical dimensions."""

    HARM = "harm"
    RIGHTS = "rights"
    FAIRNESS = "fairness"
    AUTONOMY = "autonomy"
    LEGITIMACY = "legitimacy"
    EPISTEMIC = "epistemic"
    PRIVACY = "privacy"
    SOCIETAL = "societal"
    PROCEDURAL = "procedural"


class DimensionalSignature(BaseModel):
    """Dimensional weights (should sum to ~1.0)."""

    harm: float = 0.0
    rights: float = 0.0
    fairness: float = 0.0
    autonomy: float = 0.0
    legitimacy: float = 0.0
    epistemic: float = 0.0
    privacy: float = 0.0
    societal: float = 0.0
    procedural: float = 0.0

    def dominant(self) -> EthicalDimension:
        weights = {d: getattr(self, d.value) for d in EthicalDimension}
        return max(weights, key=weights.get)


class Justification(BaseModel):
    """A justification for a Hohfeldian choice."""

    label: str
    dimensions: DimensionalSignature


# =============================================================================
# JUSTIFICATION TEMPLATES - Written to sound like real human reasoning
# =============================================================================

OBLIGATION_JUSTIFICATIONS = {
    "promise": [
        Justification(
            label="They looked you in the eye and said they'd be there",
            dimensions=DimensionalSignature(legitimacy=0.8, societal=0.2),
        ),
        Justification(
            label="You rearranged your whole week around this",
            dimensions=DimensionalSignature(harm=0.6, fairness=0.4),
        ),
        Justification(
            label="What kind of person just bails when it gets inconvenient?",
            dimensions=DimensionalSignature(societal=0.5, legitimacy=0.3, fairness=0.2),
        ),
        Justification(
            label="If you can't count on this, what can you count on?",
            dimensions=DimensionalSignature(legitimacy=0.6, epistemic=0.4),
        ),
    ],
    "debt": [
        Justification(
            label="They took something that wasn't theirs to keep",
            dimensions=DimensionalSignature(rights=0.6, fairness=0.4),
        ),
        Justification(
            label="You went without so they could have it",
            dimensions=DimensionalSignature(harm=0.5, fairness=0.5),
        ),
        Justification(
            label="They knew exactly what they were agreeing to",
            dimensions=DimensionalSignature(legitimacy=0.7, procedural=0.3),
        ),
    ],
    "secret": [
        Justification(
            label="They came to you because they trusted you specifically",
            dimensions=DimensionalSignature(legitimacy=0.6, privacy=0.4),
        ),
        Justification(
            label="Some things aren't yours to share, period",
            dimensions=DimensionalSignature(privacy=0.8, rights=0.2),
        ),
        Justification(
            label="How would you feel if someone did this to you?",
            dimensions=DimensionalSignature(fairness=0.5, harm=0.3, privacy=0.2),
        ),
    ],
    "favor": [
        Justification(
            label="This is what being close to someone means",
            dimensions=DimensionalSignature(societal=0.7, legitimacy=0.3),
        ),
        Justification(
            label="They'd do it for you without thinking twice",
            dimensions=DimensionalSignature(fairness=0.6, legitimacy=0.4),
        ),
        Justification(
            label="Sometimes you just show up for people",
            dimensions=DimensionalSignature(societal=0.6, harm=0.4),
        ),
    ],
    "generic": [
        Justification(
            label="Words mean something, or they should",
            dimensions=DimensionalSignature(legitimacy=0.8, procedural=0.2),
        ),
        Justification(
            label="Someone's going to get hurt if this falls through",
            dimensions=DimensionalSignature(harm=0.8, fairness=0.2),
        ),
        Justification(
            label="You don't get to just opt out when it suits you",
            dimensions=DimensionalSignature(fairness=0.6, legitimacy=0.4),
        ),
    ],
}

LIBERTY_JUSTIFICATIONS = {
    "promise": [
        Justification(
            label="People aren't contracts—life happens",
            dimensions=DimensionalSignature(autonomy=0.6, epistemic=0.4),
        ),
        Justification(
            label="They couldn't have known things would change like this",
            dimensions=DimensionalSignature(epistemic=0.7, autonomy=0.3),
        ),
        Justification(
            label="Guilt-tripping someone into showing up helps no one",
            dimensions=DimensionalSignature(autonomy=0.5, harm=0.3, societal=0.2),
        ),
        Justification(
            label="A reluctant helper is worse than no helper",
            dimensions=DimensionalSignature(harm=0.4, autonomy=0.4, epistemic=0.2),
        ),
    ],
    "debt": [
        Justification(
            label="Was this ever really a 'loan' or just helping out?",
            dimensions=DimensionalSignature(epistemic=0.7, procedural=0.3),
        ),
        Justification(
            label="You can't squeeze blood from a stone",
            dimensions=DimensionalSignature(harm=0.5, autonomy=0.3, epistemic=0.2),
        ),
        Justification(
            label="Is the money worth what demanding it would cost?",
            dimensions=DimensionalSignature(societal=0.6, harm=0.4),
        ),
    ],
    "secret": [
        Justification(
            label="Someone else is being played here, and they deserve to know",
            dimensions=DimensionalSignature(harm=0.5, epistemic=0.3, fairness=0.2),
        ),
        Justification(
            label="You're not a vault—this is eating at you for a reason",
            dimensions=DimensionalSignature(harm=0.4, autonomy=0.4, epistemic=0.2),
        ),
        Justification(
            label="Loyalty has limits, and this is past them",
            dimensions=DimensionalSignature(fairness=0.5, harm=0.3, procedural=0.2),
        ),
    ],
    "favor": [
        Justification(
            label="You can ask, but you can't expect",
            dimensions=DimensionalSignature(autonomy=0.7, fairness=0.3),
        ),
        Justification(
            label="Everyone's fighting their own battles",
            dimensions=DimensionalSignature(autonomy=0.5, epistemic=0.3, harm=0.2),
        ),
        Justification(
            label="Resentful help isn't really help",
            dimensions=DimensionalSignature(autonomy=0.4, harm=0.4, societal=0.2),
        ),
    ],
    "generic": [
        Justification(
            label="The world looks different from where they're standing",
            dimensions=DimensionalSignature(epistemic=0.6, autonomy=0.4),
        ),
        Justification(
            label="You can't force what has to be freely given",
            dimensions=DimensionalSignature(autonomy=0.8, societal=0.2),
        ),
        Justification(
            label="Maybe it's not as clear-cut as it feels from your side",
            dimensions=DimensionalSignature(epistemic=0.7, fairness=0.3),
        ),
    ],
}

CLAIM_JUSTIFICATIONS = {
    "promise": [
        Justification(
            label="They made you a promise. That's not nothing",
            dimensions=DimensionalSignature(legitimacy=0.7, rights=0.3),
        ),
        Justification(
            label="You held up your end—why shouldn't you expect the same?",
            dimensions=DimensionalSignature(fairness=0.7, legitimacy=0.3),
        ),
        Justification(
            label="This isn't about being demanding, it's about being reasonable",
            dimensions=DimensionalSignature(fairness=0.5, procedural=0.3, rights=0.2),
        ),
    ],
    "debt": [
        Justification(
            label="It's your money. Full stop",
            dimensions=DimensionalSignature(rights=0.7, fairness=0.3),
        ),
        Justification(
            label="You trusted them enough to help when they needed it",
            dimensions=DimensionalSignature(legitimacy=0.6, fairness=0.4),
        ),
    ],
    "secret": [
        Justification(
            label="You opened up to them—that takes something",
            dimensions=DimensionalSignature(legitimacy=0.5, privacy=0.5),
        ),
        Justification(
            label="You explicitly asked them to keep it quiet",
            dimensions=DimensionalSignature(procedural=0.5, legitimacy=0.3, privacy=0.2),
        ),
    ],
    "favor": [
        Justification(
            label="You've been there for them plenty of times",
            dimensions=DimensionalSignature(fairness=0.7, legitimacy=0.3),
        ),
        Justification(
            label="Isn't this what friends are for?",
            dimensions=DimensionalSignature(societal=0.6, legitimacy=0.4),
        ),
    ],
    "generic": [
        Justification(
            label="You're not asking for anything unreasonable",
            dimensions=DimensionalSignature(fairness=0.6, procedural=0.4),
        ),
        Justification(
            label="This was the deal, whether spoken or not",
            dimensions=DimensionalSignature(legitimacy=0.6, fairness=0.4),
        ),
    ],
}

NO_CLAIM_JUSTIFICATIONS = {
    "promise": [
        Justification(
            label="Maybe holding people to the letter of what they said isn't the point",
            dimensions=DimensionalSignature(epistemic=0.5, autonomy=0.3, societal=0.2),
        ),
        Justification(
            label="Being technically right and being right aren't the same thing",
            dimensions=DimensionalSignature(epistemic=0.6, fairness=0.4),
        ),
        Justification(
            label="Do you want to be the person who keeps score?",
            dimensions=DimensionalSignature(societal=0.5, autonomy=0.3, harm=0.2),
        ),
    ],
    "debt": [
        Justification(
            label="Did either of you really think of it that way at the time?",
            dimensions=DimensionalSignature(epistemic=0.7, procedural=0.3),
        ),
        Justification(
            label="Some things are worth more than being repaid",
            dimensions=DimensionalSignature(societal=0.6, autonomy=0.4),
        ),
    ],
    "secret": [
        Justification(
            label="You can't wrap yourself in 'confidentiality' when someone's getting hurt",
            dimensions=DimensionalSignature(harm=0.6, fairness=0.4),
        ),
        Justification(
            label="The truth has a way of being its own justification",
            dimensions=DimensionalSignature(epistemic=0.7, rights=0.3),
        ),
    ],
    "favor": [
        Justification(
            label="Generosity stops being generous when it becomes an invoice",
            dimensions=DimensionalSignature(autonomy=0.5, fairness=0.3, societal=0.2),
        ),
        Justification(
            label="You don't know what's going on in their life right now",
            dimensions=DimensionalSignature(epistemic=0.5, autonomy=0.5),
        ),
    ],
    "generic": [
        Justification(
            label="Entitlement is a bad look, even when you have a point",
            dimensions=DimensionalSignature(societal=0.5, autonomy=0.3, fairness=0.2),
        ),
        Justification(
            label="Maybe this is more complicated than who owes what to whom",
            dimensions=DimensionalSignature(epistemic=0.6, procedural=0.4),
        ),
    ],
}


def detect_scenario_type(letter: Letter) -> str:
    """Detect scenario type from letter content."""
    params = letter.protocol_params
    body_lower = letter.body.lower()

    if "series" in params:
        series = params["series"]
        if "promise" in series:
            return "promise"
        if "loan" in series or "debt" in series:
            return "debt"

    if "scenario" in params:
        scenario = params["scenario"]
        if scenario in ("lending", "debt"):
            return "debt"
        if scenario in ("secret", "confidentiality"):
            return "secret"
        if scenario == "favor":
            return "favor"

    if "promise" in body_lower or "agreed" in body_lower:
        return "promise"
    if "lent" in body_lower or "borrowed" in body_lower or "owe" in body_lower:
        return "debt"
    if "secret" in body_lower or "confidential" in body_lower:
        return "secret"
    if "favor" in body_lower:
        return "favor"

    return "generic"


def get_justifications_for_state(state: HohfeldianState, scenario: str) -> list[Justification]:
    """Get justification options for a given Hohfeldian state and scenario."""
    templates = {
        HohfeldianState.O: OBLIGATION_JUSTIFICATIONS,
        HohfeldianState.L: LIBERTY_JUSTIFICATIONS,
        HohfeldianState.C: CLAIM_JUSTIFICATIONS,
        HohfeldianState.N: NO_CLAIM_JUSTIFICATIONS,
    }
    scenario_justifications = templates[state]
    return scenario_justifications.get(scenario, scenario_justifications["generic"])


def get_justifications_for_letter(
    letter: Letter, _party_name: str, chosen_state: HohfeldianState
) -> list[Justification]:
    """
    Get justification options after player makes binary choice.

    Usage:
        1. Player chooses O or L for other party (or C or N for self)
        2. Call this to get "Why?" options
        3. Player picks justification
        4. Record both Hohfeldian state AND dimensional signature
    """
    scenario = detect_scenario_type(letter)
    return get_justifications_for_state(chosen_state, scenario)


class DimensionalVerdict(BaseModel):
    """Verdict with both Hohfeldian state and dimensional justification."""

    party_name: str
    state: HohfeldianState
    justification: str
    dimensions: DimensionalSignature
    expected: HohfeldianState | None = None

    @property
    def is_hohfeld_correct(self) -> bool | None:
        if self.expected is None:
            return None
        return self.state == self.expected
