"""
Core data models for Dear Ethicist.

This module defines:
- HohfeldianState: The four normative positions {O, C, L, N}
- D4Element: The 8 elements of the dihedral group D4
- D4 group operations
- Letter: An advice column letter from a reader
- Verdict: The player's classification (hidden measurement)
- Response: The player's published advice
- ReaderReaction: Feedback from "readers"
- GameState: Career progression tracking
"""

from datetime import datetime
from enum import Enum
from typing import Literal, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


# =============================================================================
# HOHFELDIAN CLASSIFICATION
# =============================================================================

class HohfeldianState(str, Enum):
    """
    The four Hohfeldian normative positions.

    Natural language mappings:
    - O: "Do I have to?" / "Am I obligated?"
    - C: "Am I entitled?" / "Do I have a right?"
    - L: "Can I refuse?" / "Am I free to choose?"
    - N: "Can they demand?" (no) / "They have no right"
    """
    O = "O"  # Obligation: MUST do something
    C = "C"  # Claim: OWED something / has a RIGHT
    L = "L"  # Liberty: FREE to choose
    N = "N"  # No-claim: CANNOT demand


class D4Element(str, Enum):
    """The 8 elements of the dihedral group D4."""
    E = "e"
    R = "r"
    R2 = "r2"
    R3 = "r3"
    S = "s"
    SR = "sr"
    SR2 = "sr2"
    SR3 = "sr3"


# =============================================================================
# D4 GROUP OPERATIONS
# =============================================================================

_D4_MULT_TABLE: dict[tuple[D4Element, D4Element], D4Element] = {
    (D4Element.E, D4Element.E): D4Element.E,
    (D4Element.E, D4Element.R): D4Element.R,
    (D4Element.E, D4Element.R2): D4Element.R2,
    (D4Element.E, D4Element.R3): D4Element.R3,
    (D4Element.E, D4Element.S): D4Element.S,
    (D4Element.E, D4Element.SR): D4Element.SR,
    (D4Element.E, D4Element.SR2): D4Element.SR2,
    (D4Element.E, D4Element.SR3): D4Element.SR3,
    (D4Element.R, D4Element.E): D4Element.R,
    (D4Element.R, D4Element.R): D4Element.R2,
    (D4Element.R, D4Element.R2): D4Element.R3,
    (D4Element.R, D4Element.R3): D4Element.E,
    (D4Element.R, D4Element.S): D4Element.SR3,
    (D4Element.R, D4Element.SR): D4Element.S,
    (D4Element.R, D4Element.SR2): D4Element.SR,
    (D4Element.R, D4Element.SR3): D4Element.SR2,
    (D4Element.R2, D4Element.E): D4Element.R2,
    (D4Element.R2, D4Element.R): D4Element.R3,
    (D4Element.R2, D4Element.R2): D4Element.E,
    (D4Element.R2, D4Element.R3): D4Element.R,
    (D4Element.R2, D4Element.S): D4Element.SR2,
    (D4Element.R2, D4Element.SR): D4Element.SR3,
    (D4Element.R2, D4Element.SR2): D4Element.S,
    (D4Element.R2, D4Element.SR3): D4Element.SR,
    (D4Element.R3, D4Element.E): D4Element.R3,
    (D4Element.R3, D4Element.R): D4Element.E,
    (D4Element.R3, D4Element.R2): D4Element.R,
    (D4Element.R3, D4Element.R3): D4Element.R2,
    (D4Element.R3, D4Element.S): D4Element.SR,
    (D4Element.R3, D4Element.SR): D4Element.SR2,
    (D4Element.R3, D4Element.SR2): D4Element.SR3,
    (D4Element.R3, D4Element.SR3): D4Element.S,
    (D4Element.S, D4Element.E): D4Element.S,
    (D4Element.S, D4Element.R): D4Element.SR,
    (D4Element.S, D4Element.R2): D4Element.SR2,
    (D4Element.S, D4Element.R3): D4Element.SR3,
    (D4Element.S, D4Element.S): D4Element.E,
    (D4Element.S, D4Element.SR): D4Element.R,
    (D4Element.S, D4Element.SR2): D4Element.R2,
    (D4Element.S, D4Element.SR3): D4Element.R3,
    (D4Element.SR, D4Element.E): D4Element.SR,
    (D4Element.SR, D4Element.R): D4Element.SR2,
    (D4Element.SR, D4Element.R2): D4Element.SR3,
    (D4Element.SR, D4Element.R3): D4Element.S,
    (D4Element.SR, D4Element.S): D4Element.R3,
    (D4Element.SR, D4Element.SR): D4Element.E,
    (D4Element.SR, D4Element.SR2): D4Element.R,
    (D4Element.SR, D4Element.SR3): D4Element.R2,
    (D4Element.SR2, D4Element.E): D4Element.SR2,
    (D4Element.SR2, D4Element.R): D4Element.SR3,
    (D4Element.SR2, D4Element.R2): D4Element.S,
    (D4Element.SR2, D4Element.R3): D4Element.SR,
    (D4Element.SR2, D4Element.S): D4Element.R2,
    (D4Element.SR2, D4Element.SR): D4Element.R3,
    (D4Element.SR2, D4Element.SR2): D4Element.E,
    (D4Element.SR2, D4Element.SR3): D4Element.R,
    (D4Element.SR3, D4Element.E): D4Element.SR3,
    (D4Element.SR3, D4Element.R): D4Element.S,
    (D4Element.SR3, D4Element.R2): D4Element.SR,
    (D4Element.SR3, D4Element.R3): D4Element.SR2,
    (D4Element.SR3, D4Element.S): D4Element.R,
    (D4Element.SR3, D4Element.SR): D4Element.R2,
    (D4Element.SR3, D4Element.SR2): D4Element.R3,
    (D4Element.SR3, D4Element.SR3): D4Element.E,
}

_D4_INVERSE: dict[D4Element, D4Element] = {
    D4Element.E: D4Element.E,
    D4Element.R: D4Element.R3,
    D4Element.R2: D4Element.R2,
    D4Element.R3: D4Element.R,
    D4Element.S: D4Element.S,      # All reflections are self-inverse (order 2)
    D4Element.SR: D4Element.SR,
    D4Element.SR2: D4Element.SR2,
    D4Element.SR3: D4Element.SR3,
}

_ROTATION: dict[HohfeldianState, HohfeldianState] = {
    HohfeldianState.O: HohfeldianState.C,
    HohfeldianState.C: HohfeldianState.L,
    HohfeldianState.L: HohfeldianState.N,
    HohfeldianState.N: HohfeldianState.O,
}

_REFLECTION: dict[HohfeldianState, HohfeldianState] = {
    HohfeldianState.O: HohfeldianState.C,
    HohfeldianState.C: HohfeldianState.O,
    HohfeldianState.L: HohfeldianState.N,
    HohfeldianState.N: HohfeldianState.L,
}


def d4_multiply(a: D4Element, b: D4Element) -> D4Element:
    """Compute a * b in the D4 group."""
    return _D4_MULT_TABLE[(a, b)]


def d4_inverse(a: D4Element) -> D4Element:
    """Compute the inverse of element a in D4."""
    return _D4_INVERSE[a]


def d4_apply_to_state(element: D4Element, state: HohfeldianState) -> HohfeldianState:
    """Apply a D4 group element to a Hohfeldian state."""
    result = state
    match element:
        case D4Element.E:
            pass
        case D4Element.R:
            result = _ROTATION[result]
        case D4Element.R2:
            result = _ROTATION[_ROTATION[result]]
        case D4Element.R3:
            result = _ROTATION[_ROTATION[_ROTATION[result]]]
        case D4Element.S:
            result = _REFLECTION[result]
        case D4Element.SR:
            result = _ROTATION[_REFLECTION[result]]
        case D4Element.SR2:
            result = _ROTATION[_ROTATION[_REFLECTION[result]]]
        case D4Element.SR3:
            result = _ROTATION[_ROTATION[_ROTATION[_REFLECTION[result]]]]
    return result


def correlative(state: HohfeldianState) -> HohfeldianState:
    """Get the correlative state (s-reflection): O<->C, L<->N."""
    return _REFLECTION[state]


# =============================================================================
# LETTER STRUCTURE
# =============================================================================

class Protocol(str, Enum):
    """SQND experimental protocols."""
    GATE_DETECTION = "GATE_DETECTION"      # Protocol 1: Semantic gates
    CORRELATIVE = "CORRELATIVE"            # Protocol 2: O<->C, L<->N
    PATH_DEPENDENCE = "PATH_DEPENDENCE"    # Protocol 3: Perspective order
    CONTEXT_SALIENCE = "CONTEXT_SALIENCE"  # Protocol 4: Editor pressure
    PHASE_TRANSITION = "PHASE_TRANSITION"  # Protocol 5: Ambiguity level


class LetterVoice(str, Enum):
    """Voice styles for letter writers."""
    OVERTHINKER = "overthinker"      # Agonizes over tiny details
    VALIDATOR = "validator"          # Wants confirmation they're right
    GENUINELY_STUCK = "genuinely_stuck"  # Real dilemma, needs help
    AGGRIEVED = "aggrieved"          # Clearly wronged, wants validation


class Party(BaseModel):
    """A party mentioned in a letter."""
    name: str
    role: str  # e.g., "letter_writer", "friend", "coworker"
    is_writer: bool = False
    expected_state: Optional[HohfeldianState] = None


class Letter(BaseModel):
    """An advice column letter from a reader."""
    letter_id: str
    day: int = 0
    protocol: Protocol
    protocol_params: dict = Field(default_factory=dict)

    # Letter content
    signoff: str  # e.g., "CONFUSED IN CLEVELAND"
    subject: str  # Brief subject line
    body: str     # Full letter text
    voice: LetterVoice = LetterVoice.GENUINELY_STUCK

    # Parties involved
    parties: list[Party] = Field(default_factory=list)

    # The key question(s) in the letter
    questions: list[str] = Field(default_factory=list)

    # For follow-ups
    is_followup: bool = False
    original_letter_id: Optional[str] = None

    def get_writer(self) -> Optional[Party]:
        """Get the letter writer party."""
        for party in self.parties:
            if party.is_writer:
                return party
        return None

    def get_other_party(self) -> Optional[Party]:
        """Get the main other party (not the writer)."""
        for party in self.parties:
            if not party.is_writer:
                return party
        return None


# =============================================================================
# RESPONSE AND VERDICT
# =============================================================================

class Verdict(BaseModel):
    """
    The player's Hohfeldian classification (the hidden measurement).

    This is captured separately from the response text.
    """
    party_name: str
    state: HohfeldianState
    expected: Optional[HohfeldianState] = None

    @property
    def is_correct(self) -> Optional[bool]:
        if self.expected is None:
            return None
        return self.state == self.expected


class ResponseChoice(BaseModel):
    """A choice in the guided response builder."""
    label: str
    implies_state: Optional[HohfeldianState] = None
    for_party: Optional[str] = None


class Response(BaseModel):
    """The player's published response to a letter."""
    letter_id: str
    response_text: str = ""
    verdicts: list[Verdict] = Field(default_factory=list)
    advice_choices: list[str] = Field(default_factory=list)  # Selected advice options
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    def get_verdict_for(self, party_name: str) -> Optional[Verdict]:
        """Get verdict for a specific party."""
        for v in self.verdicts:
            if v.party_name == party_name:
                return v
        return None


# =============================================================================
# READER REACTIONS
# =============================================================================

class ReaderReaction(BaseModel):
    """A reaction from a "reader" to published advice."""
    username: str
    text: str
    tone: Literal["supportive", "critical", "mixed", "humorous"]


class FollowUp(BaseModel):
    """A follow-up letter from the original writer."""
    original_letter_id: str
    outcome_text: str
    writer_satisfaction: Literal["grateful", "neutral", "disappointed"]


# =============================================================================
# GAME STATE
# =============================================================================

class CareerStage(str, Enum):
    """Career progression stages."""
    JUNIOR = "junior"           # Weeks 1-2: Learning
    STAFF = "staff"             # Weeks 3-4: Established
    SENIOR = "senior"           # Weeks 5-6: Respected
    SYNDICATED = "syndicated"   # Weeks 7-8: National


class EditorMood(str, Enum):
    """Editor's current stance (affects Protocol 4)."""
    SUPPORTIVE = "supportive"
    NEUTRAL = "neutral"
    DEMANDING = "demanding"


class GameState(BaseModel):
    """Complete game state."""
    session_id: str = Field(default_factory=lambda: str(uuid4())[:8])
    current_day: int = 1
    current_week: int = 1

    # Career
    career_stage: CareerStage = CareerStage.JUNIOR
    readership: int = 1000  # Subscriber count
    reputation: int = 50    # 0-100 scale (not shown to player)

    # Editor
    editor_mood: EditorMood = EditorMood.NEUTRAL
    editor_note: str = ""

    # Stats (for endings, not shown during play)
    letters_answered: int = 0
    total_verdicts: int = 0
    correct_verdicts: int = 0

    # Progression unlocks
    free_response_unlocked: bool = False


# =============================================================================
# TELEMETRY
# =============================================================================

class TrialRecord(BaseModel):
    """Complete telemetry record for a single letter response."""
    trial_id: UUID = Field(default_factory=uuid4)
    session_id: str
    letter_id: str
    day: int
    week: int
    protocol: Protocol
    protocol_params: dict = Field(default_factory=dict)

    # The measurement
    verdicts: list[Verdict] = Field(default_factory=list)

    # Timing
    time_reading_ms: int = 0
    time_responding_ms: int = 0

    # Context
    editor_mood: EditorMood = EditorMood.NEUTRAL
    career_stage: CareerStage = CareerStage.JUNIOR

    timestamp: datetime = Field(default_factory=datetime.utcnow)

    def to_json(self) -> dict:
        """Serialize to JSON-compatible dictionary."""
        data = self.model_dump()
        data["trial_id"] = str(self.trial_id)
        data["timestamp"] = self.timestamp.isoformat()
        return data
