"""
BIP Moral Structure Extraction - Language-Neutral Mathematical Approach

This module extracts structured moral bonds from text using LLMs,
enabling mathematical analysis of moral reasoning across cultures
without the confounding effects of language-specific embeddings.

Architecture:
    Text → LLM Extraction → Structured Bond → Canonical Form → Mathematical Analysis

The key insight is that moral structures (obligations, permissions, claims)
are language-independent concepts that can be represented symbolically.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import json


# =============================================================================
# PART 1: SCHEMA DEFINITIONS
# =============================================================================


class BondType(str, Enum):
    """Fundamental moral bond types based on Hohfeldian analysis."""

    OBLIGATION = "obligation"  # A must do X for B
    PROHIBITION = "prohibition"  # A must not do X to B
    PERMISSION = "permission"  # A may do X
    CLAIM = "claim"  # A has right that B do X
    LIBERTY = "liberty"  # A has no duty to do X
    POWER = "power"  # A can change B's normative status
    IMMUNITY = "immunity"  # A's status cannot be changed by B
    LIABILITY = "liability"  # A's status can be changed by B
    NO_BOND = "no_bond"  # No moral relationship detected


class RoleType(str, Enum):
    """Canonical role types that appear across cultures."""

    # Family roles
    PARENT = "parent"
    CHILD = "child"
    SPOUSE = "spouse"
    SIBLING = "sibling"
    ELDER = "elder"

    # Social roles
    RULER = "ruler"
    SUBJECT = "subject"
    TEACHER = "teacher"
    STUDENT = "student"
    FRIEND = "friend"

    # Economic roles
    CREDITOR = "creditor"
    DEBTOR = "debtor"
    EMPLOYER = "employer"
    WORKER = "worker"
    BUYER = "buyer"
    SELLER = "seller"

    # Moral roles
    PROMISER = "promiser"
    PROMISEE = "promisee"
    BENEFACTOR = "benefactor"
    BENEFICIARY = "beneficiary"
    WRONGDOER = "wrongdoer"
    VICTIM = "victim"

    # Generic
    AGENT = "agent"
    PATIENT = "patient"
    THIRD_PARTY = "third_party"
    COMMUNITY = "community"
    DEITY = "deity"


class ActionCategory(str, Enum):
    """Canonical action categories."""

    # Transfers
    GIVE = "give"
    RECEIVE = "receive"
    RETURN = "return"
    REPAY = "repay"

    # Care
    PROTECT = "protect"
    HELP = "help"
    NURTURE = "nurture"
    HEAL = "heal"

    # Harm
    HARM = "harm"
    KILL = "kill"
    STEAL = "steal"
    DECEIVE = "deceive"

    # Communication
    PROMISE = "promise"
    TELL_TRUTH = "tell_truth"
    KEEP_SECRET = "keep_secret"
    TEACH = "teach"

    # Respect
    HONOR = "honor"
    OBEY = "obey"
    WORSHIP = "worship"
    RESPECT = "respect"

    # Justice
    PUNISH = "punish"
    FORGIVE = "forgive"
    JUDGE = "judge"
    COMPENSATE = "compensate"

    # Generic
    ACT = "act"
    REFRAIN = "refrain"


class ContextType(str, Enum):
    """Moral context categories."""

    FAMILY = "family"
    ECONOMIC = "economic"
    POLITICAL = "political"
    RELIGIOUS = "religious"
    FRIENDSHIP = "friendship"
    PROFESSIONAL = "professional"
    LEGAL = "legal"
    WARFARE = "warfare"
    HOSPITALITY = "hospitality"
    RITUAL = "ritual"
    GENERIC = "generic"


class ModalStrength(str, Enum):
    """Deontic modal strength."""

    MUST = "must"  # Strong obligation
    SHOULD = "should"  # Weak obligation
    MAY = "may"  # Permission
    MAY_NOT = "may_not"  # Prohibition
    MUST_NOT = "must_not"  # Strong prohibition


@dataclass
class MoralBond:
    """
    A structured representation of a moral bond.

    This is the core data structure for language-neutral moral analysis.
    All fields use canonical enums to ensure cross-cultural comparability.
    """

    # Core structure
    bond_type: BondType
    agent_role: RoleType
    patient_role: RoleType
    action: ActionCategory

    # Modality
    modal_strength: ModalStrength = ModalStrength.MUST

    # Context
    context: ContextType = ContextType.GENERIC

    # Conditions (optional)
    condition: Optional[str] = None  # "if X promised"
    exception: Optional[str] = None  # "unless emergency"

    # Metadata
    source_language: Optional[str] = None
    source_tradition: Optional[str] = None
    source_text_id: Optional[str] = None
    confidence: float = 1.0

    def to_canonical_tuple(self) -> tuple:
        """Convert to canonical tuple for mathematical analysis."""
        return (
            self.bond_type.value,
            self.agent_role.value,
            self.patient_role.value,
            self.action.value,
            self.modal_strength.value,
        )

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "bond_type": self.bond_type.value,
            "agent_role": self.agent_role.value,
            "patient_role": self.patient_role.value,
            "action": self.action.value,
            "modal_strength": self.modal_strength.value,
            "context": self.context.value,
            "condition": self.condition,
            "exception": self.exception,
            "source_language": self.source_language,
            "source_tradition": self.source_tradition,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "MoralBond":
        """Create from dictionary."""
        return cls(
            bond_type=BondType(d["bond_type"]),
            agent_role=RoleType(d["agent_role"]),
            patient_role=RoleType(d["patient_role"]),
            action=ActionCategory(d["action"]),
            modal_strength=ModalStrength(d.get("modal_strength", "must")),
            context=ContextType(d.get("context", "generic")),
            condition=d.get("condition"),
            exception=d.get("exception"),
            source_language=d.get("source_language"),
            source_tradition=d.get("source_tradition"),
            confidence=d.get("confidence", 1.0),
        )


# =============================================================================
# PART 2: EXTRACTION PROMPTS
# =============================================================================

EXTRACTION_SYSTEM_PROMPT = """You are an expert in comparative ethics and moral philosophy.
Your task is to extract structured moral bonds from ancient and modern texts.

A moral bond is a normative relationship between agents, such as:
- Obligations: A must do X for B
- Prohibitions: A must not do X to B
- Permissions: A may do X
- Claims: A has a right that B do X

You must map all concepts to CANONICAL forms to enable cross-cultural comparison.
Use the exact enum values provided - do not invent new categories."""

EXTRACTION_USER_PROMPT = """Extract moral bonds from the following text.

TEXT:
{text}

SOURCE LANGUAGE: {language}
SOURCE TRADITION: {tradition}

For each moral bond found, provide a JSON object with these fields:
- bond_type: one of [obligation, prohibition, permission, claim, liberty, power, immunity, liability, no_bond]
- agent_role: the role of the person with the duty/right (use canonical roles)
- patient_role: the role of the person affected (use canonical roles)
- action: the action involved (use canonical action categories)
- modal_strength: one of [must, should, may, may_not, must_not]
- context: one of [family, economic, political, religious, friendship, professional, legal, warfare, hospitality, ritual, generic]
- condition: any conditions for the bond (string or null)
- exception: any exceptions (string or null)
- confidence: your confidence 0.0-1.0

CANONICAL ROLES:
- Family: parent, child, spouse, sibling, elder
- Social: ruler, subject, teacher, student, friend
- Economic: creditor, debtor, employer, worker, buyer, seller
- Moral: promiser, promisee, benefactor, beneficiary, wrongdoer, victim
- Generic: agent, patient, third_party, community, deity

CANONICAL ACTIONS:
- Transfers: give, receive, return, repay
- Care: protect, help, nurture, heal
- Harm: harm, kill, steal, deceive
- Communication: promise, tell_truth, keep_secret, teach
- Respect: honor, obey, worship, respect
- Justice: punish, forgive, judge, compensate
- Generic: act, refrain

Return a JSON array of moral bonds. If no bonds found, return [].

Example output:
[
  {{
    "bond_type": "obligation",
    "agent_role": "child",
    "patient_role": "parent",
    "action": "honor",
    "modal_strength": "must",
    "context": "family",
    "condition": null,
    "exception": null,
    "confidence": 0.95
  }}
]

Now extract bonds from the text:"""


def create_extraction_prompt(text: str, language: str, tradition: str) -> str:
    """Create the full extraction prompt for a given text."""
    return EXTRACTION_USER_PROMPT.format(
        text=text,
        language=language,
        tradition=tradition,
    )


# =============================================================================
# PART 3: EXTRACTION LOGIC
# =============================================================================


def parse_extraction_response(response: str, language: str, tradition: str) -> list[MoralBond]:
    """Parse LLM response into MoralBond objects."""
    # Try to find JSON array in response
    try:
        # Handle markdown code blocks
        if "```json" in response:
            start = response.find("```json") + 7
            end = response.find("```", start)
            response = response[start:end].strip()
        elif "```" in response:
            start = response.find("```") + 3
            end = response.find("```", start)
            response = response[start:end].strip()

        bonds_data = json.loads(response)
        if not isinstance(bonds_data, list):
            bonds_data = [bonds_data]

        bonds = []
        for bd in bonds_data:
            try:
                bd["source_language"] = language
                bd["source_tradition"] = tradition
                bond = MoralBond.from_dict(bd)
                bonds.append(bond)
            except (KeyError, ValueError) as e:
                print(f"Warning: Could not parse bond: {e}")
                continue

        return bonds
    except json.JSONDecodeError as e:
        print(f"Warning: Could not parse JSON response: {e}")
        return []


async def extract_bonds_async(
    client,  # anthropic.Anthropic client
    text: str,
    language: str,
    tradition: str,
    model: str = "claude-sonnet-4-20250514",
) -> list[MoralBond]:
    """Extract moral bonds from text using Claude API (async)."""
    message = await client.messages.create(
        model=model,
        max_tokens=2000,
        system=EXTRACTION_SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": create_extraction_prompt(text, language, tradition),
            }
        ],
    )

    response_text = message.content[0].text
    return parse_extraction_response(response_text, language, tradition)


def extract_bonds_sync(
    client,  # anthropic.Anthropic client
    text: str,
    language: str,
    tradition: str,
    model: str = "claude-sonnet-4-20250514",
) -> list[MoralBond]:
    """Extract moral bonds from text using Claude API (sync)."""
    message = client.messages.create(
        model=model,
        max_tokens=2000,
        system=EXTRACTION_SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": create_extraction_prompt(text, language, tradition),
            }
        ],
    )

    response_text = message.content[0].text
    return parse_extraction_response(response_text, language, tradition)


# =============================================================================
# PART 4: MATHEMATICAL ANALYSIS
# =============================================================================


@dataclass
class BondPattern:
    """A pattern discovered in moral bonds."""

    name: str
    description: str
    bonds: list[MoralBond]
    cultures_present: set[str]
    frequency: float  # 0-1, how often this pattern appears


def find_universal_patterns(bonds: list[MoralBond], min_cultures: int = 3) -> list[BondPattern]:
    """
    Find moral patterns that appear across multiple cultures.

    A universal pattern is a canonical bond tuple that appears in
    at least `min_cultures` different traditions.
    """
    from collections import defaultdict

    # Group by canonical tuple
    pattern_cultures: dict[tuple, set[str]] = defaultdict(set)
    pattern_bonds: dict[tuple, list[MoralBond]] = defaultdict(list)

    for bond in bonds:
        key = bond.to_canonical_tuple()
        if bond.source_tradition:
            pattern_cultures[key].add(bond.source_tradition)
        pattern_bonds[key].append(bond)

    # Find patterns present in multiple cultures
    universal = []
    total_bonds = len(bonds)

    for key, cultures in pattern_cultures.items():
        if len(cultures) >= min_cultures:
            bond_list = pattern_bonds[key]
            pattern = BondPattern(
                name=f"{key[0]}:{key[1]}->{key[2]}:{key[3]}",
                description=f"{key[1]} has {key[0]} to {key[3]} toward {key[2]}",
                bonds=bond_list,
                cultures_present=cultures,
                frequency=len(bond_list) / total_bonds if total_bonds > 0 else 0,
            )
            universal.append(pattern)

    # Sort by number of cultures, then frequency
    universal.sort(key=lambda p: (-len(p.cultures_present), -p.frequency))
    return universal


def compute_symmetry_group(bonds: list[MoralBond]) -> dict:
    """
    Analyze symmetry properties of moral bonds.

    Checks for:
    - Reciprocity: A→B implies B→A
    - Transitivity: A→B and B→C implies A→C
    - Reflexivity: A→A
    - Inversion: obligation ↔ prohibition
    """
    results = {
        "reciprocal_pairs": [],
        "transitive_chains": [],
        "reflexive_bonds": [],
        "inverse_pairs": [],
    }

    # Build lookup by (agent, patient, action)
    bond_lookup = {}
    for bond in bonds:
        key = (bond.agent_role, bond.patient_role, bond.action)
        bond_lookup[key] = bond

    # Check reciprocity: does (A,B,X) imply (B,A,X)?
    for bond in bonds:
        inverse_key = (bond.patient_role, bond.agent_role, bond.action)
        if inverse_key in bond_lookup:
            results["reciprocal_pairs"].append((bond, bond_lookup[inverse_key]))

    # Check reflexivity: (A,A,X)
    for bond in bonds:
        if bond.agent_role == bond.patient_role:
            results["reflexive_bonds"].append(bond)

    # Check for obligation/prohibition inversions
    for bond in bonds:
        if bond.bond_type == BondType.OBLIGATION:
            # Look for corresponding prohibition
            for other in bonds:
                if (
                    other.bond_type == BondType.PROHIBITION
                    and other.agent_role == bond.agent_role
                    and other.patient_role == bond.patient_role
                    and other.action == bond.action
                ):
                    results["inverse_pairs"].append((bond, other))

    return results


def compute_bond_algebra(bonds: list[MoralBond]) -> dict:
    """
    Compute algebraic properties of bond space.

    Returns statistics about the structure of moral bonds
    that might reveal mathematical patterns.
    """
    from collections import Counter

    stats = {
        "total_bonds": len(bonds),
        "bond_type_distribution": Counter(b.bond_type.value for b in bonds),
        "agent_role_distribution": Counter(b.agent_role.value for b in bonds),
        "patient_role_distribution": Counter(b.patient_role.value for b in bonds),
        "action_distribution": Counter(b.action.value for b in bonds),
        "context_distribution": Counter(b.context.value for b in bonds),
        "unique_canonical_tuples": len(set(b.to_canonical_tuple() for b in bonds)),
    }

    # Compute role transition matrix
    role_transitions = Counter()
    for bond in bonds:
        role_transitions[(bond.agent_role.value, bond.patient_role.value)] += 1
    stats["role_transitions"] = dict(role_transitions)

    # Compute action-bond associations
    action_bonds = Counter()
    for bond in bonds:
        action_bonds[(bond.action.value, bond.bond_type.value)] += 1
    stats["action_bond_associations"] = dict(action_bonds)

    return stats


def find_cultural_variations(
    bonds: list[MoralBond],
) -> dict[str, list[tuple[str, MoralBond]]]:
    """
    Find where cultures differ on the same moral situation.

    Returns cases where different cultures assign different
    bond types or modal strengths to the same agent/patient/action.
    """
    from collections import defaultdict

    # Group by (agent, patient, action)
    situation_bonds: dict[tuple, list[MoralBond]] = defaultdict(list)
    for bond in bonds:
        key = (bond.agent_role, bond.patient_role, bond.action)
        situation_bonds[key].append(bond)

    variations = {}
    for key, bond_list in situation_bonds.items():
        # Check if different cultures have different bond types
        culture_bonds = {}
        for bond in bond_list:
            culture = bond.source_tradition or "unknown"
            if culture not in culture_bonds:
                culture_bonds[culture] = bond

        # If more than one culture and they differ
        if len(culture_bonds) > 1:
            bond_types = set(b.bond_type for b in culture_bonds.values())
            if len(bond_types) > 1:
                variations[f"{key[0]}->{key[1]}:{key[2]}"] = list(culture_bonds.items())

    return variations


# =============================================================================
# PART 5: REPORTING
# =============================================================================


def generate_analysis_report(bonds: list[MoralBond]) -> str:
    """Generate a comprehensive analysis report."""
    lines = []
    lines.append("=" * 70)
    lines.append("MORAL STRUCTURE ANALYSIS REPORT")
    lines.append("=" * 70)
    lines.append("")

    # Basic stats
    stats = compute_bond_algebra(bonds)
    lines.append("## Overview")
    lines.append(f"Total bonds extracted: {stats['total_bonds']}")
    lines.append(f"Unique canonical patterns: {stats['unique_canonical_tuples']}")
    lines.append("")

    # Bond type distribution
    lines.append("## Bond Type Distribution")
    for bt, count in sorted(stats["bond_type_distribution"].items(), key=lambda x: -x[1]):
        pct = count / stats["total_bonds"] * 100 if stats["total_bonds"] > 0 else 0
        lines.append(f"  {bt:20s}: {count:5d} ({pct:5.1f}%)")
    lines.append("")

    # Universal patterns
    lines.append("## Universal Patterns (3+ cultures)")
    patterns = find_universal_patterns(bonds, min_cultures=3)
    if patterns:
        for p in patterns[:10]:  # Top 10
            lines.append(f"  {p.name}")
            lines.append(f"    Cultures: {', '.join(sorted(p.cultures_present))}")
            lines.append(f"    Frequency: {p.frequency:.1%}")
    else:
        lines.append("  No universal patterns found")
    lines.append("")

    # Symmetry analysis
    lines.append("## Symmetry Analysis")
    symmetry = compute_symmetry_group(bonds)
    lines.append(f"  Reciprocal pairs: {len(symmetry['reciprocal_pairs'])}")
    lines.append(f"  Reflexive bonds: {len(symmetry['reflexive_bonds'])}")
    lines.append(f"  Inverse pairs: {len(symmetry['inverse_pairs'])}")
    lines.append("")

    # Cultural variations
    lines.append("## Cultural Variations")
    variations = find_cultural_variations(bonds)
    if variations:
        for situation, culture_bonds in list(variations.items())[:5]:  # Top 5
            lines.append(f"  {situation}:")
            for culture, bond in culture_bonds:
                lines.append(f"    {culture}: {bond.bond_type.value} ({bond.modal_strength.value})")
    else:
        lines.append("  No significant variations found")
    lines.append("")

    lines.append("=" * 70)
    return "\n".join(lines)


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

EXAMPLE_TEXTS = {
    "hebrew_bible": {
        "text": "Honor your father and your mother, that your days may be long in the land.",
        "language": "hebrew",
        "tradition": "biblical",
    },
    "confucian": {
        "text": "The Master said: In serving your parents, you may gently remonstrate. "
        "If you see that they are not inclined to follow your advice, remain reverent "
        "and do not oppose them; work hard and do not murmur.",
        "language": "classical_chinese",
        "tradition": "confucian",
    },
    "dharma": {
        "text": "The son should serve his parents as the eastern direction: "
        "by rising to greet them, by serving them, by obeying them, "
        "by providing for their needs, and by honoring their traditions.",
        "language": "sanskrit",
        "tradition": "dharma",
    },
    "quran": {
        "text": "And your Lord has decreed that you worship none but Him, "
        "and that you be dutiful to your parents. If one or both of them "
        "attain old age in your care, never say to them 'uff' nor repel them, "
        "but address them with words of honor.",
        "language": "arabic",
        "tradition": "quranic",
    },
    "dear_abby": {
        "text": "My elderly mother expects me to call her every day and visit every weekend. "
        "I have my own family and job responsibilities. Am I obligated to meet all her demands?",
        "language": "english",
        "tradition": "modern_american",
    },
}


def demo_extraction():
    """Demonstrate the extraction system without API calls."""
    print("=" * 70)
    print("MORAL STRUCTURE EXTRACTION - DEMO")
    print("=" * 70)
    print()

    # Simulated extractions (what the LLM would return)
    demo_bonds = [
        MoralBond(
            bond_type=BondType.OBLIGATION,
            agent_role=RoleType.CHILD,
            patient_role=RoleType.PARENT,
            action=ActionCategory.HONOR,
            modal_strength=ModalStrength.MUST,
            context=ContextType.FAMILY,
            source_language="hebrew",
            source_tradition="biblical",
        ),
        MoralBond(
            bond_type=BondType.OBLIGATION,
            agent_role=RoleType.CHILD,
            patient_role=RoleType.PARENT,
            action=ActionCategory.OBEY,
            modal_strength=ModalStrength.SHOULD,
            context=ContextType.FAMILY,
            condition="may gently remonstrate first",
            source_language="classical_chinese",
            source_tradition="confucian",
        ),
        MoralBond(
            bond_type=BondType.OBLIGATION,
            agent_role=RoleType.CHILD,
            patient_role=RoleType.PARENT,
            action=ActionCategory.HONOR,
            modal_strength=ModalStrength.MUST,
            context=ContextType.FAMILY,
            source_language="sanskrit",
            source_tradition="dharma",
        ),
        MoralBond(
            bond_type=BondType.OBLIGATION,
            agent_role=RoleType.CHILD,
            patient_role=RoleType.PARENT,
            action=ActionCategory.HONOR,
            modal_strength=ModalStrength.MUST,
            context=ContextType.FAMILY,
            source_language="arabic",
            source_tradition="quranic",
        ),
        MoralBond(
            bond_type=BondType.OBLIGATION,
            agent_role=RoleType.CHILD,
            patient_role=RoleType.PARENT,
            action=ActionCategory.HELP,
            modal_strength=ModalStrength.SHOULD,
            context=ContextType.FAMILY,
            exception="within reasonable limits",
            source_language="english",
            source_tradition="modern_american",
        ),
    ]

    # Generate report
    report = generate_analysis_report(demo_bonds)
    print(report)

    return demo_bonds


if __name__ == "__main__":
    demo_extraction()
