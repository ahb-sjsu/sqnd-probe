#!/usr/bin/env python3
"""
Extract bond structures from passages.

Bond = (agent, patient, relation, consent)

This is the key step that enables BIP testing.
"""

import json
import re
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Set, Optional, Tuple
from enum import Enum
from collections import defaultdict
from tqdm import tqdm
import yaml

from .preprocess import (
    BondType, HohfeldianState, ConsentStatus,
    TimePeriod, Passage
)

# =============================================================================
# BOND DATA STRUCTURES
# =============================================================================

@dataclass
class Bond:
    """A single moral bond."""
    agent: str
    patient: str
    relation: str  # BondType name
    consent: str   # ConsentStatus name

    def to_tuple(self) -> tuple:
        return (self.agent, self.patient, self.relation, self.consent)

    def to_dict(self) -> dict:
        return asdict(self)

@dataclass
class BondStructure:
    """Complete bond structure of a passage."""
    bonds: List[dict]  # List of Bond.to_dict()
    primary_relation: str
    hohfeld_state: Optional[str] = None
    signature: str = ""  # Canonical representation for isomorphism

    def to_dict(self) -> dict:
        return asdict(self)

# =============================================================================
# PATTERN-BASED BOND EXTRACTOR
# =============================================================================

class BondExtractor:
    """
    Extract bond structures using pattern matching.

    This is the rule-based version. For production, combine with
    LLM-based extraction for ambiguous cases.
    """

    # Agent/patient role patterns (abstract roles, not specific names)
    ROLE_PATTERNS = {
        'SELF': [
            r'\b(I|me|my|myself|mine)\b',
            r'\bone\'s own\b',
        ],
        'OTHER': [
            r'\b(you|your|yours|yourself)\b',
            r'\b(they|them|their|theirs)\b',
            r'\b(he|him|his|she|her|hers)\b',
            r'\b(another|other|someone|anyone|person|people|individual)\b',
            r'\b(neighbor|neighbour|fellow)\b',
        ],
        'FAMILY': [
            r'\b(mother|father|parent|parents)\b',
            r'\b(son|daughter|child|children|kid|kids)\b',
            r'\b(brother|sister|sibling|siblings)\b',
            r'\b(husband|wife|spouse)\b',
            r'\b(grandfather|grandmother|grandparent)\b',
            r'\b(uncle|aunt|cousin|nephew|niece)\b',
            r'\b(family|relative|kin)\b',
        ],
        'AUTHORITY': [
            r'\b(rabbi|sage|teacher|master|rav)\b',
            r'\b(court|judge|beit din|sanhedrin)\b',
            r'\b(king|ruler|government|authority)\b',
            r'\b(God|Hashem|Lord|divine)\b',
            r'\b(boss|employer|supervisor)\b',
        ],
        'COMMUNITY': [
            r'\b(community|society|public|nation)\b',
            r'\b(Israel|people|congregation)\b',
            r'\b(group|organization|collective)\b',
        ],
        'VULNERABLE': [
            r'\b(orphan|widow|poor|needy)\b',
            r'\b(stranger|convert|ger)\b',
            r'\b(sick|ill|injured|dying)\b',
            r'\b(child|infant|minor)\b',
        ],
    }

    # Relation patterns
    RELATION_PATTERNS = {
        BondType.HARM_PREVENTION: [
            r'\b(kill|murder|slay|put to death)\b',
            r'\b(harm|hurt|injure|damage|wound)\b',
            r'\b(save|rescue|protect|preserve)\b',
            r'\b(life|death|danger|peril|risk)\b',
            r'\b(blood|violence|strike|smite)\b',
        ],
        BondType.RECIPROCITY: [
            r'\b(return|repay|restore|give back)\b',
            r'\b(owe|debt|obligation|due)\b',
            r'\b(equal|same|likewise|similarly)\b',
            r'\b(mutual|reciprocal|exchange)\b',
            r'\b(hateful|love|do unto)\b',
        ],
        BondType.AUTONOMY: [
            r'\b(choose|decision|free will|liberty)\b',
            r'\b(consent|agree|accept|refuse)\b',
            r'\b(force|coerce|compel|require)\b',
            r'\b(right|entitle|privilege)\b',
            r'\b(self-determination|agency)\b',
        ],
        BondType.PROPERTY: [
            r'\b(property|possession|belonging)\b',
            r'\b(own|ownership|owner)\b',
            r'\b(steal|theft|rob|take)\b',
            r'\b(buy|sell|trade|exchange)\b',
            r'\b(land|field|house|goods)\b',
            r'\b(lost|found|return)\b',
        ],
        BondType.FAMILY: [
            r'\b(honor|respect|revere)\b.*\b(parent|father|mother)\b',
            r'\b(marry|marriage|betroth|wed)\b',
            r'\b(divorce|separate|put away)\b',
            r'\b(inherit|inheritance|heir)\b',
            r'\b(support|provide|sustain)\b.*\b(family|child|wife)\b',
        ],
        BondType.AUTHORITY: [
            r'\b(obey|follow|heed|submit)\b',
            r'\b(command|decree|order|law)\b',
            r'\b(permit|allow|forbid|prohibit)\b',
            r'\b(judge|rule|decide|arbitrate)\b',
            r'\b(teach|instruct|guide)\b',
        ],
        BondType.EMERGENCY: [
            r'\b(emergency|urgent|immediate)\b',
            r'\b(life-threatening|mortal danger)\b',
            r'\b(pikuach nefesh|save a life)\b',
            r'\b(override|supersede|set aside)\b',
            r'\b(Sabbath|shabbat|shabbos)\b.*\b(save|life)\b',
        ],
        BondType.CONTRACT: [
            r'\b(promise|vow|oath|swear)\b',
            r'\b(agree|agreement|contract)\b',
            r'\b(commit|commitment|pledge)\b',
            r'\b(word|keep|break|fulfill)\b',
            r'\b(neder|shevua)\b',
        ],
        BondType.CARE: [
            r'\b(care|caring|tend|look after)\b',
            r'\b(help|assist|aid|support)\b',
            r'\b(feed|clothe|shelter|provide)\b',
            r'\b(visit|comfort|console)\b',
            r'\b(tzedakah|charity|give)\b',
        ],
        BondType.FAIRNESS: [
            r'\b(fair|just|justice|righteous)\b',
            r'\b(equal|equally|same|impartial)\b',
            r'\b(deserve|merit|earn|worthy)\b',
            r'\b(bias|favor|prejudice|partial)\b',
            r'\b(mishpat|din|tzedek)\b',
        ],
    }

    # Hohfeldian state patterns
    HOHFELD_PATTERNS = {
        HohfeldianState.OBLIGATION: [
            r'\b(must|shall|have to|need to)\b',
            r'\b(obligat|duty|bound|require)\b',
            r'\b(command|mitzvah|chayav)\b',
            r'\b(responsible|accountable)\b',
        ],
        HohfeldianState.RIGHT: [
            r'\b(right to|entitled|may claim)\b',
            r'\b(deserve|owed|due)\b',
            r'\b(demand|expect|require from)\b',
        ],
        HohfeldianState.LIBERTY: [
            r'\b(may|can|permitted|allowed)\b',
            r'\b(free to|liberty to|privilege)\b',
            r'\b(optional|voluntary|choice)\b',
            r'\b(mutar|reshut)\b',
        ],
        HohfeldianState.NO_RIGHT: [
            r'\b(cannot demand|no claim|not entitled)\b',
            r'\b(no right to|may not require)\b',
        ],
    }

    # Consent patterns
    CONSENT_PATTERNS = {
        ConsentStatus.EXPLICIT_YES: [
            r'\b(agree|agreed|consent|consented)\b',
            r'\b(accept|accepted|willing|willingly)\b',
            r'\b(yes|approve|approved)\b',
        ],
        ConsentStatus.EXPLICIT_NO: [
            r'\b(refuse|refused|reject|rejected)\b',
            r'\b(no|deny|denied|unwilling)\b',
            r'\b(against|oppose|opposed)\b',
        ],
        ConsentStatus.CONTESTED: [
            r'\b(dispute|disputed|disagree)\b',
            r'\b(debate|debated|argue|argued)\b',
            r'\b(machlok|machloket|controversy)\b',
            r'\b(some say|others say|one says)\b',
            r'\b(Rabbi \w+ says.*Rabbi \w+ says)\b',
        ],
        ConsentStatus.IMPOSSIBLE: [
            r'\b(future generation|unborn)\b',
            r'\b(animal|beast|creature)\b',
            r'\b(deceased|dead|departed)\b',
            r'\b(cannot consent|unable to agree)\b',
        ],
    }

    def __init__(self, config: dict = None):
        self.config = config or {}

        # Compile patterns
        self.role_compiled = {
            role: [re.compile(p, re.IGNORECASE) for p in patterns]
            for role, patterns in self.ROLE_PATTERNS.items()
        }
        self.relation_compiled = {
            rel: [re.compile(p, re.IGNORECASE) for p in patterns]
            for rel, patterns in self.RELATION_PATTERNS.items()
        }
        self.hohfeld_compiled = {
            state: [re.compile(p, re.IGNORECASE) for p in patterns]
            for state, patterns in self.HOHFELD_PATTERNS.items()
        }
        self.consent_compiled = {
            status: [re.compile(p, re.IGNORECASE) for p in patterns]
            for status, patterns in self.CONSENT_PATTERNS.items()
        }

    def extract(self, passage: Passage) -> BondStructure:
        """Extract bond structure from passage."""
        text = passage.text_english

        # Extract components
        roles = self._extract_roles(text)
        relations = self._extract_relations(text)
        hohfeld = self._extract_hohfeld(text)
        consent = self._extract_consent(text)

        # Build bonds
        bonds = self._build_bonds(roles, relations, consent)

        # Determine primary relation
        primary = relations[0].name if relations else BondType.CARE.name

        # Compute signature for isomorphism checking
        signature = self._compute_signature(bonds)

        return BondStructure(
            bonds=[b.to_dict() for b in bonds],
            primary_relation=primary,
            hohfeld_state=hohfeld.name if hohfeld else None,
            signature=signature
        )

    def _extract_roles(self, text: str) -> List[str]:
        """Extract moral roles from text."""
        roles = []
        for role, patterns in self.role_compiled.items():
            for pattern in patterns:
                if pattern.search(text):
                    roles.append(role)
                    break

        # Default if none found
        if not roles:
            roles = ['SELF', 'OTHER']

        return list(set(roles))

    def _extract_relations(self, text: str) -> List[BondType]:
        """Extract relation types from text."""
        relations = []
        relation_scores = {}

        for rel_type, patterns in self.relation_compiled.items():
            count = 0
            for pattern in patterns:
                count += len(pattern.findall(text))
            if count > 0:
                relation_scores[rel_type] = count

        # Sort by frequency
        relations = sorted(relation_scores.keys(),
                          key=lambda r: relation_scores[r],
                          reverse=True)

        if not relations:
            relations = [BondType.CARE]

        return relations

    def _extract_hohfeld(self, text: str) -> Optional[HohfeldianState]:
        """Extract Hohfeldian state."""
        state_scores = {}

        for state, patterns in self.hohfeld_compiled.items():
            count = 0
            for pattern in patterns:
                count += len(pattern.findall(text))
            if count > 0:
                state_scores[state] = count

        if state_scores:
            return max(state_scores.keys(), key=lambda s: state_scores[s])
        return None

    def _extract_consent(self, text: str) -> ConsentStatus:
        """Extract consent status."""
        for status, patterns in self.consent_compiled.items():
            for pattern in patterns:
                if pattern.search(text):
                    return status
        return ConsentStatus.IMPLICIT_YES

    def _build_bonds(
        self,
        roles: List[str],
        relations: List[BondType],
        consent: ConsentStatus
    ) -> List[Bond]:
        """Build bonds from extracted components."""
        bonds = []

        # Limit to top 3 relations to avoid explosion
        for rel in relations[:3]:
            for agent in roles:
                for patient in roles:
                    # Skip certain reflexive bonds
                    if agent == patient and rel not in [BondType.AUTONOMY, BondType.CARE]:
                        continue

                    bonds.append(Bond(
                        agent=agent,
                        patient=patient,
                        relation=rel.name,
                        consent=consent.name
                    ))

        return bonds

    def _compute_signature(self, bonds: List[Bond]) -> str:
        """
        Compute a canonical signature for bond structure.

        This enables isomorphism checking: two passages with
        the same signature have isomorphic bond structures.
        """
        if not bonds:
            return "empty"

        # Canonicalize roles by first appearance
        role_map = {}
        role_idx = 0

        canonical = []
        for bond in sorted(bonds, key=lambda b: (b.relation, b.agent, b.patient)):
            if bond.agent not in role_map:
                role_map[bond.agent] = f"R{role_idx}"
                role_idx += 1
            if bond.patient not in role_map:
                role_map[bond.patient] = f"R{role_idx}"
                role_idx += 1

            canonical.append((
                role_map[bond.agent],
                role_map[bond.patient],
                bond.relation,
                bond.consent
            ))

        # Sort for consistency
        canonical.sort()

        # Create signature string
        sig_parts = [f"{a}-{p}:{r}/{c}" for a, p, r, c in canonical]
        return "|".join(sig_parts)

# =============================================================================
# CONSENSUS CLASSIFIER
# =============================================================================

class ConsensusClassifier:
    """Classify passages by consensus tier."""

    DISPUTE_PATTERNS = [
        re.compile(r'\bdispute[ds]?\b', re.IGNORECASE),
        re.compile(r'\bdisagree[ds]?\b', re.IGNORECASE),
        re.compile(r'\bmachlok', re.IGNORECASE),
        re.compile(r'\bdebate[ds]?\b', re.IGNORECASE),
        re.compile(r'Rabbi \w+ says', re.IGNORECASE),
        re.compile(r'\bsome say\b', re.IGNORECASE),
        re.compile(r'\bothers say\b', re.IGNORECASE),
        re.compile(r'\bone opinion\b', re.IGNORECASE),
        re.compile(r'\banother view\b', re.IGNORECASE),
    ]

    UNIVERSAL_PATTERNS = [
        re.compile(r'\beveryone agrees\b', re.IGNORECASE),
        re.compile(r'\bundisputed\b', re.IGNORECASE),
        re.compile(r'\buniversally\b', re.IGNORECASE),
        re.compile(r'\balways\b', re.IGNORECASE),
        re.compile(r'\bnever\b.*\b(kill|murder|steal)\b', re.IGNORECASE),
        re.compile(r'save[s]? a life', re.IGNORECASE),
        re.compile(r'pikuach nefesh', re.IGNORECASE),
        re.compile(r'love your neighbor', re.IGNORECASE),
        re.compile(r'golden rule', re.IGNORECASE),
    ]

    def classify(self, text: str) -> Tuple[str, bool]:
        """
        Classify consensus tier.

        Returns: (tier, has_dispute)
        """
        has_dispute = any(p.search(text) for p in self.DISPUTE_PATTERNS)
        has_universal = any(p.search(text) for p in self.UNIVERSAL_PATTERNS)

        if has_dispute and not has_universal:
            return "contested", True
        elif has_universal:
            return "universal", False
        else:
            return "high", False

# =============================================================================
# MAIN EXTRACTION PIPELINE
# =============================================================================

def extract_bonds_all(config_path: str = "config_bip.yaml"):
    """Run bond extraction on all passages."""

    # Load config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Load passages
    processed_path = Path(config['data']['processed_path'])
    passages_file = processed_path / "passages.jsonl"

    print(f"Loading passages from {passages_file}...")
    passages = []
    with open(passages_file, encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            passages.append(Passage(**data))

    print(f"Loaded {len(passages)} passages")

    # Initialize extractors
    bond_extractor = BondExtractor(config.get('bond_extraction', {}))
    consensus_classifier = ConsensusClassifier()

    # Extract bonds
    print("Extracting bond structures...")
    bond_structures = []

    for passage in tqdm(passages):
        # Extract bond structure
        bond_struct = bond_extractor.extract(passage)

        # Classify consensus
        tier, has_dispute = consensus_classifier.classify(passage.text_english)

        # Update passage
        passage.bond_types = list(set(b['relation'] for b in bond_struct.bonds))
        passage.consensus_tier = tier
        passage.has_dispute = has_dispute

        bond_structures.append({
            'passage_id': passage.id,
            'bond_structure': bond_struct.to_dict()
        })

    # Save updated passages
    print("Saving updated passages...")
    with open(passages_file, 'w', encoding='utf-8') as f:
        for p in passages:
            f.write(json.dumps(p.to_dict()) + '\n')

    # Save bond structures separately
    bonds_file = processed_path / "bond_structures.jsonl"
    with open(bonds_file, 'w', encoding='utf-8') as f:
        for bs in bond_structures:
            f.write(json.dumps(bs) + '\n')

    # Print statistics
    print("\n" + "=" * 60)
    print("BOND EXTRACTION STATISTICS")
    print("=" * 60)

    # Count by relation type
    relation_counts = defaultdict(int)
    for bs in bond_structures:
        for bond in bs['bond_structure']['bonds']:
            relation_counts[bond['relation']] += 1

    print("\nBonds by relation type:")
    for rel, count in sorted(relation_counts.items(), key=lambda x: -x[1]):
        print(f"  {rel}: {count:,}")

    # Count by consensus
    consensus_counts = defaultdict(int)
    for p in passages:
        consensus_counts[p.consensus_tier] += 1

    print("\nPassages by consensus tier:")
    for tier, count in sorted(consensus_counts.items()):
        print(f"  {tier}: {count:,}")

    # Count unique signatures (for isomorphism)
    signatures = set(bs['bond_structure']['signature'] for bs in bond_structures)
    print(f"\nUnique bond signatures: {len(signatures):,}")

    print(f"\nBond structures saved to {bonds_file}")

    return passages, bond_structures

if __name__ == "__main__":
    extract_bonds_all()
