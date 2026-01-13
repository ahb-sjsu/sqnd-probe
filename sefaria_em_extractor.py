#!/usr/bin/env python3
"""
Sefaria EM-DAG Extractor
========================

Extracts Ethical Modules (EM-DAG) from Sefaria-Export repository.
Maps 2,000 years of Hebrew normative reasoning onto the D₄ gauge structure.

Based on NA-SQND v4.1 framework:
- Hohfeldian states: O (Obligation), L (Liberty), C (Claim), N (No-claim)
- D₄ gauge group acting on Hohfeldian square
- Semantic gates as group elements

Usage:
    python sefaria_em_extractor.py /path/to/Sefaria-Export --output hebrew_em_dag.json

Author: Based on Bond (2026) Dear Abby EM-DAG methodology
"""

import json
import re
import os
import argparse
from pathlib import Path
from collections import defaultdict, Counter
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple, Set
from enum import Enum
import hashlib


# =============================================================================
# HOHFELDIAN FRAMEWORK
# =============================================================================

class HohfeldianState(Enum):
    """The four Hohfeldian normative positions."""
    O = "Obligation"   # Duty to act
    L = "Liberty"      # Permission to act or not
    C = "Claim"        # Right against another
    N = "No-claim"     # No right against another


class D4Element(Enum):
    """
    D₄ group elements acting on Hohfeldian square.
    D₄ = ⟨r, s | r⁴ = s² = e, srs = r⁻¹⟩
    
    r (rotation): O → C → L → N → O
    s (reflection): O ↔ C, L ↔ N (correlative exchange)
    """
    E = "e"           # Identity
    R = "r"           # 90° rotation (O→C→L→N)
    R2 = "r²"         # 180° rotation (negation: O↔L, C↔N)
    R3 = "r³"         # 270° rotation
    S = "s"           # Reflection (correlative: O↔C, L↔N)
    SR = "sr"         # Reflection then rotate
    SR2 = "sr²"       # Reflection then 180°
    SR3 = "sr³"       # Reflection then 270°


@dataclass
class SemanticGate:
    """A semantic trigger that implements a D₄ group element."""
    trigger_pattern: str
    group_element: D4Element
    transition: Tuple[HohfeldianState, HohfeldianState]
    confidence: float
    language: str  # 'he' or 'en'
    examples: List[str] = field(default_factory=list)


# =============================================================================
# HEBREW/ENGLISH GATE PATTERNS
# =============================================================================

# Hebrew binding triggers (L → O)
HEBREW_BINDING_TRIGGERS = {
    # Vows and oaths
    r'נדר': ('neder/vow', 0.95),
    r'שבועה': ('shevuah/oath', 0.95),
    r'נשבע': ('swore', 0.90),
    r'קבל עליו': ('accepted upon himself', 0.90),
    r'התחייב': ('obligated himself', 0.92),
    r'חייב': ('obligated/must', 0.85),
    r'מצווה': ('commanded/mitzvah', 0.88),
    r'צריך': ('needs to/must', 0.75),
    r'אסור': ('forbidden', 0.90),  # Negative obligation
    r'מוכרח': ('compelled', 0.85),
    r'קיבל': ('received/accepted', 0.70),
    # Contractual
    r'קנין': ('kinyan/acquisition', 0.92),
    r'שטר': ('document/contract', 0.85),
    r'כתובה': ('ketubah', 0.95),
    r'גט': ('divorce document', 0.90),
}

# Hebrew release triggers (O → L)
HEBREW_RELEASE_TRIGGERS = {
    r'מותר': ('permitted', 0.90),
    r'פטור': ('exempt', 0.92),
    r'התרה': ('annulment', 0.95),
    r'מחילה': ('forgiveness/waiver', 0.90),
    r'הפקר': ('ownerless/renounced', 0.88),
    r'ביטול': ('nullification', 0.90),
    r'אונס': ('coercion/duress', 0.95),  # Nullifier
    r'שוגג': ('unintentional', 0.85),
    r'מוכרח': ('forced', 0.80),
    r'רשות': ('permission', 0.85),
}

# Hebrew correlative triggers (perspective shift, s element)
HEBREW_CORRELATIVE_TRIGGERS = {
    r'כנגד': ('corresponding to', 0.85),
    r'חובת': ('duty of', 0.80),
    r'זכות': ('right of', 0.85),
    r'כלפי': ('toward/against', 0.75),
    r'על ידי': ('by means of', 0.70),
}

# English equivalents (from translations)
ENGLISH_BINDING_TRIGGERS = {
    r'\b(must|shall|obligat|requir|command|bound)\b': ('obligation_marker', 0.85),
    r'\b(vow|oath|swear|swore|sworn|promise)\b': ('vow_marker', 0.92),
    r'\b(forbidden|prohibit|may not|cannot)\b': ('prohibition_marker', 0.88),
    r'\b(duty|duties|responsible|liability)\b': ('duty_marker', 0.82),
}

ENGLISH_RELEASE_TRIGGERS = {
    r'\b(permitted|allow|may|exempt|free|release)\b': ('permission_marker', 0.85),
    r'\b(annul|void|nullif|cancel)\b': ('annulment_marker', 0.90),
    r'\b(forgive|waive|relinquish)\b': ('forgiveness_marker', 0.88),
    r'\b(coerce|force|duress|compel)\b': ('coercion_nullifier', 0.92),
}

# Universal nullifiers (from Dear Abby: abuse nullifies all obligations)
NULLIFIER_PATTERNS = {
    r'\b(abuse|abus|violent|violence|harm|danger)\b': ('abuse_nullifier', 0.95),
    r'\b(pikuach nefesh|life.{0,10}danger|save.{0,5}life)\b': ('life_danger', 0.98),
    r'פיקוח נפש': ('pikuach_nefesh', 0.99),
    r'סכנת נפשות': ('mortal_danger', 0.98),
}


# =============================================================================
# SEFARIA CATEGORIES → ETHICAL DOMAINS
# =============================================================================

CATEGORY_TO_DOMAIN = {
    # Talmud tractates by domain
    'Seder Nezikin': 'civil_damages',      # Bava Kamma, Bava Metzia, Bava Batra, Sanhedrin
    'Seder Nashim': 'family_law',          # Kiddushin, Gittin, Ketubot, Sotah
    'Seder Kodashim': 'sacred_obligations', # Temple, sacrifices
    'Seder Zeraim': 'agricultural_law',    # Tithes, agricultural obligations
    'Seder Moed': 'temporal_obligations',  # Shabbat, holidays
    'Seder Tohorot': 'purity_law',         # Ritual purity
    
    # Specific tractates
    'Pirkei Avot': 'ethics',
    'Nedarim': 'vows',
    'Shevuot': 'oaths',
    'Bava Kamma': 'damages',
    'Bava Metzia': 'property',
    'Bava Batra': 'commerce',
    'Sanhedrin': 'criminal_law',
    'Kiddushin': 'marriage',
    'Gittin': 'divorce',
    'Ketubot': 'marriage_contracts',
    
    # Other categories
    'Midrash': 'homiletics',
    'Halakhah': 'law',
    'Musar': 'ethics',
    'Teshuvot': 'responsa',
    'Philosophy': 'philosophy',
}

# Time periods for temporal analysis
PERIOD_MARKERS = {
    'Tanakh': (-1000, -200),
    'Mishnah': (-200, 220),
    'Talmud': (220, 500),
    'Geonic': (500, 1000),
    'Rishonim': (1000, 1500),
    'Acharonim': (1500, 1800),
    'Modern': (1800, 2025),
}


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class Passage:
    """A single text passage with normative content."""
    ref: str                          # Sefaria reference (e.g., "Bava Metzia.62a.1")
    hebrew: str                       # Hebrew text
    english: str                      # English translation
    source: str                       # Source text name
    category: str                     # Sefaria category
    domain: str                       # Ethical domain
    
    # Hohfeldian analysis
    primary_state: Optional[HohfeldianState] = None
    state_confidence: float = 0.0
    detected_gates: List[Dict] = field(default_factory=list)
    
    # Consensus tier (from Dear Abby methodology)
    # UNIVERSAL (>95%), HIGH (80-95%), MODERATE (60-80%), CONTESTED (<60%)
    consensus_tier: str = "UNKNOWN"
    
    # Is this a dispute (machlokhet)?
    is_disputed: bool = False
    dispute_parties: List[str] = field(default_factory=list)


@dataclass 
class EthicalModule:
    """A domain-specific ethical module in the EM-DAG."""
    name: str
    domain: str
    
    # Empirical basis
    n_passages: int = 0
    sources: Set[str] = field(default_factory=set)
    time_range: Tuple[int, int] = (0, 0)
    
    # Default state distribution (like Dear Abby domain stats)
    state_distribution: Dict[str, float] = field(default_factory=dict)
    
    # Semantic gates
    binding_gates: List[Dict] = field(default_factory=list)   # L → O triggers
    release_gates: List[Dict] = field(default_factory=list)   # O → L triggers
    correlative_gates: List[Dict] = field(default_factory=list)  # Perspective shifts
    
    # Nullifiers (override all obligations)
    nullifiers: List[Dict] = field(default_factory=list)
    
    # Sub-modules
    children: List[str] = field(default_factory=list)
    
    # Key passages
    exemplar_passages: List[Dict] = field(default_factory=list)


@dataclass
class EMDAG:
    """The complete Ethical Module Directed Acyclic Graph."""
    meta: Dict = field(default_factory=dict)
    modules: Dict[str, EthicalModule] = field(default_factory=dict)
    
    # Structural constraints (from D₄ gauge theory)
    structural_constraints: Dict = field(default_factory=dict)
    
    # Cross-temporal stability analysis
    temporal_stability: Dict = field(default_factory=dict)
    
    # Gate statistics
    gate_statistics: Dict = field(default_factory=dict)


# =============================================================================
# PARSER
# =============================================================================

class SefariaEMExtractor:
    """Extracts EM-DAG from Sefaria-Export repository."""
    
    def __init__(self, sefaria_path: str):
        self.sefaria_path = Path(sefaria_path)
        self.json_path = self.sefaria_path / "json"
        
        if not self.json_path.exists():
            raise ValueError(f"JSON directory not found: {self.json_path}")
        
        self.passages: List[Passage] = []
        self.modules: Dict[str, EthicalModule] = {}
        self.gate_counts = defaultdict(Counter)
        
    def extract(self, 
                categories: Optional[List[str]] = None,
                max_passages: int = None,
                verbose: bool = True) -> EMDAG:
        """
        Main extraction pipeline.
        
        Args:
            categories: List of categories to process (None = all)
            max_passages: Maximum passages to process (None = all)
            verbose: Print progress
            
        Returns:
            EMDAG object
        """
        if verbose:
            print("=" * 60)
            print("Sefaria EM-DAG Extraction")
            print("=" * 60)
        
        # 1. Discover and parse JSON files
        if verbose:
            print("\n[1/5] Discovering texts...")
        json_files = self._discover_json_files(categories)
        if verbose:
            print(f"      Found {len(json_files)} text files")
        
        # 2. Parse passages
        if verbose:
            print("\n[2/5] Parsing passages...")
        self._parse_all_passages(json_files, max_passages, verbose)
        if verbose:
            print(f"      Parsed {len(self.passages)} passages")
        
        # 3. Detect Hohfeldian states and gates
        if verbose:
            print("\n[3/5] Detecting Hohfeldian states and semantic gates...")
        self._detect_all_gates(verbose)
        
        # 4. Build domain modules
        if verbose:
            print("\n[4/5] Building domain modules...")
        self._build_modules()
        
        # 5. Construct EM-DAG
        if verbose:
            print("\n[5/5] Constructing EM-DAG...")
        em_dag = self._construct_dag()
        
        if verbose:
            print("\n" + "=" * 60)
            print("Extraction complete!")
            print(f"  Modules: {len(em_dag.modules)}")
            print(f"  Passages: {len(self.passages)}")
            print("=" * 60)
        
        return em_dag
    
    def _discover_json_files(self, categories: Optional[List[str]] = None) -> List[Path]:
        """Find all JSON text files in the repository."""
        json_files = []
        
        for root, dirs, files in os.walk(self.json_path):
            # Skip non-text directories
            if 'merged.json' not in files:
                continue
                
            rel_path = Path(root).relative_to(self.json_path)
            parts = rel_path.parts
            
            # Filter by category if specified
            if categories:
                if not any(cat in str(rel_path) for cat in categories):
                    continue
            
            merged_file = Path(root) / "merged.json"
            if merged_file.exists():
                json_files.append(merged_file)
        
        return json_files
    
    def _parse_all_passages(self, 
                            json_files: List[Path], 
                            max_passages: int = None,
                            verbose: bool = True):
        """Parse all JSON files into Passage objects."""
        count = 0
        
        for json_file in json_files:
            if max_passages and count >= max_passages:
                break
                
            try:
                passages = self._parse_json_file(json_file)
                for p in passages:
                    if max_passages and count >= max_passages:
                        break
                    self.passages.append(p)
                    count += 1
                    
                if verbose and count % 1000 == 0:
                    print(f"      ... {count} passages")
                    
            except Exception as e:
                if verbose:
                    print(f"      Warning: Could not parse {json_file}: {e}")
    
    def _parse_json_file(self, json_file: Path) -> List[Passage]:
        """Parse a single merged.json file into passages."""
        passages = []
        
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Determine source and category from path
        rel_path = json_file.parent.relative_to(self.json_path)
        parts = rel_path.parts
        
        # Extract category and source
        category = parts[0] if parts else "Unknown"
        source = parts[-1] if len(parts) > 1 else parts[0] if parts else "Unknown"
        
        # Map to domain
        domain = self._get_domain(category, source)
        
        # Parse text structure
        def extract_passages(text_data, ref_prefix=""):
            if isinstance(text_data, str):
                # Leaf node - actual text
                if text_data.strip():
                    passages.append(Passage(
                        ref=ref_prefix,
                        hebrew=text_data if self._is_hebrew(text_data) else "",
                        english=text_data if not self._is_hebrew(text_data) else "",
                        source=source,
                        category=category,
                        domain=domain
                    ))
            elif isinstance(text_data, list):
                for i, item in enumerate(text_data):
                    new_ref = f"{ref_prefix}.{i+1}" if ref_prefix else str(i+1)
                    extract_passages(item, new_ref)
            elif isinstance(text_data, dict):
                # Handle dict structure
                if 'text' in text_data:
                    extract_passages(text_data['text'], ref_prefix)
                if 'he' in text_data:
                    # Hebrew text
                    he_text = text_data['he']
                    en_text = text_data.get('en', '')
                    if isinstance(he_text, str) and he_text.strip():
                        passages.append(Passage(
                            ref=ref_prefix,
                            hebrew=he_text,
                            english=en_text if isinstance(en_text, str) else "",
                            source=source,
                            category=category,
                            domain=domain
                        ))
        
        # Handle different JSON structures
        if isinstance(data, dict):
            if 'text' in data:
                extract_passages(data['text'], source)
            elif 'he' in data:
                extract_passages(data, source)
            else:
                for key, value in data.items():
                    if key not in ['versions', 'versionTitle', 'language']:
                        extract_passages(value, f"{source}.{key}")
        elif isinstance(data, list):
            extract_passages(data, source)
        
        return passages
    
    def _is_hebrew(self, text: str) -> bool:
        """Check if text is primarily Hebrew."""
        if not text:
            return False
        hebrew_chars = sum(1 for c in text if '\u0590' <= c <= '\u05FF')
        return hebrew_chars > len(text) * 0.3
    
    def _get_domain(self, category: str, source: str) -> str:
        """Map category/source to ethical domain."""
        # Check specific source first
        if source in CATEGORY_TO_DOMAIN:
            return CATEGORY_TO_DOMAIN[source]
        # Then category
        if category in CATEGORY_TO_DOMAIN:
            return CATEGORY_TO_DOMAIN[category]
        # Fuzzy match
        for key, domain in CATEGORY_TO_DOMAIN.items():
            if key.lower() in source.lower() or key.lower() in category.lower():
                return domain
        return "general"
    
    def _detect_all_gates(self, verbose: bool = True):
        """Detect Hohfeldian states and semantic gates in all passages."""
        for i, passage in enumerate(self.passages):
            self._detect_gates(passage)
            
            if verbose and (i + 1) % 1000 == 0:
                print(f"      ... analyzed {i + 1} passages")
        
        if verbose:
            # Print gate statistics
            total_binding = sum(self.gate_counts['binding'].values())
            total_release = sum(self.gate_counts['release'].values())
            print(f"      Binding gates detected: {total_binding}")
            print(f"      Release gates detected: {total_release}")
    
    def _detect_gates(self, passage: Passage):
        """Detect semantic gates in a single passage."""
        combined_text = f"{passage.hebrew} {passage.english}"
        
        detected = []
        state_scores = {s: 0.0 for s in HohfeldianState}
        
        # Check Hebrew binding triggers
        for pattern, (name, conf) in HEBREW_BINDING_TRIGGERS.items():
            if re.search(pattern, passage.hebrew):
                detected.append({
                    'type': 'binding',
                    'trigger': name,
                    'pattern': pattern,
                    'confidence': conf,
                    'language': 'he',
                    'transition': ('L', 'O'),
                    'group_element': 'r³'  # L → O
                })
                state_scores[HohfeldianState.O] += conf
                self.gate_counts['binding'][name] += 1
        
        # Check Hebrew release triggers
        for pattern, (name, conf) in HEBREW_RELEASE_TRIGGERS.items():
            if re.search(pattern, passage.hebrew):
                detected.append({
                    'type': 'release',
                    'trigger': name,
                    'pattern': pattern,
                    'confidence': conf,
                    'language': 'he',
                    'transition': ('O', 'L'),
                    'group_element': 'r'  # O → L
                })
                state_scores[HohfeldianState.L] += conf
                self.gate_counts['release'][name] += 1
        
        # Check English binding triggers
        for pattern, (name, conf) in ENGLISH_BINDING_TRIGGERS.items():
            if re.search(pattern, passage.english, re.IGNORECASE):
                detected.append({
                    'type': 'binding',
                    'trigger': name,
                    'pattern': pattern,
                    'confidence': conf,
                    'language': 'en',
                    'transition': ('L', 'O'),
                    'group_element': 'r³'
                })
                state_scores[HohfeldianState.O] += conf * 0.8  # Lower weight for English
                self.gate_counts['binding'][name] += 1
        
        # Check English release triggers  
        for pattern, (name, conf) in ENGLISH_RELEASE_TRIGGERS.items():
            if re.search(pattern, passage.english, re.IGNORECASE):
                detected.append({
                    'type': 'release',
                    'trigger': name,
                    'pattern': pattern,
                    'confidence': conf,
                    'language': 'en',
                    'transition': ('O', 'L'),
                    'group_element': 'r'
                })
                state_scores[HohfeldianState.L] += conf * 0.8
                self.gate_counts['release'][name] += 1
        
        # Check nullifiers
        for pattern, (name, conf) in NULLIFIER_PATTERNS.items():
            if re.search(pattern, combined_text, re.IGNORECASE):
                detected.append({
                    'type': 'nullifier',
                    'trigger': name,
                    'pattern': pattern,
                    'confidence': conf,
                    'note': 'Universal nullifier - overrides obligations'
                })
                self.gate_counts['nullifier'][name] += 1
        
        # Check for disputes (machloket indicators)
        dispute_patterns = [
            r'מחלוקת',           # machloket
            r'פליגי',            # dispute
            r'רבי .+ אומר',     # Rabbi X says
            r'תנא קמא',         # first tanna
            r'dispute',
            r'disagree',
        ]
        for pattern in dispute_patterns:
            if re.search(pattern, combined_text, re.IGNORECASE):
                passage.is_disputed = True
                break
        
        # Determine primary state
        passage.detected_gates = detected
        if state_scores:
            max_state = max(state_scores, key=state_scores.get)
            max_score = state_scores[max_state]
            if max_score > 0:
                passage.primary_state = max_state
                passage.state_confidence = min(max_score, 1.0)
    
    def _build_modules(self):
        """Build domain-specific ethical modules."""
        # Group passages by domain
        domain_passages = defaultdict(list)
        for p in self.passages:
            domain_passages[p.domain].append(p)
        
        # Build module for each domain
        for domain, passages in domain_passages.items():
            module = EthicalModule(
                name=domain.replace('_', ' ').title(),
                domain=domain,
                n_passages=len(passages),
                sources=set(p.source for p in passages)
            )
            
            # Calculate state distribution
            state_counts = Counter(p.primary_state for p in passages if p.primary_state)
            total = sum(state_counts.values())
            if total > 0:
                module.state_distribution = {
                    s.name: state_counts.get(s, 0) / total 
                    for s in HohfeldianState
                }
            
            # Collect gates
            for p in passages:
                for gate in p.detected_gates:
                    if gate['type'] == 'binding':
                        module.binding_gates.append(gate)
                    elif gate['type'] == 'release':
                        module.release_gates.append(gate)
                    elif gate['type'] == 'nullifier':
                        module.nullifiers.append(gate)
            
            # Find exemplar passages (high-confidence, representative)
            exemplars = sorted(
                [p for p in passages if p.state_confidence > 0.7],
                key=lambda x: x.state_confidence,
                reverse=True
            )[:10]
            
            module.exemplar_passages = [
                {
                    'ref': p.ref,
                    'hebrew': p.hebrew[:200] + '...' if len(p.hebrew) > 200 else p.hebrew,
                    'english': p.english[:200] + '...' if len(p.english) > 200 else p.english,
                    'state': p.primary_state.name if p.primary_state else None,
                    'confidence': p.state_confidence
                }
                for p in exemplars
            ]
            
            self.modules[domain] = module
    
    def _construct_dag(self) -> EMDAG:
        """Construct the final EM-DAG."""
        dag = EMDAG()
        
        # Meta information
        dag.meta = {
            'source': 'Sefaria-Export',
            'framework': 'NA-SQND v4.1 D₄×U(1)_H',
            'extraction_date': '2026-01-10',
            'n_passages': len(self.passages),
            'n_modules': len(self.modules),
            'baseline_comparison': 'Dear Abby 32-year corpus',
            'temporal_span': '~2000 years (Tanakh through Modern)',
        }
        
        # Structural constraints (D₄ gauge theory)
        dag.structural_constraints = {
            'correlative_lock': {
                'description': 'D₄ s-reflection: exact correlative pairing',
                'rules': [
                    'IF party_A has O toward party_B THEN party_B has C against party_A',
                    'IF party_A has L toward party_B THEN party_B has N against party_A',
                ]
            },
            'negation_symmetry': {
                'description': 'D₄ r² element: negation pairing',
                'rules': [
                    'O and L are negations (r²)',
                    'C and N are negations (r²)',
                ]
            },
            'gate_composition': {
                'description': 'Non-abelian gate composition',
                'rule': 'Gate₁ then Gate₂ = g₁·g₂ (order matters)',
            },
            'universal_nullifiers': {
                'description': 'Gates that override all obligations',
                'examples': ['pikuach_nefesh (life danger)', 'abuse', 'coercion']
            }
        }
        
        # Gate statistics
        dag.gate_statistics = {
            'binding_triggers': dict(self.gate_counts['binding'].most_common(20)),
            'release_triggers': dict(self.gate_counts['release'].most_common(20)),
            'nullifiers': dict(self.gate_counts['nullifier'].most_common(10)),
        }
        
        # Add modules (convert to dicts for JSON serialization)
        for name, module in self.modules.items():
            dag.modules[name] = module
        
        # Temporal stability analysis placeholder
        dag.temporal_stability = {
            'note': 'Cross-temporal gate stability analysis',
            'hypothesis': 'Core semantic gates stable across 2000 years',
            'comparison_to_dear_abby': {
                'promise_binding': 'Dear Abby: 32.2% O-rate | Hebrew: TBD',
                'abuse_nullifier': 'Dear Abby: universal (n=582) | Hebrew: pikuach nefesh',
            }
        }
        
        return dag


# =============================================================================
# SERIALIZATION
# =============================================================================

def serialize_dag(dag: EMDAG) -> dict:
    """Convert EMDAG to JSON-serializable dict."""
    result = {
        'meta': dag.meta,
        'structural_constraints': dag.structural_constraints,
        'gate_statistics': dag.gate_statistics,
        'temporal_stability': dag.temporal_stability,
        'modules': {}
    }
    
    for name, module in dag.modules.items():
        result['modules'][name] = {
            'name': module.name,
            'domain': module.domain,
            'n_passages': module.n_passages,
            'sources': list(module.sources),
            'state_distribution': module.state_distribution,
            'binding_gates_count': len(module.binding_gates),
            'release_gates_count': len(module.release_gates),
            'nullifiers_count': len(module.nullifiers),
            'top_binding_triggers': Counter(
                g['trigger'] for g in module.binding_gates
            ).most_common(5),
            'top_release_triggers': Counter(
                g['trigger'] for g in module.release_gates
            ).most_common(5),
            'exemplar_passages': module.exemplar_passages,
        }
    
    return result


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Extract EM-DAG from Sefaria-Export repository'
    )
    parser.add_argument(
        'sefaria_path',
        help='Path to Sefaria-Export repository'
    )
    parser.add_argument(
        '--output', '-o',
        default='hebrew_em_dag.json',
        help='Output JSON file (default: hebrew_em_dag.json)'
    )
    parser.add_argument(
        '--categories', '-c',
        nargs='+',
        help='Only process specific categories (e.g., Talmud Mishnah)'
    )
    parser.add_argument(
        '--max-passages', '-m',
        type=int,
        help='Maximum passages to process'
    )
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress progress output'
    )
    
    args = parser.parse_args()
    
    # Run extraction
    extractor = SefariaEMExtractor(args.sefaria_path)
    dag = extractor.extract(
        categories=args.categories,
        max_passages=args.max_passages,
        verbose=not args.quiet
    )
    
    # Serialize and save
    output_data = serialize_dag(dag)
    
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nEM-DAG saved to: {args.output}")
    print(f"File size: {os.path.getsize(args.output) / 1024:.1f} KB")


if __name__ == '__main__':
    main()
