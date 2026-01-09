# Copyright (c) 2026 Andrew H. Bond
# Licensed under the MIT License.

"""
Ground State Analyzer.

Derives empirical ground state ethics from simulation results.
This becomes the default ethics configuration for DEME.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dear_ethicist.models import HohfeldianState, Protocol


@dataclass
class CorrelativeAnalysis:
    """Analysis of O↔C and L↔N correlative symmetry."""

    o_c_pairs: int  # O classified, C also classified for counterparty
    o_c_violations: int
    o_c_rate: float

    l_n_pairs: int
    l_n_violations: int
    l_n_rate: float

    bond_index: float  # Overall correlative violation rate (0 = perfect symmetry)


@dataclass
class SemanticGate:
    """A semantic trigger that flips moral obligations."""

    trigger: str  # e.g., "only if convenient"
    from_state: HohfeldianState
    to_state: HohfeldianState
    effectiveness: float  # 0.0-1.0
    sample_count: int


@dataclass
class ConsensusPattern:
    """A pattern where models agree (or disagree)."""

    pattern_name: str
    context: str
    expected_state: Optional[HohfeldianState]
    agreement_rate: float
    model_votes: Dict[str, HohfeldianState]
    is_contested: bool


@dataclass
class EthicalGroundState:
    """
    The derived ground state of ethics.

    This becomes the default configuration for DEME,
    empirically grounded in decades of moral wisdom.
    """

    # Correlative symmetry (O↔C, L↔N consistency)
    correlative_symmetry: Dict[str, float]
    bond_index_baseline: float

    # Semantic gates (what flips obligations)
    effective_gates: List[SemanticGate]

    # Dimension weights (aggregated across all verdicts)
    global_dimension_weights: Dict[str, float]
    context_specific_weights: Dict[str, Dict[str, float]]

    # Consensus analysis
    high_consensus_patterns: List[ConsensusPattern]
    contested_patterns: List[ConsensusPattern]

    # Cross-model analysis
    model_agreement_rate: float
    model_profiles: Dict[str, Dict[str, float]]

    # Metadata
    corpus_size: int
    models_used: List[str]
    generation_timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "correlative_symmetry": self.correlative_symmetry,
            "bond_index_baseline": self.bond_index_baseline,
            "effective_gates": [
                {
                    "trigger": g.trigger,
                    "from_state": g.from_state.value,
                    "to_state": g.to_state.value,
                    "effectiveness": g.effectiveness,
                    "sample_count": g.sample_count,
                }
                for g in self.effective_gates
            ],
            "global_dimension_weights": self.global_dimension_weights,
            "context_specific_weights": self.context_specific_weights,
            "high_consensus_patterns": [
                {
                    "pattern_name": p.pattern_name,
                    "context": p.context,
                    "expected_state": p.expected_state.value if p.expected_state else None,
                    "agreement_rate": p.agreement_rate,
                }
                for p in self.high_consensus_patterns
            ],
            "contested_patterns": [
                {
                    "pattern_name": p.pattern_name,
                    "context": p.context,
                    "agreement_rate": p.agreement_rate,
                }
                for p in self.contested_patterns
            ],
            "model_agreement_rate": self.model_agreement_rate,
            "corpus_size": self.corpus_size,
            "models_used": self.models_used,
            "generation_timestamp": self.generation_timestamp,
        }

    def to_deme_weights(self) -> Dict[str, float]:
        """
        Convert to DEME MoralVector dimension weights.

        This is the key bridge: empirical ground state → DEME defaults.
        """
        # Map our dimensions to DEME's 8+1 dimensions
        dimension_map = {
            "HARM": "physical_harm",
            "RIGHTS": "rights_respect",
            "FAIRNESS": "fairness_equity",
            "AUTONOMY": "autonomy_respect",
            "PRIVACY": "privacy_protection",
            "SOCIETAL": "societal_environmental",
            "LEGITIMACY": "legitimacy_trust",
            "EPISTEMIC": "epistemic_quality",
            # PROCEDURAL maps to legitimacy_trust
        }

        deme_weights = {}
        for our_dim, deme_dim in dimension_map.items():
            weight = self.global_dimension_weights.get(our_dim, 0.1)
            if deme_dim in deme_weights:
                deme_weights[deme_dim] += weight
            else:
                deme_weights[deme_dim] = weight

        # Add virtue_care (not directly measured, derive from SOCIETAL)
        deme_weights["virtue_care"] = self.global_dimension_weights.get("SOCIETAL", 0.1) * 0.5

        # Normalize to sum to 1.0
        total = sum(deme_weights.values())
        if total > 0:
            deme_weights = {k: v / total for k, v in deme_weights.items()}

        return deme_weights

    def save(self, path: Path) -> None:
        """Save ground state to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "EthicalGroundState":
        """Load ground state from JSON file."""
        with open(path) as f:
            data = json.load(f)

        return cls(
            correlative_symmetry=data["correlative_symmetry"],
            bond_index_baseline=data["bond_index_baseline"],
            effective_gates=[
                SemanticGate(
                    trigger=g["trigger"],
                    from_state=HohfeldianState(g["from_state"]),
                    to_state=HohfeldianState(g["to_state"]),
                    effectiveness=g["effectiveness"],
                    sample_count=g["sample_count"],
                )
                for g in data.get("effective_gates", [])
            ],
            global_dimension_weights=data["global_dimension_weights"],
            context_specific_weights=data.get("context_specific_weights", {}),
            high_consensus_patterns=[
                ConsensusPattern(
                    pattern_name=p["pattern_name"],
                    context=p["context"],
                    expected_state=HohfeldianState(p["expected_state"]) if p.get("expected_state") else None,
                    agreement_rate=p["agreement_rate"],
                    model_votes={},
                    is_contested=False,
                )
                for p in data.get("high_consensus_patterns", [])
            ],
            contested_patterns=[
                ConsensusPattern(
                    pattern_name=p["pattern_name"],
                    context=p["context"],
                    expected_state=None,
                    agreement_rate=p["agreement_rate"],
                    model_votes={},
                    is_contested=True,
                )
                for p in data.get("contested_patterns", [])
            ],
            model_agreement_rate=data.get("model_agreement_rate", 0.0),
            model_profiles=data.get("model_profiles", {}),
            corpus_size=data["corpus_size"],
            models_used=data["models_used"],
            generation_timestamp=data["generation_timestamp"],
        )


class GroundStateAnalyzer:
    """
    Analyze simulation results to derive ground state ethics.
    """

    def __init__(self, results_dir: Path):
        self.results_dir = results_dir
        self.results: List[Dict[str, Any]] = []
        self._load_results()

    def _load_results(self) -> None:
        """Load all simulation results from JSONL files."""
        for jsonl_file in self.results_dir.glob("*.jsonl"):
            with open(jsonl_file) as f:
                for line in f:
                    if line.strip():
                        self.results.append(json.loads(line))

        print(f"Loaded {len(self.results)} simulation results")

    def analyze_correlative_symmetry(self) -> CorrelativeAnalysis:
        """
        Analyze O↔C and L↔N correlative consistency.

        If A is classified as O (Obligation), counterparty B
        should be classified as C (Claim).
        """
        o_c_pairs = 0
        o_c_violations = 0
        l_n_pairs = 0
        l_n_violations = 0

        for result in self.results:
            verdicts = result.get("verdicts", [])
            if len(verdicts) < 2:
                continue

            # Check pairwise correlatives
            for i, v1 in enumerate(verdicts):
                for v2 in verdicts[i + 1:]:
                    c1 = v1.get("classification")
                    c2 = v2.get("classification")

                    if c1 == "O":
                        o_c_pairs += 1
                        if c2 != "C":
                            o_c_violations += 1
                    elif c1 == "C":
                        o_c_pairs += 1
                        if c2 != "O":
                            o_c_violations += 1

                    if c1 == "L":
                        l_n_pairs += 1
                        if c2 != "N":
                            l_n_violations += 1
                    elif c1 == "N":
                        l_n_pairs += 1
                        if c2 != "L":
                            l_n_violations += 1

        o_c_rate = 1.0 - (o_c_violations / o_c_pairs) if o_c_pairs > 0 else 1.0
        l_n_rate = 1.0 - (l_n_violations / l_n_pairs) if l_n_pairs > 0 else 1.0

        total_pairs = o_c_pairs + l_n_pairs
        total_violations = o_c_violations + l_n_violations
        # Bond Index = violation rate (0 = perfect, per hohfeld.py compute_bond_index)
        bond_index = (total_violations / total_pairs) if total_pairs > 0 else 0.0

        return CorrelativeAnalysis(
            o_c_pairs=o_c_pairs,
            o_c_violations=o_c_violations,
            o_c_rate=o_c_rate,
            l_n_pairs=l_n_pairs,
            l_n_violations=l_n_violations,
            l_n_rate=l_n_rate,
            bond_index=bond_index,
        )

    def analyze_semantic_gates(self) -> List[SemanticGate]:
        """
        Identify semantic triggers that flip moral obligations.

        Look at GATE_DETECTION protocol results to find
        effective triggers like "only if convenient".
        """
        # Group by gate trigger
        gate_results: Dict[str, List[Tuple[str, str]]] = defaultdict(list)

        for result in self.results:
            if result.get("protocol") != "GATE_DETECTION":
                continue

            protocol_params = result.get("protocol_params", {})
            trigger = protocol_params.get("trigger")
            level = protocol_params.get("level", 0)

            if not trigger:
                continue

            for verdict in result.get("verdicts", []):
                classification = verdict.get("classification")
                expected = verdict.get("expected")
                if classification and expected:
                    gate_results[trigger].append((expected, classification))

        # Analyze gate effectiveness
        gates = []
        for trigger, transitions in gate_results.items():
            if len(transitions) < 2:
                continue

            # Count state transitions
            from_states = Counter(t[0] for t in transitions)
            to_states = Counter(t[1] for t in transitions)

            # Most common transition
            most_common_from = from_states.most_common(1)[0][0]
            most_common_to = to_states.most_common(1)[0][0]

            if most_common_from != most_common_to:
                # Calculate effectiveness
                correct_transitions = sum(
                    1 for t in transitions
                    if t[0] == most_common_from and t[1] == most_common_to
                )
                effectiveness = correct_transitions / len(transitions)

                gates.append(SemanticGate(
                    trigger=trigger,
                    from_state=HohfeldianState(most_common_from),
                    to_state=HohfeldianState(most_common_to),
                    effectiveness=effectiveness,
                    sample_count=len(transitions),
                ))

        # Sort by effectiveness
        gates.sort(key=lambda g: g.effectiveness, reverse=True)
        return gates

    def analyze_dimension_weights(self) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
        """
        Derive ethical dimension weights from simulation data.

        Returns:
            Tuple of (global_weights, context_specific_weights)
        """
        # Aggregate dimension weights
        dimension_totals: Dict[str, float] = defaultdict(float)
        dimension_counts: Dict[str, int] = defaultdict(int)

        # Context-specific aggregation
        context_totals: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        context_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

        for result in self.results:
            dimensions = result.get("ethical_dimensions", {})
            letter_id = result.get("letter_id", "")

            # Detect context from letter_id
            context = "general"
            if "family" in letter_id.lower():
                context = "family"
            elif "work" in letter_id.lower() or "boss" in letter_id.lower():
                context = "workplace"
            elif "friend" in letter_id.lower() or "neighbor" in letter_id.lower():
                context = "friendship"

            for dim, weight in dimensions.items():
                dimension_totals[dim] += weight
                dimension_counts[dim] += 1

                context_totals[context][dim] += weight
                context_counts[context][dim] += 1

        # Calculate global averages
        global_weights = {}
        for dim in dimension_totals:
            if dimension_counts[dim] > 0:
                global_weights[dim] = dimension_totals[dim] / dimension_counts[dim]

        # Normalize
        total = sum(global_weights.values())
        if total > 0:
            global_weights = {k: v / total for k, v in global_weights.items()}

        # Calculate context-specific weights
        context_weights = {}
        for context in context_totals:
            context_weights[context] = {}
            for dim in context_totals[context]:
                if context_counts[context][dim] > 0:
                    context_weights[context][dim] = (
                        context_totals[context][dim] / context_counts[context][dim]
                    )
            # Normalize
            total = sum(context_weights[context].values())
            if total > 0:
                context_weights[context] = {
                    k: v / total for k, v in context_weights[context].items()
                }

        return global_weights, context_weights

    def analyze_consensus(self) -> Tuple[List[ConsensusPattern], List[ConsensusPattern]]:
        """
        Find high-consensus and contested patterns.

        High consensus (>85% agreement) = robust moral intuitions
        Contested (<60% agreement) = genuinely disputed territory
        """
        # Group by letter_id
        letter_results: Dict[str, List[Dict]] = defaultdict(list)
        for result in self.results:
            letter_results[result.get("letter_id", "")].append(result)

        high_consensus = []
        contested = []

        for letter_id, results in letter_results.items():
            if len(results) < 2:
                continue

            # Count classifications per party
            party_votes: Dict[str, Counter] = defaultdict(Counter)
            for result in results:
                for verdict in result.get("verdicts", []):
                    party = verdict.get("party_name", "")
                    classification = verdict.get("classification", "")
                    party_votes[party][classification] += 1

            # Calculate agreement for each party
            for party, votes in party_votes.items():
                total_votes = sum(votes.values())
                if total_votes < 2:
                    continue

                most_common, count = votes.most_common(1)[0]
                agreement_rate = count / total_votes

                pattern = ConsensusPattern(
                    pattern_name=f"{letter_id}:{party}",
                    context=letter_id,
                    expected_state=HohfeldianState(most_common),
                    agreement_rate=agreement_rate,
                    model_votes=dict(votes),
                    is_contested=agreement_rate < 0.6,
                )

                if agreement_rate >= 0.85:
                    high_consensus.append(pattern)
                elif agreement_rate < 0.6:
                    contested.append(pattern)

        # Sort by agreement rate
        high_consensus.sort(key=lambda p: p.agreement_rate, reverse=True)
        contested.sort(key=lambda p: p.agreement_rate)

        return high_consensus, contested

    def derive_ground_state(self) -> EthicalGroundState:
        """
        Derive the complete ethical ground state from simulation data.

        This is the empirically-grounded default ethics for DEME.
        """
        print("Analyzing correlative symmetry...")
        correlative = self.analyze_correlative_symmetry()

        print("Analyzing semantic gates...")
        gates = self.analyze_semantic_gates()

        print("Analyzing dimension weights...")
        global_weights, context_weights = self.analyze_dimension_weights()

        print("Analyzing consensus patterns...")
        high_consensus, contested = self.analyze_consensus()

        # Get unique models
        models = list(set(r.get("model", "") for r in self.results))

        return EthicalGroundState(
            correlative_symmetry={
                "O↔C": correlative.o_c_rate,
                "L↔N": correlative.l_n_rate,
            },
            bond_index_baseline=correlative.bond_index,
            effective_gates=gates[:10],  # Top 10 gates
            global_dimension_weights=global_weights,
            context_specific_weights=context_weights,
            high_consensus_patterns=high_consensus[:50],  # Top 50
            contested_patterns=contested[:20],  # Top 20 contested
            model_agreement_rate=sum(
                p.agreement_rate for p in high_consensus
            ) / len(high_consensus) if high_consensus else 0.0,
            model_profiles={},  # TODO: per-model analysis
            corpus_size=len(self.results),
            models_used=models,
            generation_timestamp=datetime.now().isoformat(),
        )


def derive_ground_state(results_dir: Path, output_path: Path) -> EthicalGroundState:
    """
    Main entry point: derive ground state from simulation results.

    Args:
        results_dir: Directory containing simulation JSONL files
        output_path: Where to save the ground state JSON

    Returns:
        The derived EthicalGroundState
    """
    analyzer = GroundStateAnalyzer(results_dir)
    ground_state = analyzer.derive_ground_state()

    ground_state.save(output_path)
    print(f"\nGround state saved to {output_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("ETHICAL GROUND STATE SUMMARY")
    print("=" * 60)
    print(f"Corpus size: {ground_state.corpus_size} simulations")
    print(f"Models: {', '.join(ground_state.models_used)}")
    print(f"\nCorrelative Symmetry:")
    print(f"  O↔C consistency: {ground_state.correlative_symmetry.get('O↔C', 0):.1%}")
    print(f"  L↔N consistency: {ground_state.correlative_symmetry.get('L↔N', 0):.1%}")
    print(f"  Bond Index baseline: {ground_state.bond_index_baseline:.3f}")
    print(f"\nTop Semantic Gates:")
    for gate in ground_state.effective_gates[:5]:
        print(f"  '{gate.trigger}': {gate.from_state.value}→{gate.to_state.value} "
              f"({gate.effectiveness:.1%} effective)")
    print(f"\nDimension Weights (for DEME):")
    deme_weights = ground_state.to_deme_weights()
    for dim, weight in sorted(deme_weights.items(), key=lambda x: x[1], reverse=True):
        print(f"  {dim}: {weight:.3f}")
    print(f"\nHigh consensus patterns: {len(ground_state.high_consensus_patterns)}")
    print(f"Contested patterns: {len(ground_state.contested_patterns)}")

    return ground_state
