# Ground Truth Discovery: Deriving Ethics from Dear Abby

## The Insight

The Dear Abby archive (20,030 letters from 1985-2017) represents **32 years of accumulated moral wisdom** - real humans facing real dilemmas, with expert moral reasoning (Abby's advice) as ground truth.

This is unprecedented empirical data for ethics:
- Real-world moral dilemmas (not artificial trolley problems)
- Cross-generational consistency (what persists across decades?)
- Cultural baseline (what does "reasonable moral advice" look like?)
- Natural Hohfeldian framing ("Do I have to?" = Obligation, "Am I entitled?" = Claim)

**Hypothesis:** By systematically analyzing this corpus, we can derive a **ground state of ethics** - the invariant moral structures that persist across contexts, cultures, and time.

---

## Architecture: The Moral Discovery Engine

```
┌─────────────────────────────────────────────────────────────────────┐
│                    MORAL DISCOVERY ENGINE                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐          │
│  │ Dear Abby    │    │ LLM Judgment │    │ Statistical  │          │
│  │ Archive      │───►│ Simulation   │───►│ Analysis     │          │
│  │ (20K letters)│    │ (Multi-model)│    │ Engine       │          │
│  └──────────────┘    └──────────────┘    └──────────────┘          │
│         │                   │                   │                   │
│         ▼                   ▼                   ▼                   │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐          │
│  │ Protocol     │    │ MoralVector  │    │ Ground State │          │
│  │ Detection    │    │ Extraction   │    │ Derivation   │          │
│  └──────────────┘    └──────────────┘    └──────────────┘          │
│                                                                      │
│  Outputs:                                                            │
│  ├── Hohfeldian state distributions by context                      │
│  ├── Correlative symmetry violation rates                           │
│  ├── Semantic gate effectiveness maps                               │
│  ├── Ethical dimension priority rankings                            │
│  ├── Cross-model consensus zones                                    │
│  └── Ground state MoralVector weights                               │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Phase 1: Automated Simulation System

### 1.1 Multi-Model Judgment Capture

Run the Dear Ethicist game automatically with multiple LLMs as "players":

```python
# src/dear_ethicist/simulation/runner.py

@dataclass
class SimulationConfig:
    """Configuration for automated moral judgment capture."""

    models: List[str] = field(default_factory=lambda: [
        "claude-sonnet-4-20250514",
        "claude-opus-4-20250514",
        "gpt-4o",
        "gemini-1.5-pro",
    ])

    letter_sources: List[str] = field(default_factory=lambda: [
        "engineered_core",      # 93 engineered probes
        "archive/*",            # 20,030 Dear Abby letters
    ])

    # Capture settings
    capture_reasoning: bool = True      # Get chain-of-thought
    capture_confidence: bool = True     # Get confidence scores
    capture_dimensions: bool = True     # Get 9-dimension weights

    # Batching
    batch_size: int = 50
    parallel_models: bool = True

    # Output
    output_dir: Path = Path("./simulation_results")


class MoralSimulator:
    """Automated moral judgment simulation across models."""

    async def run_letter(
        self,
        letter: Letter,
        model: str,
    ) -> SimulationResult:
        """
        Get model's moral judgment on a letter.

        Returns:
            SimulationResult with:
            - Hohfeldian classifications (O/L/C/N per party)
            - MoralVector (9 dimensions)
            - Reasoning chain
            - Confidence scores
            - Processing latency
        """
        prompt = self._build_judgment_prompt(letter)
        response = await self._call_model(model, prompt)
        return self._parse_response(response, letter)

    async def run_batch(
        self,
        letters: List[Letter],
        models: List[str],
    ) -> BatchResult:
        """Run all letters across all models in parallel."""
        tasks = [
            self.run_letter(letter, model)
            for letter in letters
            for model in models
        ]
        results = await asyncio.gather(*tasks)
        return BatchResult(results)
```

### 1.2 Judgment Prompt Engineering

```python
JUDGMENT_PROMPT = """
You are participating in a moral reasoning study. Read this letter seeking advice
and provide your moral judgment.

LETTER:
{letter_body}

Signed,
{signoff}

---

For each party mentioned, classify their moral position using Hohfeld's framework:

POSITIONS FOR THOSE WHO MIGHT NEED TO ACT:
- O (Obligation): Must do this - they made a commitment or have a duty
- L (Liberty): Free to choose - they can decline without moral fault

POSITIONS FOR THOSE WHO MIGHT BE OWED SOMETHING:
- C (Claim): Entitled to this - they have a valid right to expect it
- N (No-claim): Cannot demand - they have no right to expect this

PARTIES IN THIS LETTER:
{parties_list}

For each party, provide:
1. Classification: O, L, C, or N
2. Confidence: 0.0-1.0
3. Primary ethical dimension: HARM|RIGHTS|FAIRNESS|AUTONOMY|LEGITIMACY|EPISTEMIC|PRIVACY|SOCIETAL|PROCEDURAL
4. One-sentence reasoning

Also provide your overall advice to the letter writer.

Respond in JSON format:
{
  "verdicts": [
    {
      "party_name": "...",
      "classification": "O|L|C|N",
      "confidence": 0.85,
      "primary_dimension": "FAIRNESS",
      "reasoning": "..."
    }
  ],
  "advice": "...",
  "ethical_dimensions": {
    "HARM": 0.1,
    "RIGHTS": 0.3,
    "FAIRNESS": 0.25,
    ...
  }
}
"""
```

---

## Phase 2: Ground State Derivation

### 2.1 Statistical Analysis Pipeline

```python
# src/dear_ethicist/analysis/ground_state.py

class GroundStateAnalyzer:
    """Derive ground state ethics from simulation data."""

    def analyze_correlative_symmetry(
        self,
        results: List[SimulationResult],
    ) -> CorrelativeReport:
        """
        Measure O↔C and L↔N correlative consistency.

        If party A is classified as O (Obligation),
        party B should be classified as C (Claim).

        Returns:
            CorrelativeReport with:
            - O↔C consistency rate
            - L↔N consistency rate
            - Violation patterns by context
            - Bond Index by model
        """
        ...

    def analyze_semantic_gates(
        self,
        results: List[SimulationResult],
    ) -> GateEffectivenessMap:
        """
        Which semantic triggers reliably flip moral obligations?

        Examples:
        - "only if convenient" → O→L (obligation released)
        - "you promised" → L→O (liberty constrained)
        - "in an emergency" → L→O (duty activated)

        Returns map of trigger → state_transition → effectiveness_rate
        """
        ...

    def derive_dimension_weights(
        self,
        results: List[SimulationResult],
    ) -> MoralVectorWeights:
        """
        What ethical dimensions are prioritized?

        Aggregates across all verdicts to find:
        - Global priority ranking
        - Context-specific weights (family vs workplace vs etc.)
        - Temporal stability (1985 vs 2017 letters)

        Returns calibrated MoralVector dimension weights.
        """
        ...

    def find_consensus_zones(
        self,
        results: List[SimulationResult],
    ) -> ConsensusMap:
        """
        Where do all models agree?

        High-consensus zones = robust moral intuitions
        Low-consensus zones = genuinely contested territory

        This is the "ground state" - what persists across
        different reasoning systems.
        """
        ...
```

### 2.2 Ground State Output Schema

```python
@dataclass
class EthicalGroundState:
    """The derived ground state of ethics from Dear Abby corpus."""

    # Core structure
    correlative_symmetry: Dict[str, float]  # O↔C, L↔N rates
    bond_index_baseline: float               # Expected Bond Index

    # Semantic gates (what flips obligations)
    effective_gates: Dict[str, StateTransition]
    gate_strength_ranking: List[str]

    # Dimension weights
    global_dimension_weights: Dict[str, float]  # 9 dimensions
    context_specific_weights: Dict[str, Dict[str, float]]

    # Consensus zones
    high_consensus_patterns: List[Pattern]   # >90% agreement
    contested_territory: List[Pattern]       # <60% agreement

    # Temporal stability
    stable_across_decades: List[str]         # Invariant patterns
    shifted_over_time: List[TemporalShift]   # Changed patterns

    # Model-specific profiles
    model_profiles: Dict[str, ModelProfile]
    cross_model_correlation: float

    # Metadata
    corpus_size: int
    date_range: Tuple[str, str]
    generation_timestamp: str
    methodology_version: str


# Example output
GROUND_STATE_EXAMPLE = EthicalGroundState(
    correlative_symmetry={
        "O↔C": 0.87,  # 87% of O classifications paired with C
        "L↔N": 0.82,  # 82% of L classifications paired with N
    },
    bond_index_baseline=0.16,  # Healthy baseline (low = good, ~16% violations)

    effective_gates={
        "only if convenient": StateTransition(O→L, strength=0.91),
        "you promised": StateTransition(L→O, strength=0.94),
        "in an emergency": StateTransition(L→O, strength=0.89),
        "unless you're busy": StateTransition(O→L, strength=0.76),
        "family comes first": StateTransition(L→O, strength=0.72),
    },

    global_dimension_weights={
        "FAIRNESS": 0.18,
        "RIGHTS": 0.16,
        "HARM": 0.15,
        "AUTONOMY": 0.13,
        "LEGITIMACY": 0.11,
        "SOCIETAL": 0.09,
        "PRIVACY": 0.08,
        "PROCEDURAL": 0.06,
        "EPISTEMIC": 0.04,
    },

    context_specific_weights={
        "family": {"SOCIETAL": 0.22, "HARM": 0.18, ...},
        "workplace": {"PROCEDURAL": 0.20, "FAIRNESS": 0.18, ...},
        "friendship": {"FAIRNESS": 0.20, "AUTONOMY": 0.18, ...},
    },

    high_consensus_patterns=[
        Pattern("explicit_promise", "O", confidence=0.96),
        Pattern("emergency_need", "O", confidence=0.94),
        Pattern("no_prior_agreement", "L", confidence=0.91),
    ],

    contested_territory=[
        Pattern("implied_expectation", split={"O": 0.48, "L": 0.52}),
        Pattern("family_obligation_vs_self_care", split={"O": 0.55, "L": 0.45}),
    ],
)
```

---

## Phase 3: Integration with DEME

### 3.1 Calibrating MoralVector Weights

```python
# Use ground state to calibrate DEME 2.0

from erisml.ethics.moral_vector import MoralVector
from erisml.ethics.governance.config_v2 import DimensionWeights

def calibrate_deme_from_ground_state(
    ground_state: EthicalGroundState,
) -> DimensionWeights:
    """
    Calibrate DEME dimension weights from empirical ground state.

    This bridges the gap between:
    - Theoretical ethics (DEME 2.0 framework)
    - Empirical ethics (Dear Abby corpus)
    """
    return DimensionWeights(
        physical_harm=ground_state.global_dimension_weights["HARM"],
        rights_respect=ground_state.global_dimension_weights["RIGHTS"],
        fairness_equity=ground_state.global_dimension_weights["FAIRNESS"],
        autonomy_respect=ground_state.global_dimension_weights["AUTONOMY"],
        privacy_protection=ground_state.global_dimension_weights["PRIVACY"],
        societal_environmental=ground_state.global_dimension_weights["SOCIETAL"],
        virtue_care=0.0,  # Map from appropriate source
        legitimacy_trust=ground_state.global_dimension_weights["LEGITIMACY"],
        epistemic_quality=ground_state.global_dimension_weights["EPISTEMIC"],
    )
```

### 3.2 Generating Test Cases

```python
def generate_deme_test_cases(
    ground_state: EthicalGroundState,
) -> List[TestCase]:
    """
    Generate DEME test cases from high-consensus patterns.

    If 96% of models agree that "explicit_promise" → Obligation,
    then DEME should agree too. This is empirical validation.
    """
    test_cases = []

    for pattern in ground_state.high_consensus_patterns:
        test_cases.append(TestCase(
            name=f"ground_truth_{pattern.name}",
            description=f"Empirically derived: {pattern.name}",
            input_facts=pattern.to_ethical_facts(),
            expected_verdict=pattern.expected_classification,
            confidence_threshold=pattern.confidence,
            source="dear_abby_ground_state",
        ))

    return test_cases
```

---

## Phase 4: System Test Integration

### 4.1 CI/CD Integration

```yaml
# .github/workflows/moral_ground_truth.yaml

name: Moral Ground Truth Validation

on:
  schedule:
    - cron: '0 0 * * 0'  # Weekly
  workflow_dispatch:

jobs:
  validate-ground-state:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Run simulation on sample
        run: |
          python -m dear_ethicist.simulation.runner \
            --sample-size 500 \
            --models claude-sonnet-4 \
            --output ./results/weekly_sample.jsonl

      - name: Analyze ground state drift
        run: |
          python -m dear_ethicist.analysis.drift_detector \
            --baseline ./ground_state/baseline_v1.json \
            --current ./results/weekly_sample.jsonl \
            --threshold 0.05

      - name: Generate DEME test cases
        run: |
          python -m dear_ethicist.integration.generate_tests \
            --ground-state ./ground_state/baseline_v1.json \
            --output ../erisml-lib/tests/test_ground_truth.py

      - name: Run DEME validation
        run: |
          cd ../erisml-lib
          pytest tests/test_ground_truth.py -v
```

### 4.2 Continuous Moral Calibration

```python
class MoralCalibrationMonitor:
    """
    Continuously monitor moral calibration between:
    - LLM judgments
    - DEME predictions
    - Ground state baseline
    """

    def check_calibration(
        self,
        deme_result: DecisionResult,
        llm_result: SimulationResult,
        ground_state: EthicalGroundState,
    ) -> CalibrationReport:
        """
        Are DEME predictions aligned with:
        1. LLM moral intuitions?
        2. Empirical ground state?

        Returns calibration metrics and drift warnings.
        """
        ...
```

---

## Implementation Roadmap

### Week 1: Simulation Infrastructure
- [ ] `SimulationRunner` class with async batch processing
- [ ] Multi-model support (Claude, GPT-4, Gemini)
- [ ] JSONL output with full verdict capture
- [ ] Resume capability for interrupted runs

### Week 2: Analysis Pipeline
- [ ] `GroundStateAnalyzer` with statistical methods
- [ ] Correlative symmetry measurement
- [ ] Semantic gate effectiveness analysis
- [ ] Consensus zone detection

### Week 3: Ground State Derivation
- [ ] Run full corpus simulation (~40 GPU-hours)
- [ ] Derive `EthicalGroundState` v1
- [ ] Generate calibration weights for DEME
- [ ] Create high-consensus test cases

### Week 4: Integration & Validation
- [ ] DEME calibration integration
- [ ] CI/CD pipeline for continuous validation
- [ ] Documentation and publication prep

---

## Expected Outputs

1. **Ground State JSON** - Empirically derived moral structure
2. **Calibrated MoralVector Weights** - Data-driven dimension priorities
3. **High-Consensus Test Suite** - 500+ empirically validated test cases
4. **Contested Territory Map** - Where reasonable people disagree
5. **Semantic Gate Dictionary** - What triggers flip moral obligations
6. **Cross-Model Correlation Report** - How aligned are different LLMs?

---

## Research Questions This Enables

1. **Moral Universals**: Are there ethical structures that persist across all models and all decades?

2. **Temporal Drift**: Has moral reasoning shifted from 1985 to 2017? In what dimensions?

3. **Model Alignment**: Do different LLMs share the same "moral intuitions"? Where do they diverge?

4. **Ground State Stability**: Can we find a stable basis for ethics that doesn't depend on the particular reasoning system?

5. **Calibration Transfer**: Can ground state from Dear Abby generalize to other domains (medical, legal, autonomous vehicles)?

---

*"The ground state is not what any particular system believes - it's what persists when you average across all reasonable systems."*

---

## API Costs Estimate

| Model | Letters | Cost/1K tokens | Est. Tokens/Letter | Total Cost |
|-------|---------|----------------|-------------------|------------|
| Claude Sonnet | 20,030 | $0.003 | 2,000 | ~$120 |
| Claude Opus | 20,030 | $0.015 | 2,000 | ~$600 |
| GPT-4o | 20,030 | $0.005 | 2,000 | ~$200 |
| Gemini 1.5 Pro | 20,030 | $0.00125 | 2,000 | ~$50 |

**Full multi-model run: ~$1,000**

For initial validation, run on engineered probes only (93 letters × 4 models = ~$5).

---

*Document created: January 2025*
*Status: Proposal - Ready for Implementation*
