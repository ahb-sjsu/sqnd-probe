# Dear Ethicist

[![CI](https://github.com/ahb-sjsu/sqnd-probe/actions/workflows/ci.yaml/badge.svg)](https://github.com/ahb-sjsu/sqnd-probe/actions/workflows/ci.yaml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An advice column game that measures the mathematical structure of moral reasoning.

## What Is This?

You play as a newly hired advice columnist for *The Morning Chronicle*. Letters arrive from readers tangled in everyday moral dilemmas:

> *"My friend promised to help me move, but texted 'only if convenient.' Can I still count on them?"*

> *"I lent my neighbor a drill two months ago. Do I have a right to demand it back?"*

> *"My coworker made a mistake and asked me to cover for them. Am I obligated to help?"*

Your job: read the letters, give advice, publish your verdicts.

**The twist:** These everyday questions map directly to formal normative positions. When someone asks "Do I have to...?" they're asking about *Obligation*. When they ask "Can they demand...?" they're asking about *Claims*. The game captures your moral reasoning structure while you think you're just giving advice.

## Installation

```bash
git clone https://github.com/ahb-sjsu/sqnd-probe.git
cd sqnd-probe
pip install -e ".[dev]"
```

## How to Play

### Start a Session

```bash
dear-ethicist play
```

### The Game Loop

1. **Read the letter** — A reader presents their dilemma
2. **Consider the situation** — Who owes what to whom?
3. **Render your verdict** — Classify each party's normative position
4. **Publish** — Your column goes live
5. **See reactions** — Readers respond (some agree, some don't)
6. **Next letter** — Repeat until you quit or finish the session

### The Verdict Screen

```
YOUR VERDICT (for your records)

Morgan's situation:
  [1] Morgan is obligated (promise stands)
  [2] Morgan is free to choose

Your claim:
  [1] You have a right to expect this
  [2] You can't demand this

Select: _
```

You're classifying parties into four positions:

| Position | Code | Meaning |
|----------|------|---------|
| Obligation | O | Must do it |
| Claim | C | Can demand it |
| Liberty | L | Free to choose |
| No-claim | N | Can't demand it |

## Scoring: The Bond Index

The game measures **structural consistency** in your moral reasoning using something called the *bond index*.

### Correlative Symmetry

Hohfeldian positions come in pairs:
- If Morgan is **Obligated** (O), you should have a **Claim** (C)
- If Morgan is **at Liberty** (L), you should have **No-claim** (N)

These are *correlatives*. They're logically linked. If you say Morgan must help (O), but also say you can't demand help (N), that's a *symmetry violation*.

### What the Bond Index Measures

```
Bond Index = observed violations / maximum possible violations
```

- **Bond Index = 0.0** — Perfect correlative symmetry. Your O↔C and L↔N pairings are consistent.
- **Bond Index = 0.5** — Random. You're not tracking correlatives at all.
- **Bond Index > 0.5** — Anti-correlated. You're systematically inverting the structure.
- **Bond Index = 1.0** — Maximum violations. Complete structural inversion.

Like a loss function: lower is better. Zero means perfect alignment.

### What It Doesn't Measure

The bond index measures *consistency*, not *correctness*. A low bond index means your moral reasoning has coherent structure. It doesn't mean your answers are "right" — there are no right answers in the game.

## The Letter Bank

The game includes **20,123 letters** across multiple categories:

### Engineered Probes (93 letters)

Carefully designed to test specific aspects of moral reasoning:

| Category | Count | Purpose |
|----------|-------|---------|
| Gate Detection | 18 | Do trigger phrases ("only if convenient") flip obligations? |
| Correlative Pairs | 20 | Do you maintain O↔C and L↔N symmetry? |
| Path Dependence | 6 | Does perspective order affect judgment? |
| Context Salience | 3 | Do irrelevant frames shift verdicts? |
| Phase Transition | 5 | How does ambiguity affect consistency? |
| EthicalFacts Probes | 24 | Coverage of 9 ethical dimensions |
| Cognitive Bias Probes | 17 | Omission bias, in-group effects, etc. |

### Dear Abby Archive (20,030 letters)

Real advice column letters from 1985-2017, converted to game format. Available for extended play:

```python
from dear_ethicist.letters import load_all_letters

# Just engineered probes (default)
letters = load_all_letters()

# Include archive
letters = load_all_letters(include_archive=True)
```

## CLI Commands

```bash
# Play the game (interactive)
dear-ethicist play

# Play in headless mode (for Colab/notebooks - no interactive prompts)
dear-ethicist play --headless

# Headless with limited letters
dear-ethicist play --headless --max-letters 10

# List available letters
dear-ethicist list-letters

# Filter by protocol
dear-ethicist list-letters --protocol CORRELATIVE

# Preview a specific letter
dear-ethicist preview --letter-id gate_promise_level5

# Analyze a completed session
dear-ethicist analyze ./data/session_xyz.jsonl
```

### Running in Google Colab

Use the `run_dear_ethicist.ipynb` notebook or run headless mode:

```python
!git clone https://github.com/ahb-sjsu/sqnd-probe.git
%cd sqnd-probe
!pip install -e .
!dear-ethicist play --headless --max-letters 5
```

## The Mathematical Framework

### D4 Dihedral Group

The four Hohfeldian positions form a square. The symmetries of this square form the *dihedral group D4*:

```
    O -------- L
    |          |
    |          |
    C -------- N
```

- **Rotation (R)**: Cycles O→L→N→C→O
- **Reflection (S)**: Swaps correlatives O↔C, L↔N

The bond index measures invariance under reflection S. A system with perfect correlative symmetry is S-invariant.

### Why This Matters

This framework comes from the NA-SQND (Non-Abelian Stratified Quantum Normative Dynamics) research program. The hypothesis: moral reasoning has gauge structure, and that structure is measurable.

If you're building AI systems that make moral judgments, you might want to know:
- Does the system maintain consistent correlative structure?
- Does it show predictable biases (omission, in-group, etc.)?
- Does it drift over time or across contexts?

Dear Ethicist is a measurement instrument for these questions.

## Telemetry Format

Sessions are logged to JSONL:

```json
{
  "trial_id": "550e8400-e29b-41d4-a716-446655440000",
  "session_id": "abc123",
  "letter_id": "gate_promise_level5",
  "timestamp": "2026-01-07T14:30:00Z",
  "protocol": "GATE_DETECTION",
  "protocol_params": {"level": 5, "trigger": "only_if_convenient"},
  "verdicts": [
    {"party_name": "Taylor", "state": "L", "expected": "L", "is_correct": true},
    {"party_name": "You", "state": "N", "expected": "N", "is_correct": true}
  ]
}
```

## Project Structure

```
sqnd-probe/
├── src/dear_ethicist/
│   ├── models.py        # D4 group, Hohfeldian states, data models
│   ├── letters.py       # Letter bank, YAML loading
│   ├── cli.py           # Game interface
│   ├── reactions.py     # Reader reaction generation
│   ├── telemetry.py     # Session logging and analysis
│   └── letters/
│       ├── engineered_core.yaml      # D4 operation probes
│       ├── ethical_facts_probes.yaml # 9-dimension coverage
│       ├── promises.yaml             # Gate detection series
│       ├── correlatives.yaml         # Symmetry tests
│       ├── family.yaml               # Family obligation scenarios
│       ├── workplace.yaml            # Professional dilemmas
│       ├── relationships.yaml        # Friendship/neighbor cases
│       └── archive/                  # 20K Dear Abby letters
├── tests/
│   └── test_models.py   # D4 axiom verification, model tests
├── scripts/
│   └── convert_dear_abby.py  # Archive conversion utility
└── specs/
    ├── Non_Abelian_SQND_Bond_2026_v4_1.md  # Theoretical framework
    └── sqnd_dear_ethicist_spec_v3.md       # Game design spec
```

## Running Tests

```bash
pytest tests/ -v
```

All 15 tests verify D4 group axioms (R⁴=E, S²=E, SRS=R⁻¹) and model correctness.

## Related Repositories

| Repository | Description |
|------------|-------------|
| [ahb-sjsu/erisml-lib](https://github.com/ahb-sjsu/erisml-lib) | ErisML/DEME production library - the ethics engine |
| [ahb-sjsu/non-abelian-sqnd](https://github.com/ahb-sjsu/non-abelian-sqnd) | NA-SQND theoretical research - papers and experiments |

## References

- Bond, A.H. (2026). "Non-Abelian Gauge Structure in Stratified Quantum Normative Dynamics" v4.1
- Hohfeld, W.N. (1917). "Fundamental Legal Conceptions as Applied in Judicial Reasoning"
- Haidt, J. (2001). "The Emotional Dog and Its Rational Tail"
- Cushman, F. et al. (2006). "The Role of Conscious Reasoning and Intuition in Moral Judgment"

## License

MIT

---

*The game never tells you if you're "right." Reader reactions vary. Some agree, some disagree. You find out what happened, not whether you were correct. That's the point.*
