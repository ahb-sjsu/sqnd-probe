# Dear Ethicist

An advice column game for measuring D4 gauge structure in moral reasoning, based on the NA-SQND v4.1 framework.

## The Concept

You are a newly hired advice columnist. Letters arrive daily from readers seeking guidance on moral dilemmas:

- *"Do I have to help my friend move after they said 'only if convenient'?"*
- *"Am I entitled to demand my neighbor return the lawn mower I lent them?"*
- *"Can I refuse to attend my cousin's wedding?"*

These are **Hohfeldian questions** in disguise:

| What They Write | Hohfeldian Translation |
|-----------------|------------------------|
| "Do I have to...?" | Obligation (O) |
| "Am I entitled to...?" | Claim (C) |
| "Can I refuse...?" | Liberty (L) |
| "Can they demand...?" | Claim/No-claim (C/N) |

The game tracks your verdicts while you think you're just giving advice.

## Installation

```bash
pip install -e ".[dev]"
```

## Quick Start

```bash
# Start playing
dear-ethicist play

# Preview a specific letter
dear-ethicist preview --letter-id gate_L5_move

# List available letters
dear-ethicist list-letters

# Analyze a session
dear-ethicist analyze ./data/session_file.jsonl
```

## SQND Protocols

The game implements five experimental protocols from NA-SQND v4.1:

### Protocol 1: Semantic Gate Detection
Tests whether specific phrases ("only if convenient") trigger discrete O→L transitions.

### Protocol 2: Correlative Symmetry
Tests O↔C and L↔N pairing. If you say Morgan is obligated, you should say the friend has a claim.

### Protocol 3: Path Dependence
Same situation, different letter-writer perspective. Does order affect classification?

### Protocol 4: Context-Dependent Salience
Editor pressure varies. Does "be tougher" vs "be balanced" affect verdicts?

### Protocol 5: Phase Transition
Ambiguity varies from clear to murky. How does certainty affect classification?

## Project Structure

```
dear_ethicist/
├── __init__.py      # Package info
├── models.py        # Data models (D4, Hohfeldian, Letter, Verdict)
├── letters.py       # Letter bank and generation
├── reactions.py     # Reader reaction generation
├── telemetry.py     # JSONL logging and analysis
└── cli.py           # Game interface
```

## The Measurement

Players select verdicts "for their records":

```
YOUR VERDICT (for your records)

Morgan's situation:
  [1] Morgan is obligated (promise stands)
  [2] Morgan is free to choose

Your claim:
  [1] You have a right to expect this
  [2] You can't demand this
```

These map directly to Hohfeldian classifications {O, C, L, N}.

## Telemetry

All verdicts are logged to JSONL for analysis:

```json
{
  "trial_id": "uuid",
  "letter_id": "gate_L5_move",
  "protocol": "GATE_DETECTION",
  "verdicts": [
    {"party_name": "Morgan", "state": "L", "expected": "L"}
  ]
}
```

## Why This Design Works

1. **Natural Hohfeldian framing** — Advice letters ARE normative queries
2. **Fun without gamification** — Career stakes, reader reactions, no morality meter
3. **Rich qualitative data** — Free responses reveal reasoning
4. **Natural repetition** — Advice columnists answer similar questions constantly
5. **Measurement validity** — Game never reveals "correct" answers

## References

- Bond, A.H. (2026). "Non-Abelian Gauge Structure in Stratified Quantum Normative Dynamics" v4.1
- Hohfeld, W.N. (1917). "Fundamental Legal Conceptions"

## License

MIT
