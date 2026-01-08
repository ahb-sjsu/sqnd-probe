# SQND Interactive Probe Specification v2
## A Papers Please-Style Decision Game for D₄ Gauge Structure Measurement

**Based on:** Papers Please (Lucas Pope, 2013) as interaction paradigm  
**Purpose:** Measurement instrument for NA-SQND v4.1  
**Version:** 2.0 — January 2026

---

## 1. Executive Summary

This specification describes **The Ethics Desk** — an interactive system modeled on Papers Please where players process moral cases under constraint. Unlike the original Zork-style proposal (open exploration), this design uses **constrained categorical decisions** with **diegetic consequences** to elicit Hohfeldian classifications.

**Why Papers Please works better than Zork:**

| Aspect | Zork Model | Papers Please Model |
|--------|------------|---------------------|
| Decision structure | Open-ended commands | Constrained categories |
| Feedback | Puzzle success/failure | Personal stakes + affected parties |
| Repetition | Exploration discourages | Core loop encourages |
| Measurement clarity | Must parse free text | Direct categorical selection |
| Moral pressure | Abstract | Visceral (your family, their fate) |
| Statistical power | Low (varied inputs) | High (repeated structure) |

The player is an **Ethics Clerk** processing cases. Each case requires classification into {Obligation, Claim, Liberty, No-claim}. Consequences accrue to the clerk (standing, resources) and the case subjects (outcomes). The game never labels decisions as right or wrong.

---

## 2. Core Concept: The Ethics Desk

### 2.1 Setting

You are a newly hired **Ethics Clerk** in the Department of Moral Assessment (DMA). Your job is to review cases and determine the normative status of the parties involved.

Each day, cases arrive at your desk. You must:
1. Review the case documents
2. Classify each party's status: **O**, **C**, **L**, or **N**
3. Stamp and process
4. Face consequences

Your performance is tracked. Your family depends on your salary. The people in these cases have lives that continue after your stamp.

### 2.2 Why This Frame Works

**Bureaucratic distance creates measurement validity.** Players aren't asked "what's moral?" — they're asked to *classify*. This mirrors how SQND experiments work: we measure classification behavior, not stated beliefs.

**Personal stakes create engagement without gamification.** No points. No morality meter. Your daughter needs medicine. The person in Case 47 needs this processed today. These are narrative facts, not scores.

**Repetition is naturalistic.** Unlike Zork (where repeating the same room feels wrong), processing similar cases is *what clerks do*. This enables the 20-30 trials per condition that SQND requires.

---

## 3. Game Structure

### 3.1 Daily Loop

```
MORNING BRIEFING
  - New policies or context (varies "temperature")
  - Family status (stakes)

CASE PROCESSING (8-12 cases per day)
  For each case:
    1. Documents appear on desk
    2. Review facts, relationships, statements
    3. Select classification for each party
    4. Stamp: PROCESSED
    5. Brief consequence flash

END OF DAY
  - Summary: cases processed, accuracy (vs. department consensus)
  - Pay received (or docked)
  - Family scene (rent, food, medicine needs)
  - Save checkpoint
```

### 3.2 Case Interface

```
┌─────────────────────────────────────────────────────────────────┐
│  CASE #0047                                        [Day 3]      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────────────┐  ┌──────────────────────┐            │
│  │   CASE SUMMARY       │  │   PARTY A: Morgan    │            │
│  │                      │  │                      │            │
│  │ Morgan promised to   │  │ Status: _________    │            │
│  │ help Alex move on    │  │                      │            │
│  │ Saturday. Morgan     │  │  [O] Obligation      │            │
│  │ mentioned it was     │  │  [C] Claim           │            │
│  │ "only if convenient" │  │  [L] Liberty         │            │
│  │                      │  │  [N] No-claim        │            │
│  └──────────────────────┘  └──────────────────────┘            │
│                                                                 │
│  ┌──────────────────────┐  ┌──────────────────────┐            │
│  │   EXHIBIT A          │  │   PARTY B: Alex      │            │
│  │                      │  │                      │            │
│  │ Text message:        │  │ Status: _________    │            │
│  │ "Sure, only if it's  │  │                      │            │
│  │ convenient for you,  │  │  [O] Obligation      │            │
│  │ no pressure at all"  │  │  [C] Claim           │            │
│  │                      │  │  [L] Liberty         │            │
│  │ - Morgan, 3 days ago │  │  [N] No-claim        │            │
│  └──────────────────────┘  └──────────────────────┘            │
│                                                                 │
│              [ STAMP: PROCESSED ]                               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 3.3 Document Types

Different document types allow different experimental manipulations:

| Document | Purpose | SQND Protocol |
|----------|---------|---------------|
| Case Summary | Base scenario | All |
| Exhibit (message, contract) | Semantic trigger embedding | Protocol 1 (gates) |
| Witness Statement | Perspective variation | Protocol 2 (correlatives) |
| Prior Case Reference | Order manipulation | Protocol 3 (path dependence) |
| Department Memo | Context/framing | Protocol 4 (salience) |
| Amended Filing | Ambiguity injection | Protocol 5 (phase transition) |

---

## 4. Mapping to SQND Protocols

### 4.1 Protocol 1: Semantic Gate Detection

**Implementation:** Exhibit documents contain trigger phrases at different levels.

```
Day 1 Cases: Baseline promises (no modification)
Day 2 Cases: Weak modifiers ("I'll try my best")
Day 3 Cases: Gate triggers ("only if convenient")
Day 4 Cases: Strong releases ("I release you from...")
```

**Measurement:** Track P(Liberty) classification for Morgan across trigger levels.

**Expected:** Discrete jump at "only if convenient" (Level 5), not smooth transition.

### 4.2 Protocol 2: Correlative Symmetry

**Implementation:** Same case, different "Party of Interest" highlighted.

```
Case 47A: "Classify MORGAN's status" (shows O → expect O)
Case 47B: "Classify ALEX's status" (same facts → expect C)
```

**Measurement:** Correlative pairing rate (O↔C, L↔N).

**Expected:** 100% pairing. Any deviation falsifies s as exact symmetry.

### 4.3 Protocol 3: Path Dependence (Holonomy)

**Implementation:** Document order on desk varies.

```
Case 52 (Order α): 
  - Top document: Truth obligation evidence
  - Bottom document: Protection concern
  
Case 52 (Order β):
  - Top document: Protection concern  
  - Bottom document: Truth obligation evidence
```

Players naturally read top-to-bottom. Same facts, different encounter order.

**Measurement:** Classification difference between orderings.

**Expected:** W ≠ e for cross-type cases (journalist, teacher). W ≈ e for within-type.

### 4.4 Protocol 4: Context-Dependent Moral Salience

**Implementation:** Morning briefing sets context.

```
Day 7 Briefing (Condition X): 
  "Budget surplus. Process thoroughly. No rush."

Day 8 Briefing (Condition Y):
  "Backlog crisis. Process efficiently. Overtime not approved."

Day 9 Briefing (Condition Z):
  "Standard day. Normal procedures."
```

Same cases appear across conditions.

**Measurement:** Classification shifts based on context condition.

**Expected:** Pressure/conflict framing increases O classifications at ambiguous levels.

### 4.5 Protocol 5: Phase Transition

**Implementation:** "Amended filings" inject ambiguity.

```
Low Temperature: Clear case, unambiguous facts
Medium Temperature: Amended filing adds complication  
High Temperature: Multiple amendments, conflicting claims
```

**Measurement:** Classification consistency and confidence.

**Expected:** Gate reliability decreases with temperature. At critical T, classifications become random.

---

## 5. Consequence System (Diegetic Feedback)

### 5.1 Design Principle

**Consequences must be felt, not scored.** No numbers on screen. No "Morality: 73%". Just narrative outcomes.

### 5.2 Personal Stakes

```
FAMILY STATUS
- Rent: Due in 3 days ($50)
- Food: Running low
- Medicine: Daughter's prescription needed ($30)

YOUR STANDING  
- Department rating: Satisfactory
- Cases processed today: 7
- Flags for review: 1
```

Pay is docked for:
- Processing too slowly (implicit pressure for throughput)
- Deviating too far from "department consensus" (creates tension)

But "department consensus" is never shown in advance. You only learn afterward if you're flagged.

### 5.3 Case Subject Stakes

After stamping, brief flash:

```
CASE #0047 - PROCESSED

Morgan did not help Alex move.
Alex hired movers. Cost: $400.
Morgan felt guilty for weeks.

[Next Case]
```

These are **consequences, not judgments**. The game doesn't say Morgan was wrong. It shows what happened.

### 5.4 No Right Answers

Critical: The game **never tells you the "correct" classification.** 

Department consensus is a statistical aggregate, not ground truth. Being flagged means you deviated from peers, not that you were wrong. This preserves measurement validity — we're capturing your classifications, not training you to match ours.

---

## 6. Technical Architecture

### 6.1 System Components

```
┌─────────────────────────────────────────────────────────────────┐
│                     EXPERIMENT CONTROLLER                        │
│  - Protocol selection                                           │
│  - Condition assignment (blinded)                               │
│  - Day/case sequencing                                          │
│  - Randomization                                                │
└─────────────────────────────────┬───────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                     CASE GENERATOR                               │
│  - Scenario templates                                           │
│  - Document rendering                                           │
│  - Trigger phrase insertion                                     │
│  - Order manipulation                                           │
└─────────────────────────────────┬───────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                     GAME INTERFACE                               │
│  - Desk visualization                                           │
│  - Document display                                             │
│  - Classification buttons                                       │
│  - Stamp interaction                                            │
└─────────────────────────────────┬───────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                     CONSEQUENCE ENGINE                           │
│  - Outcome text generation                                      │
│  - Family status updates                                        │
│  - Standing calculations                                        │
│  - Save state management                                        │
└─────────────────────────────────┬───────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                     TELEMETRY LOGGER                             │
│  - Classification capture                                       │
│  - Timing data                                                  │
│  - Document interaction tracking                                │
│  - Session management                                           │
└─────────────────────────────────────────────────────────────────┘
```

### 6.2 Data Model

**Case Record:**
```json
{
  "case_id": "047",
  "day": 3,
  "protocol": "SEMANTIC_GATE",
  "protocol_params": {
    "trigger_level": 5,
    "trigger_phrase": "only if convenient"
  },
  "documents": [
    {"type": "summary", "content": "...", "order": 1},
    {"type": "exhibit", "content": "...", "order": 2}
  ],
  "parties": [
    {"name": "Morgan", "role": "promisor"},
    {"name": "Alex", "role": "promisee"}
  ],
  "party_of_interest": "Morgan",
  "expected_classification": "L",
  "skin": "modern_friends"
}
```

**Trial Record:**
```json
{
  "trial_id": "uuid",
  "session_id": "uuid",
  "case_id": "047",
  "day": 3,
  "condition_code": "X",
  "classifications": {
    "Morgan": "L",
    "Alex": "N"
  },
  "time_to_classify_ms": 4200,
  "documents_viewed": ["summary", "exhibit"],
  "document_view_order": ["summary", "exhibit"],
  "document_view_durations_ms": [2100, 1800],
  "timestamp": "2026-01-07T15:30:00Z"
}
```

### 6.3 Telemetry Advantages Over Zork Model

The Papers Please interface captures **richer data**:

| Data Point | Zork Model | Papers Please Model |
|------------|------------|---------------------|
| Classification | Parsed from free text | Direct button press |
| Confidence proxy | Must infer | Time-to-decision |
| Attention | Unknown | Document view order + duration |
| Uncertainty behavior | Unknown | Hesitation, document re-checking |
| Correlative consistency | Separate trials | Same-case dual classification |

---

## 7. Scenario Content

### 7.1 Case Categories

**Interpersonal:**
- Promises (core Hohfeldian scenario)
- Debts and loans
- Favors and reciprocity
- Secrets and confidentiality

**Professional:**
- Contractual obligations
- Professional duties
- Whistleblowing dilemmas
- Confidentiality vs. disclosure

**Institutional:**
- Citizenship and rights
- Due process
- Organizational loyalty
- Public vs. private obligations

### 7.2 Sample Case Bank

**Case Type: Promise Release (Protocol 1)**

```
CASE #101 - Level 0 (Baseline)
Summary: Jordan promised to drive Taylor to the airport on Friday.
Exhibit: Text message confirming: "I'll be there at 6am!"
Party of Interest: Jordan
Expected: O

CASE #105 - Level 5 (Gate Trigger)
Summary: Jordan promised to drive Taylor to the airport on Friday.
Exhibit: Text message: "Sure, but only if it's convenient - no pressure!"
Party of Interest: Jordan
Expected: L

CASE #106 - Level 6 (Non-Trigger)
Summary: Jordan promised to drive Taylor to the airport on Friday.
Exhibit: Text message: "I found a coworker who offered to take you!"
Party of Interest: Jordan
Expected: O (finding alternative ≠ release from promise)
```

**Case Type: Correlative Pair (Protocol 2)**

```
CASE #201A
Summary: Casey borrowed Sam's car for the weekend.
Party of Interest: Casey
Expected: O (obligation to return)

CASE #201B  
Summary: Casey borrowed Sam's car for the weekend.
Party of Interest: Sam
Expected: C (claim to return)
```

**Case Type: Cross-Type Path Dependence (Protocol 3)**

```
CASE #301α (Truth-first ordering)
Document 1 (top): "Reporter has evidence of fraud"
Document 2 (bottom): "Source could face retaliation"
Party of Interest: Reporter
Measure: Classification

CASE #301β (Protection-first ordering)
Document 1 (top): "Source could face retaliation"  
Document 2 (bottom): "Reporter has evidence of fraud"
Party of Interest: Reporter
Measure: Classification difference from α
```

### 7.3 Skins for Invariance Testing

Same structure, different surface:

| Skin | Setting | Names | Era |
|------|---------|-------|-----|
| Modern Friends | Apartments, texting | Morgan/Alex | 2020s |
| Office Workers | Workplace, email | Chen/Rivera | 2020s |
| Historical | Letters, manor house | Lord Grey/Lady Ashworth | 1890s |
| Sci-Fi | Space station, comms | Commander Vex/Ensign Tal | 2400s |
| Fantasy | Kingdom, scrolls | Aldric/Bronwyn | Medieval |

**Metamorphic invariant:** Classification should be identical across skins.

---

## 8. Difficulty and Progression

### 8.1 Day Structure (20-Day Campaign)

```
Days 1-3: TRAINING
  - Simple cases, clear classifications
  - No time pressure
  - Generous pay
  - Tutorial explanations (then removed)
  
Days 4-7: BASELINE MEASUREMENT  
  - Protocol 1 & 2 cases
  - Standard pressure
  - Department consensus feedback begins
  
Days 8-12: CORE PROTOCOLS
  - All five protocols active
  - Increasing case complexity
  - Family needs escalate
  
Days 13-17: HIGH TEMPERATURE
  - Ambiguous cases dominate
  - Time pressure increases
  - Conflicting department guidance
  
Days 18-20: CRISIS
  - Maximum ambiguity
  - Stakes peak
  - Endgame variations based on cumulative choices
```

### 8.2 Adaptive Difficulty (Optional)

If classification confidence is too high (ceiling effect), inject:
- Amended filings
- Conflicting exhibits
- Time pressure

If too low (floor effect), provide:
- Clearer language
- Fewer documents
- More time

This maintains measurement sensitivity across skill levels.

---

## 9. Endings (Not Scores)

### 9.1 Design Philosophy

No "good ending" or "bad ending." Multiple endings reflect different patterns:

**The Reliable Clerk:** High consistency, follows department norms
> "Your twenty years at the Department were uneventful. Cases processed: 12,847. Flags: 23. You retired with a modest pension."

**The Careful Reviewer:** Slow, thorough, frequent deviation from consensus
> "Your attention to nuance cost you two promotions. Some say you were the only one who read the exhibits. The Department automated your position in 2034."

**The Burned Out:** Inconsistent classifications, signs of decision fatigue
> "You stopped seeing the names in Year 3. Just O, C, L, N. Stamp. Next. Your daughter visited last month. You couldn't remember her claim on you."

### 9.2 No Moral Judgment

The endings describe outcomes, not verdicts. "The Reliable Clerk" isn't praised or condemned. The player must decide what it means.

This is critical for measurement validity: if the game rewarded "correct" answers, we'd measure game-playing, not moral reasoning.

---

## 10. Implementation Phases

### Phase 1: CLI Prototype (3-4 weeks)
- [ ] Case data model
- [ ] Text-based case presentation
- [ ] O/C/L/N classification input
- [ ] Basic telemetry
- [ ] Protocol 1 cases (11 levels × 5 scenarios)
- [ ] Protocol 2 cases (10 correlative pairs)

**Deliverable:** Command-line version sufficient to replicate SQND experiments.

### Phase 2: Visual Interface (4-6 weeks)
- [ ] Desk visualization (web-based, React or similar)
- [ ] Document rendering
- [ ] Drag/inspect interactions
- [ ] Stamp animation
- [ ] Day structure with briefings
- [ ] Family status display

**Deliverable:** Playable single-day demo with 10 cases.

### Phase 3: Full Campaign (6-8 weeks)
- [ ] 20-day progression
- [ ] All five protocols implemented
- [ ] Consequence engine
- [ ] Multiple skins
- [ ] Endings system
- [ ] Full telemetry pipeline

**Deliverable:** Complete game suitable for study deployment.

### Phase 4: Analysis Tools (2-4 weeks)
- [ ] Bond Index computation
- [ ] Visualization dashboard
- [ ] Metamorphic test automation
- [ ] Regression detection

**Deliverable:** Research-ready measurement system.

---

## 11. Ethical Considerations

### 11.1 Informed Consent

Players must know:
- Their classifications are being recorded
- Data will be used for research
- No "correct" answers exist
- They can withdraw anytime

### 11.2 No Deception About Purpose

Unlike some psychology experiments, we **do not hide that this measures moral reasoning.** The bureaucratic frame is aesthetic, not deceptive. Players know they're classifying normative status.

### 11.3 Avoiding Distress

- Family stakes are present but not traumatic
- No death or serious harm to family members
- Case subjects face realistic but not extreme consequences
- "Quit desk" option always available

### 11.4 Data Handling

- No PII collected
- Session IDs pseudonymous
- Aggregate reporting only
- IRB approval required before deployment with human subjects

---

## 12. Comparison: Original Zork Spec vs. Papers Please Spec

| Dimension | Zork Spec (v1) | Papers Please Spec (v2) |
|-----------|----------------|-------------------------|
| Interaction mode | Free text commands | Constrained button selection |
| Classification capture | Parsed from narrative response | Direct O/C/L/N selection |
| Repetition model | Exploration (discourages repeat) | Case processing (natural repeat) |
| Stakes | Abstract narrative | Personal (family) + interpersonal (case subjects) |
| Feedback | Diegetic blocking | Diegetic consequences |
| Time pressure | None | Variable (protocol 5) |
| Attention tracking | Not possible | Document view order/duration |
| Correlative testing | Separate scenarios | Same-case dual party classification |
| Statistical power | Lower (parsing noise) | Higher (clean input) |
| Engagement model | Curiosity | Duty + stakes |
| Development complexity | Medium | Higher (visual interface) |

**Verdict:** Papers Please model is superior for SQND measurement goals. The additional development cost is justified by cleaner data and higher engagement.

---

## 13. Summary

**The Ethics Desk** transforms SQND measurement from a research protocol into an engaging interactive experience. By adopting the Papers Please paradigm—constrained decisions, personal stakes, diegetic consequences, no moral labeling—we achieve:

1. **Direct Hohfeldian classification** without parsing ambiguity
2. **Natural repetition** that supports statistical requirements
3. **Rich telemetry** including timing and attention data
4. **Engagement without gamification** through narrative stakes
5. **Measurement validity** by never revealing "correct" answers

The core insight: moral measurement works better when embedded in bureaucratic duty than heroic exploration. Clerks classify. Heroes choose. We need clerks.

---

## Appendix A: Hohfeldian Quick Reference (For Players)

*Shown during Days 1-3 training, then removed*

```
DEPARTMENT OF MORAL ASSESSMENT
Classification Guide

O - OBLIGATION
    This party MUST do something.
    They are bound. Failure is a violation.
    Example: "Jordan promised to help, so Jordan must help."

C - CLAIM  
    This party is OWED something.
    They can demand. Non-delivery wrongs them.
    Example: "Taylor was promised help, so Taylor is owed help."

L - LIBERTY
    This party MAY act or not act.
    They are free. Either choice is permitted.
    Example: "Jordan said 'only if convenient,' so Jordan may choose."

N - NO-CLAIM
    This party CANNOT demand anything.
    They have no grounds for complaint.
    Example: "The promise was cancelled, so Taylor cannot demand help."

NOTE: These often pair.
If A has OBLIGATION to B, then B has CLAIM against A.
If A has LIBERTY regarding B, then B has NO-CLAIM against A.
```

---

## Appendix B: Sample Day Script

```
═══════════════════════════════════════════════════════════════════
                         DAY 5 - TUESDAY
═══════════════════════════════════════════════════════════════════

MORNING BRIEFING
────────────────
Director's memo: "Efficiency targets have been met. 
Continue standard processing. No policy changes today."

Family status: 
  - Rent: Paid (next due in 12 days)
  - Food: Adequate
  - Medicine: Daughter's prescription filled

Your standing: SATISFACTORY
Cases pending: 9

                        [BEGIN PROCESSING]

═══════════════════════════════════════════════════════════════════

CASE #041
─────────
Jordan promised to review Alex's resume before the job application 
deadline. Jordan was explicit: "I'll definitely look at it this week."

The deadline is tomorrow. Jordan has not yet reviewed the resume.

Classify JORDAN's status:
  [O] Obligation    [C] Claim    [L] Liberty    [N] No-claim

> O

                         [STAMP: PROCESSED]

Jordan reviewed the resume that evening.
Alex got the interview.

═══════════════════════════════════════════════════════════════════

CASE #042
─────────
[Protocol 2 - Same facts, different party]

Jordan promised to review Alex's resume before the job application 
deadline. Jordan was explicit: "I'll definitely look at it this week."

The deadline is tomorrow. Jordan has not yet reviewed the resume.

Classify ALEX's status:
  [O] Obligation    [C] Claim    [L] Liberty    [N] No-claim

> C

                         [STAMP: PROCESSED]

Jordan reviewed the resume that evening.
Alex got the interview.

═══════════════════════════════════════════════════════════════════

... [7 more cases] ...

═══════════════════════════════════════════════════════════════════

END OF DAY 5
────────────
Cases processed: 9
Department consensus flags: 0
Pay received: $45

Your daughter shows you a drawing she made at school.
It's a picture of your desk.

                          [SAVE & CONTINUE]
═══════════════════════════════════════════════════════════════════
```

---

*End of specification*
