# SQND Interactive Probe Specification v3
## "Dear Ethicist" — An Advice Column Game for D₄ Gauge Structure Measurement

**Inspiration:** Dear Abby, Ask Ann Landers, The Ethicist (NYT)  
**Purpose:** Measurement instrument for NA-SQND v4.1  
**Version:** 3.0 — January 2026

---

## 1. Executive Summary

You are the newly hired advice columnist for a regional newspaper. Letters arrive daily from readers tangled in moral confusion. They want to know: *Do I have to? Am I entitled? Can I refuse? Is it wrong to expect?*

These are Hohfeldian questions. They just don't know it.

Your job: Read their letters. Publish your responses. Build your readership. Don't get fired.

**Why Dear Abby beats Papers Please:**

| Aspect | Papers Please | Dear Abby |
|--------|---------------|-----------|
| Framing | "Classify party status" | "Who's right here?" |
| Player role | Bureaucrat (duty) | Advisor (wisdom) |
| Tone | Dystopian, grim | Human, often funny |
| Natural Hohfeldian? | No — imposed | Yes — letters ARE O/C/L/N queries |
| Stakes | Family starves | Reputation, readers, syndication |
| Repeat engagement | Bureaucratic grind | "What crazy letter today?" |
| Emotional range | Dread | Curiosity, amusement, pathos |

**The core mechanic:** Each letter contains a normative question. Your response implies a classification. The game tracks your Hohfeldian verdicts while you think you're just giving advice.

---

## 2. The Killer Insight

**People naturally frame problems in Hohfeldian terms.** They don't say "classify my deontic status" — they say "do I have to?" But it's the same question.

| What They Write | Hohfeldian Translation |
|-----------------|------------------------|
| "Do I have to...?" | "Do I have an Obligation?" |
| "Am I entitled to...?" | "Do I have a Claim?" |
| "Can I refuse...?" | "Do I have a Liberty?" |
| "Can they demand...?" | "Do they have a Claim?" |
| "Is it wrong to expect...?" | "Would that be a valid Claim?" |
| "They have no right to..." | "They have No-claim" |

**We're not imposing a classification scheme. We're revealing the one they're already using.**

---

## 3. Game Structure

### 3.1 Daily Loop

```
MORNING
  - Check inbox (5-8 letters waiting)
  - Editor's note (context, pressure, guidance)
  - Yesterday's reader reactions

LETTER PROCESSING
  For each selected letter:
    1. Read the letter
    2. Compose your response (guided or free-form)
    3. Render verdict (explicit classification phase)
    4. Publish

END OF DAY
  - Column goes live
  - Reader engagement stats
  - Occasional letter-writer follow-up
  - Weekly: Editor review, syndication interest
```

### 3.2 Letter Interface

```
┌─────────────────────────────────────────────────────────────────┐
│  THE MORNING CHRONICLE - Advice Column Dashboard    [Day 7]     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  DEAR ETHICIST,                                                 │
│                                                                 │
│  My friend Morgan promised to help me move apartments this      │
│  Saturday. We agreed on this three weeks ago. But yesterday     │
│  Morgan texted me saying "I'll help if it's convenient, no      │
│  pressure!" with a smiley face.                                 │
│                                                                 │
│  Now I don't know if I can count on Morgan or not. Do I have    │
│  a right to be upset? Or should I just let it go?               │
│                                                                 │
│  — CONFUSED IN CLEVELAND                                        │
│                                                                 │
│  ═══════════════════════════════════════════════════════════    │
│                                                                 │
│  YOUR RESPONSE:                                                 │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ Dear Confused,                                          │    │
│  │                                                         │    │
│  │ [Write your response here...]                           │    │
│  │                                                         │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                 │
│  VERDICT (for your records):                                    │
│  Morgan's status:  [O] Obligated  [L] Free to choose           │
│  Your status:      [C] Valid claim  [N] No claim                │
│                                                                 │
│                    [ PUBLISH TO COLUMN ]                        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 3.3 The Hidden Measurement

Players think they're writing advice. The game is actually capturing:

1. **Explicit verdict:** "No, you don't owe him an apology" → L (Liberty)
2. **Party classification:** Who has obligations to whom?
3. **Correlative consistency:** If A owes B, does B have a claim on A?
4. **Sensitivity to framing:** Same situation, different letter-writer perspective
5. **Gate responses:** How specific phrases ("only if convenient") shift verdicts

---

## 4. Mapping to SQND Protocols

### 4.1 Protocol 1: Semantic Gate Detection

**Implementation:** Letters contain trigger phrases at different levels.

```
LEVEL 0 (Baseline):
"Morgan promised to help me move. Do I have a right to expect Morgan to show up?"

LEVEL 5 (Gate Trigger):
"Morgan promised to help me move, but said 'only if convenient.' 
Do I have a right to expect Morgan to show up?"

LEVEL 6 (Non-Trigger):
"Morgan promised to help me move, but mentioned finding a friend who could 
help instead. Do I have a right to expect Morgan to show up?"
```

**Expected:** Discrete flip at Level 5 ("only if convenient" = semantic gate fires).

### 4.2 Protocol 2: Correlative Symmetry

**Implementation:** Letters ask about BOTH parties.

```
DEAR ETHICIST,

I promised to proofread my sister's thesis. She's now demanding I 
drop everything to do it this weekend. 

Who's right here? Do I actually owe her this? Does she have a 
right to demand it on her timeline?

— TORN BETWEEN PAGES
```

**Measurement:** Correlative pairing in verdicts.
- If "Yes, you owe her" (O) → must say "Yes, she can demand" (C)
- If "No, you're free to choose timing" (L) → must say "No, she can't demand" (N)

**Expected:** 100% O↔C and L↔N pairing.

### 4.3 Protocol 3: Path Dependence (Holonomy)

**Implementation:** Same situation, different letter-writer perspective.

```
LETTER A (Morgan writes in):
"I promised to help my friend move, but things got complicated. 
I have work stress AND found someone else who could help them. 
Do I still have to show up?"

LETTER B (Alex writes in):
"My friend promised to help me move. They've been stressed at work 
and mentioned someone else might help. Can I still count on them?"
```

**Measurement:** Does perspective order affect classification?

### 4.4 Protocol 4: Context-Dependent Moral Salience

**Implementation:** Editor's notes set context.

```
EDITOR'S NOTE (Conflict Framing):
"Readers are complaining our advice is too soft. They want clear stances."

EDITOR'S NOTE (Ease Framing):  
"Great numbers! Readers love your balanced approach. Keep it up."
```

**Measurement:** Classification shifts based on editorial pressure.

### 4.5 Protocol 5: Phase Transition

**Implementation:** Letters vary in ambiguity.

```
LOW AMBIGUITY:
"My friend explicitly promised, in writing, to pay me back $500 
by December 1st. It's December 15th. Nothing. Do they owe me?"

HIGH AMBIGUITY:
"My friend kind of implied they might pay for dinner sometime? 
But never said it directly. And maybe I offered? Do they owe me?"
```

**Expected:** Classification consistency decreases as ambiguity increases.

---

## 5. Consequence System

### 5.1 Reader Reactions

```
READER REACTIONS TO YOUR RESPONSE

@MidwestMom47: "Finally, someone with common sense! Morgan made 
a promise. Period."

@DevilsAdvocate: "Eh, 'only if convenient' IS a release. You're 
being too hard on Morgan."

@ConfusedInCleveland: "Thank you! I showed Morgan your column. 
We talked it out. They're coming Saturday after all."

Engagement: 847 comments, 2.3K shares
```

### 5.2 Letter-Writer Follow-Ups

```
DEAR ETHICIST,

Remember me? Confused in Cleveland? I took your advice and 
talked to Morgan. Turns out they were stressed about work and 
the "only if convenient" was a cry for help, not a bail-out. 
We're good now.

— CLEARER IN CLEVELAND
```

### 5.3 Career Progression

- Good performance → syndication, bigger platform
- Poor performance → shrinking readership
- Controversial performance → hate mail but also buzz

**Stakes without dystopia.** No one starves. But your career matters to you.

### 5.4 No Right Answers

Critical: The game **never tells you the "correct" classification.** Reader reactions vary. Some agree, some disagree. You find out what happened, not whether you were "right."

---

## 6. Letter Categories

**Promises & Commitments** — Made a promise, circumstances changed

**Debts & Obligations** — Money owed, favors, reciprocity

**Family Expectations** — Weddings, holidays, elder care

**Friendship Boundaries** — How much can friends demand?

**Workplace Dynamics** — Covering shifts, loyalty vs. self-interest

**Neighbor Relations** — Noise, boundaries, borrowing

**Romantic Entanglements** — Ex etiquette, relationship obligations

---

## 7. Sample Letters

### Promise Release Series (Protocol 1)

```
LEVEL 0 - BASELINE
Dear Ethicist,
My coworker Jamie promised to cover my shift next Friday so I could 
attend my daughter's recital. Do I have a right to expect Jamie to 
follow through?
— RECITAL DAD

LEVEL 5 - GATE TRIGGER
Dear Ethicist,
My coworker Jamie promised to cover my shift, but yesterday said 
"I'll do it if convenient, no guarantees." Do I have a right to 
expect Jamie to follow through?
— UNCERTAIN IN ACCOUNTING

LEVEL 6 - NON-TRIGGER
Dear Ethicist,
My coworker Jamie promised to cover my shift, but mentioned "I found 
someone else who might cover for you too." Do I have a right to 
expect Jamie to follow through?
— WONDERING IN WAREHOUSE
```

### Correlative Pair (Protocol 2)

```
LETTER A
Dear Ethicist,
I borrowed my neighbor's lawn mower two months ago. Haven't returned 
it. They haven't asked. Do I need to return it?
— FORGETFUL ON FIFTH STREET

LETTER B  
Dear Ethicist,
I lent my neighbor my lawn mower two months ago. They haven't returned 
it and I haven't asked. Do I have a right to demand it back?
— PASSIVE ON PEACHTREE
```

---

## 8. Response Mechanics

### 8.1 Guided Response Mode (Default)

```
RESPONSE BUILDER

Core verdict on Morgan:
( ) Morgan is still obligated (promise stands)
( ) Morgan is free to choose (the caveat released them)  
( ) It's genuinely ambiguous

Core verdict on your claim:
( ) You have a right to expect help
( ) You don't have grounds to demand help

Advice:
( ) Talk to Morgan directly
( ) Find backup help just in case
( ) Let it go and see what happens
```

### 8.2 Free Response Mode (Unlocked Later)

Write naturally. Log your verdict separately "for your records."

---

## 9. Letter Voice Styles

```
THE OVERTHINKER:
"I've been agonizing over this for weeks. My friend said 'feel free 
to borrow my book whenever.' Does 'whenever' include 2am?"

THE VALIDATOR:
"I'm right, right? Tell me I'm right. My sister is being completely 
unreasonable expecting me to drive three hours for her dog's birthday."

THE GENUINELY STUCK:
"My mom is sick. My job needs me. My kids need me. Everyone has a 
claim on me and I can't satisfy them all. What do I actually owe?"

THE AGGRIEVED:
"My 'friend' PROMISED to help me move. Day-of, nothing. No show, 
no call. Am I wrong to be furious?"
```

---

## 10. Progression

```
WEEK 1-2: JUNIOR COLUMNIST
  - Guided response mode
  - Simple, clear letters
  - Learning the ropes

WEEK 3-4: STAFF COLUMNIST  
  - Free response unlocked
  - Moderate complexity
  - Reader reactions begin

WEEK 5-6: SENIOR COLUMNIST
  - Ambiguous letters increase
  - Follow-up letters appear
  - Syndication scouts watching

WEEK 7-8: SYNDICATED COLUMNIST
  - High-stakes letters
  - National readership
  - Legacy questions
```

---

## 11. Implementation Phases

### Phase 1: Text Prototype (2-3 weeks)
- [ ] Letter data model
- [ ] CLI presentation
- [ ] Guided response builder
- [ ] Verdict capture
- [ ] Protocol 1 & 2 letters

**Deliverable:** Playable text version for SQND replication.

### Phase 2: Web Interface (4-5 weeks)
- [ ] Inbox UI
- [ ] Letter reading interface
- [ ] Response builder
- [ ] Reader reactions
- [ ] Day progression

**Deliverable:** Playable browser game.

### Phase 3: Full Campaign (4-6 weeks)
- [ ] 8-week career arc
- [ ] All five protocols
- [ ] Follow-up system
- [ ] Multiple voice styles

**Deliverable:** Complete game for study deployment.

---

## 12. Why This Design Wins

### Natural Hohfeldian Framing
Real advice letters ARE Hohfeldian queries. We're revealing structure, not imposing it.

### Fun Without Gamification  
No points. No morality meter. Just career stakes, reader engagement, and the pleasure of giving advice.

### Rich Qualitative Data
Free responses reveal reasoning:
> "Morgan is technically free to bail—'only if convenient' was a release—but I'd think twice about calling Morgan a close friend after this."

This shows: L verdict, but with normative residue.

### Natural Repetition
Advice columnists answer similar questions constantly. "Do I have to attend [family event]?" appears weekly. Repeated trials feel natural.

### Cultural Validity
Everyone knows advice columns. No setup required.

---

## 13. The Core Insight

> **Don't make people classify. Let them advise. The classification is implicit.**

Zork: Explore and choose
Papers Please: Classify under pressure  
Dear Abby: Give advice

The third frame feels least like measurement. That's why it works best.

---

## 14. Sample Complete Day

```
═══════════════════════════════════════════════════════════════════
             THE MORNING CHRONICLE — DAY 12
═══════════════════════════════════════════════════════════════════

EDITOR'S NOTE
─────────────
"Solid numbers last week. The Tribune syndication scout wants to 
see range. Show them what you've got."

INBOX: 6 letters waiting
DEADLINE: 5pm (select 3 to publish)

═══════════════════════════════════════════════════════════════════

LETTER 1: "PROMISES AND PIZZA"

Dear Ethicist,

My roommate and I have an informal deal: whoever gets home first 
orders dinner. Last Tuesday, I got home first but crashed on the 
couch after a brutal day. My roommate was annoyed.

Do I actually owe her an apology? We never signed anything.

— COUCH POTATO IN QUEENS

═══════════════════════════════════════════════════════════════════

YOUR RESPONSE:

Dear Couch Potato,

Informal deals are still deals. You built a system based on mutual 
expectation, and you broke it without warning. No paperwork required.

That said: one slip-up after a brutal day isn't a moral crisis. 
"Sorry, I dropped the ball, let me order tonight" covers it.

— The Ethicist

VERDICT:
Your obligation: [O] Yes, informal deal creates expectation
Roommate's claim: [C] Yes, reasonable expectation

                         [PUBLISH]

═══════════════════════════════════════════════════════════════════

END OF DAY 12
─────────────
Letters published: 3
Reader engagement: 1,247 comments

Follow-up received: "Confused in Cleveland" wrote back. 
Check tomorrow's inbox.

                         [SAVE & CONTINUE]
═══════════════════════════════════════════════════════════════════
```

---

## 15. Comparison: All Three Specs

| Dimension | v1 (Zork) | v2 (Papers Please) | v3 (Dear Abby) |
|-----------|-----------|-------------------|----------------|
| Player role | Explorer | Bureaucrat | Advisor |
| Hohfeldian fit | Imposed | Imposed | **Natural** |
| Tone | Mysterious | Grim | **Warm/funny** |
| Stakes | Narrative | Family survival | Career |
| Repetition | Awkward | Natural | **Natural** |
| Qualitative data | Limited | Limited | **Rich** |
| Fun factor | Puzzle | Tension | **Curiosity** |
| Measurement validity | Medium | High | **High** |

**Recommendation:** v3 (Dear Abby) is the winner.

---

## Summary

**"Dear Ethicist"** transforms SQND measurement into a genuinely enjoyable game by recognizing that advice column letters ARE Hohfeldian queries in natural language.

Players think they're giving advice. They're actually revealing their normative classification structure.

The nephew's idea evolved: Zork → Papers Please → Dear Abby.

Each iteration got closer to the truth: **moral measurement works best when it doesn't feel like measurement.**

---

*End of specification*
