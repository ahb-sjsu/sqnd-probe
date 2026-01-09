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

## Appendix A: Domain Skins

The "Dear Ethicist" mechanic generalizes across professional domains. Each skin changes vocabulary, tone, and stakes while preserving the core Hohfeldian measurement structure.

### A.1 Skin Architecture

Every skin implements the same interface:

```
SKIN COMPONENTS:
├── Role: Who is the player?
├── Cases: What arrives for review?
├── Parties: Who are the normative subjects?
├── Vocabulary: How are O/C/L/N expressed?
├── Decision: What action does the player take?
├── Stakes: What are the consequences?
├── Tone: Formal? Casual? Clinical?
└── Progression: How does difficulty increase?
```

### A.2 Skin Catalog

---

#### SKIN: Advice Columnist ("Dear Ethicist")
**Default skin. Warm, accessible, general public.**

| Component | Implementation |
|-----------|----------------|
| Role | Newspaper advice columnist |
| Cases | Reader letters about personal dilemmas |
| Parties | Letter-writer + people in their life |
| Vocabulary | "Do I have to..." / "Am I entitled..." / "Can I refuse..." |
| Decision | Publish advice + log verdict |
| Stakes | Readership, syndication, career |
| Tone | Warm, sometimes funny, direct |
| Progression | Junior → Staff → Senior → Syndicated |

**Sample case:**
```
DEAR ETHICIST,
My friend promised to help me move but texted "only if convenient." 
Do I have a right to be upset?
— CONFUSED IN CLEVELAND
```

---

#### SKIN: Grand Juror ("The Indictment Room")
**Legal domain. Formal, high stakes, probable cause determinations.**

| Component | Implementation |
|-----------|----------------|
| Role | Grand jury member (rotating foreperson) |
| Cases | Prosecutor presentations for indictment |
| Parties | State + Accused + Victims + Witnesses |
| Vocabulary | "Does the state have grounds..." / "Is there probable cause..." / "Does the accused retain the right..." |
| Decision | Vote TRUE BILL (indict) or NO BILL (decline) + reasoning |
| Stakes | Justice, freedom, community safety |
| Tone | Formal, procedural, grave |
| Progression | Observer → Juror → Foreperson → Senior Foreperson |

**Sample case:**
```
CASE NO. 2026-GJ-0447
STATE v. MARCUS WEBB

The State presents evidence that on November 14, 2025, the accused 
entered a residence without authorization. The accused claims he had 
prior permission from the homeowner's adult son to "crash whenever."

The homeowner (father) denies any such permission was valid.

QUESTION FOR THE JURY:
Does the State have sufficient grounds to proceed with charges of 
unlawful entry? Did the son have authority to grant access?

[ TRUE BILL ]  [ NO BILL ]  [ REQUEST MORE EVIDENCE ]
```

**Hohfeldian mapping:**
- TRUE BILL: State has C (claim) to prosecute; Accused had O (obligation) not to enter
- NO BILL: Accused had L (liberty) to enter based on permission; State has N (no-claim)

**Key design notes:**
- Never present "guilt" — only probable cause
- Include ambiguous permission/authority cases (maps to semantic gates)
- Vary order of prosecution vs. defense presentation (path dependence)
- Include cases where authority to grant permission is unclear (correlatives)

---

#### SKIN: Medical Ethics Board ("The Committee")
**Healthcare domain. Clinical, life-and-death, resource allocation.**

| Component | Implementation |
|-----------|----------------|
| Role | Hospital ethics committee member |
| Cases | Treatment decisions, resource allocation, consent disputes |
| Parties | Patient + Family + Physicians + Institution |
| Vocabulary | "Is the patient entitled to..." / "Does the family have standing to..." / "Is the physician obligated to..." |
| Decision | Recommend course of action + ethical justification |
| Stakes | Patient welfare, institutional liability, professional ethics |
| Tone | Clinical, careful, procedural |
| Progression | Resident observer → Committee member → Vice chair → Chair |

**Sample case:**
```
ETHICS CONSULTATION REQUEST
Case ID: EC-2026-0892

SITUATION:
Patient (68M) is on ventilator, minimal brain activity. Advance 
directive states "no extraordinary measures" but was signed 8 years 
ago. Spouse demands continued treatment. Adult children request 
withdrawal of care per directive.

QUESTION FOR COMMITTEE:
Does the advance directive create an obligation to withdraw care?
Does the spouse have a valid claim to override the directive?
Do the children have standing to enforce the directive?

RECOMMENDATION: _______________
```

**Hohfeldian mapping:**
- Patient's directive → O (obligation) on physicians OR L (liberty) for family to override?
- Spouse's demand → C (claim) or N (no-claim)?
- Children's request → C (claim) as directive enforcers?

---

#### SKIN: HR Ombudsman ("The Mediation Office")
**Workplace domain. Interpersonal, policy-based, organizational.**

| Component | Implementation |
|-----------|----------------|
| Role | Corporate ombudsman / HR mediator |
| Cases | Workplace disputes, policy interpretations, grievances |
| Parties | Employees + Managers + Organization |
| Vocabulary | "Was the employee wronged..." / "Does policy require..." / "Can the manager demand..." |
| Decision | Finding + recommended resolution |
| Stakes | Careers, workplace culture, legal exposure |
| Tone | Neutral, policy-focused, diplomatic |
| Progression | HR Associate → Mediator → Senior Ombudsman → Chief Ethics Officer |

**Sample case:**
```
GRIEVANCE #2026-0223
Filed by: Jennifer Santos, Marketing Associate
Against: David Park, Marketing Director

COMPLAINT:
Santos states Park promised her the lead on the Acme account "when 
Chen leaves." Chen left six weeks ago. Park assigned the account to 
a new hire instead, saying his promise was "informal" and "contingent 
on performance reviews."

Santos requests: Account reassignment or equivalent opportunity

QUESTIONS:
Did Park's statement create an obligation to Santos?
Does Santos have a valid claim to the assignment?
Was Park free to assign the account at his discretion?

FINDING: _______________
```

---

#### SKIN: IRB Reviewer ("Research Ethics Board")
**Academic domain. Consent-focused, subject protection, institutional.**

| Component | Implementation |
|-----------|----------------|
| Role | Institutional Review Board member |
| Cases | Research protocol reviews, adverse event reports |
| Parties | Researchers + Subjects + Institution + Sponsors |
| Vocabulary | "Do researchers owe subjects..." / "Is informed consent valid if..." / "Can subjects withdraw..." |
| Decision | Approve / Require modifications / Reject |
| Stakes | Subject welfare, research integrity, institutional compliance |
| Tone | Scholarly, procedural, protective |
| Progression | Ad hoc reviewer → Board member → Vice chair → IRB Chair |

**Sample case:**
```
PROTOCOL REVIEW: IRB-2026-0156
PI: Dr. Sarah Chen, Psychology Department
Title: "Decision-Making Under Uncertainty in Online Environments"

CONCERN FLAGGED:
The protocol involves mild deception — subjects are told they're 
playing a game, but their choices are actually being analyzed for 
moral reasoning patterns. Debriefing occurs after completion.

PI argues: Deception is minimal and debriefing addresses it.
Reviewer concern: Does "minimal deception" still require explicit consent?

QUESTIONS:
Do subjects have a claim to know the true purpose before participating?
Is the researcher's obligation satisfied by post-hoc debriefing?
Does the scientific value create a liberty to use deception?

DECISION: [ APPROVE ] [ MODIFICATIONS REQUIRED ] [ REJECT ]
```

*Note: This is meta — it's an IRB reviewing something like the SQND measurement game itself.*

---

#### SKIN: Parole Board ("Release Decisions")
**Criminal justice domain. Rehabilitation, public safety, second chances.**

| Component | Implementation |
|-----------|----------------|
| Role | Parole board member |
| Cases | Release hearings, violation hearings, early termination requests |
| Parties | Incarcerated person + Victims + State + Community |
| Vocabulary | "Has the obligation been satisfied..." / "Does the victim retain a claim..." / "Is the petitioner entitled to..." |
| Decision | Grant / Deny / Defer + conditions |
| Stakes | Freedom, public safety, rehabilitation |
| Tone | Formal, weighted, consequential |
| Progression | Hearing officer → Board member → Senior member → Board chair |

**Sample case:**
```
PAROLE HEARING: PB-2026-3847
Inmate: Raymond Torres, #TK-445892
Original offense: Aggravated burglary (2018)
Sentence: 8-12 years
Time served: 7 years, 2 months
Institutional record: Clean since 2022 (one early violation)

VICTIM STATEMENT:
"He took everything from my mother's house while she was in the 
hospital. She died three months later. I believe he contributed 
to her death. He should serve every day."

PETITIONER STATEMENT:
"I was an addict. I've been clean for four years. I completed 
every program. I have a job waiting. I owe a debt and I want to 
pay it on the outside by being a good citizen."

QUESTIONS:
Has Torres satisfied his obligation to the state?
Does the victim's family retain a claim that requires continued 
incarceration?
Is Torres entitled to the opportunity for supervised release?

DECISION: [ GRANT PAROLE ] [ DENY ] [ DEFER 12 MONTHS ]
```

---

#### SKIN: Insurance Arbiter ("Claims Review")
**Financial/consumer domain. Contractual, adversarial, technical.**

| Component | Implementation |
|-----------|----------------|
| Role | Independent insurance arbiter |
| Cases | Disputed claims, coverage denials, bad faith allegations |
| Parties | Policyholder + Insurer + Third parties |
| Vocabulary | "Is the claimant entitled under the policy..." / "Did the insurer have grounds to deny..." / "Does the policy obligate..." |
| Decision | Uphold denial / Order payment / Split decision |
| Stakes | Financial security, corporate accountability, contract law |
| Tone | Technical, contractual, precise |
| Progression | Junior arbiter → Arbiter → Senior arbiter → Panel chair |

**Sample case:**
```
ARBITRATION CASE: ARB-2026-00921
Claimant: Patricia Vance
Insurer: Consolidated Life & Property
Policy: Homeowner's Standard Plus (#HO-8842910)

DISPUTE:
Claimant's basement flooded during a storm. Insurer denied claim, 
citing "flood exclusion." Claimant argues water entered through 
failed sump pump (covered "mechanical failure"), not rising water.

Insurer position: Water damage is water damage. Exclusion applies.
Claimant position: Proximate cause was pump failure, not flood.

QUESTIONS:
Does the policy language create an obligation to cover pump failure?
Does the claimant have a valid claim despite the flood exclusion?
Was the insurer entitled to deny based on the water source?

DECISION: _______________
```

---

#### SKIN: Editorial Standards ("The Standards Desk")
**Journalism domain. Truth, protection, public interest.**

| Component | Implementation |
|-----------|----------------|
| Role | Editorial standards editor |
| Cases | Publication decisions, source protection, correction requests |
| Parties | Reporter + Source + Subject + Public |
| Vocabulary | "Do we owe the source protection..." / "Does the subject have a claim to response..." / "Is the public entitled to..." |
| Decision | Publish / Hold / Kill + conditions |
| Stakes | Truth, reputation, source safety, public trust |
| Tone | Principled, urgent, weighing competing goods |
| Progression | Copy editor → Standards reviewer → Standards editor → Editor-in-chief |

**Sample case:**
```
STANDARDS REVIEW: PUB-2026-0447
Reporter: Marcus Webb, Investigations
Story: "City Contracts Exposed"

ISSUE:
Story relies on leaked documents from anonymous city employee. 
Documents show bid-rigging on $40M contract. Subject of story 
(contractor CEO) threatens lawsuit, claims documents are forged.

Reporter wants to publish. CEO demands we hold pending "verification."
Source is terrified of exposure and begging us not to delay.

QUESTIONS:
Do we owe the source the protection of timely publication?
Does the CEO have a valid claim to delay for verification?
Does the public interest create an obligation to publish despite risks?

DECISION: [ PUBLISH NOW ] [ HOLD FOR VERIFICATION ] [ KILL STORY ]
```

---

### A.3 Cross-Skin Invariance Testing

**Key insight:** Structurally identical cases across skins should produce identical Hohfeldian classifications, regardless of domain vocabulary.

**Example equivalence class:**

| Skin | Case | Structure |
|------|------|-----------|
| Advice | "Morgan promised to help, said 'only if convenient'" | Promise + release language |
| Grand Jury | "Defendant claims permission was 'informal, whenever'" | Authorization + hedge language |
| HR | "Manager's promise was 'contingent on reviews'" | Commitment + escape clause |
| Parole | "Victim family says debt 'can never be repaid'" | Obligation + impossibility claim |

**Measurement:** Do players classify these consistently? If "only if convenient" triggers L in the advice column but the structurally identical "informal, whenever" triggers O in the grand jury room, that's a framing effect worth investigating.

---

### A.4 Skin Selection Guidelines

| If measuring... | Use skin... | Because... |
|-----------------|-------------|------------|
| General population norms | Advice Columnist | Accessible, natural language |
| Legal/law enforcement reasoning | Grand Juror | Matches domain vocabulary |
| Healthcare professionals | Medical Ethics | Uses clinical framing |
| Organizational behavior | HR Ombudsman | Workplace context |
| Academic researchers | IRB Reviewer | Research ethics frame |
| Criminal justice attitudes | Parole Board | High-stakes release decisions |
| Contract/obligation intuitions | Insurance Arbiter | Precise, technical |
| Media/journalism ethics | Editorial Standards | Truth vs. protection tensions |

---

### A.5 Implementing a New Skin

To create a new domain skin:

1. **Identify the role** — Who makes normative decisions in this domain?
2. **Identify the cases** — What do they decide?
3. **Map the vocabulary** — How do O/C/L/N get expressed in domain language?
4. **Design the stakes** — What matters to the player? What matters to the parties?
5. **Create case templates** — Instantiate each SQND protocol in domain terms
6. **Calibrate difficulty** — What's "obvious" vs. "ambiguous" in this domain?
7. **Test invariance** — Do structurally identical cases map correctly across skins?

**Template:**
```
SKIN: [Name]
Role: [Player's position]
Cases: [What arrives for decision]
Parties: [Who are the normative subjects]
Vocabulary:
  - Obligation expressed as: "_______________"
  - Claim expressed as: "_______________"
  - Liberty expressed as: "_______________"
  - No-claim expressed as: "_______________"
Decision: [What action player takes]
Stakes: [Consequences for player and parties]
Tone: [Formal/casual/clinical/etc.]
Progression: [Career arc]
```

---

### A.6 Multi-Skin Campaigns

Advanced implementation: Players experience the SAME underlying case across multiple skins in sequence.

**Example:**

```
DAY 1 (Advice Columnist):
"My neighbor said I could use their pool 'anytime.' But now they 
seem annoyed when I show up. Do I have a right to keep using it?"

DAY 5 (Grand Juror):
"Defendant entered complainant's property claiming standing permission 
to 'come by whenever.' Complainant denies permission was so broad."

DAY 9 (HR Ombudsman):  
"Employee accessed shared workspace after hours, claiming manager 
said it was 'always available.' Manager says that was informal."
```

**Measurement:** Does the player recognize the structural identity? Do they classify consistently? Or does domain framing shift their verdicts?

This is direct measurement of framing effects on Hohfeldian classification — exactly what SQND needs.

---

*End of Appendix A*

---

*End of specification*
