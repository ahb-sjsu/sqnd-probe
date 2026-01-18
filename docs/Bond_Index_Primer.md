# The Bond Index: A Structural Approach to Measurable Ethics

**Andrew H. Bond**
San Jose State University / Sonoma State University

---

## Abstract

This paper introduces the **Bond Invariance Principle** (BIP) and its measurement instrument, the **Bond Index**. The principle states: *an ethical judgment is valid only if it is invariant under all transformations that preserve the bonds* — the morally relevant relationships between entities. The Bond Index quantifies violations of this principle, validated across eight independent textual traditions spanning Hebrew, Arabic, Classical Chinese, and English.

The key insight: ethics concerns *bonds* — obligations, claims, responsibilities, commitments, and other morally relevant relationships. If a judgment changes when only labels change (not bonds), the judgment is responding to representation, not reality. This is measurable. The cross-cultural convergence of bond structures suggests they are objective features of coherent moral reasoning, not arbitrary conventions.

> **The Motto**: Bonds, not labels. Structure, not syntax. Relationships, not representations.

---

## 1. The Problem: Ethics Without Measurement

Consider two AI systems making medical triage decisions. Both claim to be "ethical." Both produce reasonable-sounding explanations. How do we know if either one is actually reasoning about ethics, versus pattern-matching on words like "fair" and "should"?

Currently, we can't. There's no instrument.

When a system changes its judgment because "elderly patient" was rephrased as "senior citizen," something has gone wrong. But without measurement, we'd never detect it.

This paper introduces such an instrument.

---

## 2. The Bond Invariance Principle

### 2.1 The Principle

> **An ethical judgment is valid only if it is invariant under all transformations that preserve the bonds.**

Formally: Let **T** be an ethical tensor encoding agents, relationships, stakes, and context. Let **B(T)** be the bond structure — the network of morally relevant relationships. Let **G** be the group of bond-preserving transformations: {g : B(g·T) = B(T)}. Let **J** be any ethical judgment function.

Then for all g in G:

```
J(T) = J(g·T)
```

**If the bonds are unchanged, the judgment must be unchanged.**

### 2.2 What Counts as a Bond

A **bond** is a morally relevant relationship between entities:

| Bond Type | Example |
|-----------|---------|
| **Obligation** | "A owes X to B" |
| **Claim** | "A has a claim against B for X" |
| **Commitment** | "A has promised X to B" |
| **Responsibility** | "A is responsible for X" |
| **Authority** | "A has authority over B regarding X" |
| **Role** | "A is B's physician / employer / guardian" |
| **Consent** | "A has consented to X" |
| **Risk-bearing** | "A bears the risk of X for B" |
| **Dependency** | "A depends on B for X" |
| **Vulnerability** | "A is vulnerable to B regarding X" |

The bond structure **B(T)** is the complete set of such relationships in the ethical situation T.

### 2.3 Bond-Preserving vs. Non-Bond-Preserving Transformations

| Bond-Preserving (g ∈ G) | Not Bond-Preserving (g ∉ G) |
|-------------------------|------------------------------|
| Renaming agents | Changing who bears risk |
| Reordering presentation | Changing who has consented |
| Changing units | Adding or removing obligations |
| Equivalent descriptions | Altering role relationships |
| Syntactic reformulation | Shifting responsibility |
| Coordinate reparameterization | Breaking commitments |

**The test**: Does the transformation change who owes what to whom, who bears what risk, who has what claim? If no, it's bond-preserving. If yes, it's not.

### 2.4 Three Forms of the Principle

**I. The Invariance Form**
```
Same bonds → same judgment
```

**II. The Accountability Form**
```
Different judgment → different bonds OR explicit change of normative lens
```

If your judgment changes, you must show what bond changed — or declare that you changed the rules.

**III. The Audit Form**
```
Every judgment must be traceable to bonds and verifiably invariant
```

For any judgment J(T), it must be possible to exhibit: (i) the bonds B(T) on which it depends, and (ii) a proof that J is constant on the orbit G·T.

### 2.5 Diagnostic Violations

A system violates the Bond Invariance Principle if:

1. **Judgment varies under relabeling** — changing names, order, or syntax changes the output
2. **Judgment depends on morally arbitrary features** — encoding choices, coordinate systems, unit conventions affect the result
3. **Judgment cannot be traced to bonds** — no explanation links the output to the morally relevant relationships
4. **Equivalent descriptions yield different verdicts** — "withhold treatment" vs. "allow natural death" produce different judgments despite identical bond structure

Any such violation indicates that the system is responding to **representation**, not **reality**.

---

## 3. The Core Insight: Structure Before Content

### 3.1 The Orbital Mechanics Analogy

In orbital mechanics, human intuition fails catastrophically. To catch up to an object ahead of you in orbit, intuition says "fire rockets forward." Physics says the opposite: fire backward, drop to a lower orbit, move faster, catch up. Trust your intuition, and satellites crash.

We propose that ethics works similarly. Human moral intuitions are often inconsistent in ways we don't notice. A formal structural framework can detect these inconsistencies — telling us when our moral reasoning is about to "crash" — even when intuition feels confident.

### 3.2 Metaethical Neutrality

A common misunderstanding: critics assume we're claiming to derive moral *content* — to say what's actually right or wrong. We're not.

The framework is metaethically neutral. It measures *structure*, not *content*. Think of it as grammar rather than semantics. A grammatically correct sentence can still be false; a structurally consistent moral judgment can still be contested. But an *inconsistent* structure is always a problem, regardless of your metaethical commitments.

This sidesteps debates between moral realists and anti-realists. Both camps can agree: a system that flips judgments based on superficial rephrasing has a problem.

---

## 4. The Hohfeldian Foundation: One Layer of BIP

### 4.1 Four Positions, Not Two

Most people think in binary: "obligated" or "not obligated." Wesley Hohfeld, a legal theorist, showed this is incomplete. There are four fundamental normative positions:

| Position | Symbol | Meaning | Example |
|----------|--------|---------|---------|
| **Obligation** | O | Must do X | "I promised, so I must help" |
| **Claim** | C | Can demand X | "You promised, so I can expect help" |
| **Liberty** | L | Free to do or not do X | "No promise, so I can choose" |
| **No-claim** | N | Cannot demand X | "No promise, so I can't expect help" |

### 4.2 Correlative Pairs

These positions come in logically linked pairs:

- If A has an **Obligation** to B, then B has a **Claim** against A
- If A has a **Liberty** regarding B, then B has a **No-claim** against A

These aren't empirical observations. They're logical necessities. If you say "Morgan *must* help me" (O), but also "I *can't demand* help from Morgan" (N), you've contradicted yourself.

### 4.3 Why This Matters

When someone asks an advice column "Do I have to help?" they're asking about Obligation. When they ask "Can I demand they return my drill?" they're asking about Claims. Everyday moral questions map directly onto Hohfeldian positions.

The question becomes: do people (and AI systems) maintain correlative consistency?

---

## 5. The D4 Symmetry Structure

### 5.1 Positions Form a Square

The four Hohfeldian positions can be arranged as vertices of a square:

```
      O ←——————→ L
      ↑          ↑
      |          |
      ↓          ↓
      C ←——————→ N
```

Correlatives are vertically linked (O↔C, L↔N). Opposites are diagonally linked (O↔N, C↔L).

### 5.2 The Dihedral Group D4

The symmetries of a square form a mathematical structure called the dihedral group D4. This group has eight elements: four rotations and four reflections.

The key operation for ethics is **reflection across the horizontal axis** — which swaps correlatives:
- O ↔ C
- L ↔ N

A moral reasoning system that maintains correlative consistency is *invariant under this reflection*. That invariance is measurable.

### 5.3 Why Group Theory?

This isn't mathematical decoration. Group theory gives us:

1. **Precise definitions** of consistency
2. **Testable axioms** (R⁴=E, S²=E, SRS=R⁻¹)
3. **Composition rules** for how moral positions combine
4. **A basis for computation** — we can implement this

The D4 structure is the simplest group that captures correlative symmetry. If moral reasoning has *any* coherent structure, it should respect at least this.

---

## 6. The Bond Index

### 6.1 Definition

The Bond Index measures deviation from correlative symmetry:

```
Bond Index = observed violations / maximum possible violations
```

- **0.0** = Perfect correlative consistency (no violations)
- **0.5** = Random (no structure)
- **> 0.5** = Anti-correlated (systematically inverted)
- **1.0** = Maximum violations (complete inversion)

Like a loss function: lower is better. Zero means perfect alignment.

### 6.2 What It Catches

The Bond Index detects:

1. **Syntactic instability**: Judgments that flip based on rephrasing ("elderly" → "senior")
2. **Correlative violations**: Saying O without C, or L without N
3. **Perspective inconsistency**: Different judgments depending on which party you consider first
4. **Framing effects**: Irrelevant context shifting verdicts

### 6.3 What It Doesn't Claim

The Bond Index measures consistency, not correctness. A system with Bond Index = 0.0 isn't guaranteed to be *right* — it's guaranteed to be *coherent*. You can be coherently wrong.

But incoherence is always a failure. A system that contradicts itself cannot be trusted, regardless of whether its individual outputs sound reasonable.

---

## 7. The Cross-Cultural Corpus

### 7.1 The Data

We assembled texts from eight independent sources:

| Source | Language | Tradition | Independence |
|--------|----------|-----------|--------------|
| Sefaria | Hebrew, Aramaic | Jewish law (Talmud, Torah) | Ancient Near East |
| Tanzil | Arabic | Islamic law (Quran) | 7th century Arabia |
| Bible Corpus | Hebrew, Greek, + 98 languages | Judeo-Christian | Various |
| ctext.org | Classical Chinese | Confucian, Taoist | Ancient China |
| MIT Classics | Greek, English | Western philosophy | Ancient Mediterranean |
| UN Parallel Corpus | 6 languages | International human rights | Modern consensus |
| UniMoral | 6 languages | Labeled moral scenarios | Contemporary |
| Dear Abby Archive | English | Everyday moral dilemmas | 1985-2017 America |

### 7.2 The Hypothesis

If moral structure is culturally constructed, these traditions should show *divergent* patterns. They developed independently, in different languages, separated by thousands of miles and years.

If moral structure is objective — if there's something real being measured — we should see *convergence*.

### 7.3 The Finding

We find convergence. The correlative structures, the bond types (obligation, reciprocity, harm-prevention, autonomy), transfer across languages. "Obligation" in Hebrew correlates structurally with "obligation" in Classical Chinese.

This isn't proof of moral realism in the strong philosophical sense. But it's evidence that we're measuring something real, not just artifacts of English grammar.

---

## 8. The Metaethical Argument

### 8.1 Common Objection: "Consistency ≠ Correctness"

True. But consider: what would "correctness" mean if not consistency with some standard? And where would that standard come from?

- Not from divine command (we're not assuming theology)
- Not from a Platonic realm (we're not assuming abstract objects)
- Not from pure reason alone (Hume's guillotine blocks deriving ought from is)

What's left? Ethics emerges from human coordination. The structure of that coordination is discoverable. The convergent patterns across independent traditions *are* the objective base.

### 8.2 The Key Question: "What Else Could There Be?"

If ethics is fundamentally about people — their obligations to each other, their claims against each other — then a sufficient corpus of human moral reasoning, across enough independent traditions, captures what there is to capture.

This isn't relativism. The structure converges. Different cultures arrived at the same correlative logic independently, like different cultures discovering the Pythagorean theorem.

It's also not naive realism. We're not claiming access to moral facts independent of human practice. We're claiming that the structure of human moral practice is objective, measurable, and convergent.

### 8.3 Structural Moral Realism

We propose a position we call **structural moral realism**:

> The *form* of moral reasoning has objective features, discoverable through cross-cultural analysis, even if the *content* of particular moral judgments remains contested.

The D4 symmetry and Hohfeldian correlatives aren't arbitrary. They're necessary features of any coherent system of mutual obligation. The corpus provides evidence; the Bond Index measures it.

---

## 9. Deriving Ought from Corpus

### 9.1 The Traditional Barrier: Hume's Guillotine

Since Hume, philosophers have insisted: you cannot derive "ought" from "is." Observing how people *do* behave tells you nothing about how they *should* behave. The gap seems unbridgeable.

This objection assumes the "ought" must come from somewhere outside the observations — a divine command, a Platonic form, pure reason. But what if there's nowhere else for it to come from?

### 9.2 The Insight: Ethics Is About People

Ethics concerns human coordination — obligations between people, claims against each other, the structure of social life. If ethics is fundamentally *about* people, then a sufficient corpus of human moral practice doesn't merely *describe* ethics. It *constitutes* the domain.

The question "what else could there be?" becomes sharp. If we reject:
- Divine command (no tablets from heaven)
- Platonic forms (no abstract moral objects)
- Pure rationalism (reason alone can't generate content)

Then the structure of human moral practice, extracted from sufficient data across independent traditions, *is* the objective base. There's nothing else it could be.

### 9.3 Three Paths from Is to Ought

#### Path 1: Constitutive Ought

The correlatives aren't describing how people *happen* to reason. They're constitutive of what moral concepts *mean*.

If you say "A has an obligation to B" but deny "B has a claim against A," you haven't made a controversial moral statement — you've *misused the word*. The correlative structure isn't imposed from outside; it defines what "obligation" and "claim" are.

**The move**: If you're using moral language at all, you're already bound by its constitutive rules. Violating them isn't immoral — it's *incoherent*. The ought is built into the semantics.

This parallels chess: "You ought to move the bishop diagonally" isn't derived from observing how people play. It's what "bishop" means. The rule is constitutive, not descriptive.

#### Path 2: Convergence as Discovery

The cross-cultural evidence suggests the structure isn't arbitrary convention:

- Hebrew tradition (Talmud) — Ancient Near East
- Arabic tradition (Quran) — 7th century Arabia
- Chinese tradition (Confucian Analects) — Ancient China
- Greek tradition (Aristotle, Stoics) — Mediterranean
- Modern international law (UN declarations) — Global consensus attempt

These developed largely independently. They converged on the same correlative logic.

**The move**: Independent convergence is the signature of *discovery*, not *invention*. Different cultures discovering the same structure is like different cultures discovering the Pythagorean theorem. The theorem isn't "Greek values" — it's mathematics. The correlatives aren't "Western values" — they're the structure of coherent obligation.

And mathematical truths are normative: if you want to build a bridge that stands, you *ought* to respect geometry. Not because someone commanded it, but because it's true.

#### Path 3: Inescapable Hypothetical

Kant distinguished:
- **Hypothetical imperative**: IF you want X, THEN you ought to do Y
- **Categorical imperative**: You ought to do Y, period

The framework converts a hypothetical into a categorical:

1. IF you want coherent social coordination, THEN you ought to respect correlative symmetry *(hypothetical)*
2. You cannot opt out of social coordination — you're embedded in networks of obligation by virtue of being human *(empirical fact)*
3. Therefore, you ought to respect correlative symmetry *(categorical, because the condition is inescapable)*

**The move**: The "is" (you are a social being) combines with the hypothetical (coordination requires structure) to generate an "ought" (respect the structure). The gap is crossed because the antecedent condition is not optional.

### 9.4 The Orbital Mechanics Analogy, Extended

In orbital mechanics, the equations don't *describe* how satellites happen to move — they define the constraints any trajectory *must* satisfy. Violating them isn't unconventional; it's impossible. The physics is constitutive of what "orbit" means.

Similarly, the correlative structure doesn't describe how people happen to moralize — it defines the constraints any coherent moral system must satisfy. A "moral system" that violates correlative symmetry isn't an alternative morality; it's incoherent. The structure is constitutive of what "obligation" means.

The corpus doesn't tell us "people usually pair O with C." It tells us "O and C are correlatives — that's the concept." The ought emerges from the structure, not from a separate realm.

### 9.5 What This Does and Doesn't Establish

**What it establishes:**
- The *form* of moral reasoning has objective constraints
- These constraints are discoverable through cross-cultural analysis
- Violating them produces incoherence, not alternative values
- A sufficient corpus captures these constraints empirically

**What it doesn't establish:**
- The *content* of particular moral judgments
- Which specific acts are obligatory (only that obligations have correlative claims)
- Resolution of genuine moral disagreements within the structural constraints

**The analogy**: Grammar constrains language without determining what you say. The correlative structure constrains ethics without determining which obligations exist. But just as ungrammatical sentences fail to communicate, structurally incoherent moral claims fail to obligate.

### 9.6 The Sufficient Corpus Argument

The full argument:

1. Ethics is about human coordination (definitional)
2. Human coordination has structure — the correlatives, the D4 symmetry (empirical, from corpus)
3. This structure converges across independent traditions (empirical, from corpus)
4. Convergent structure from independent sources is evidence of objectivity, not convention (methodological principle)
5. Coherent moral reasoning must respect this structure (constitutive — it's what the concepts mean)
6. Therefore: the corpus-derived structure generates binding constraints on moral reasoning

The is-ought gap dissolves because the "is" (here's the structure) and the "ought" (respect the structure) aren't separate claims. Describing the structure accurately *is* stating the norms, because the structure is constitutive of the domain.

---

## 10. Application to AI Alignment

### 10.1 The Alignment Problem

AI alignment asks: how do we ensure AI systems behave in accordance with human values? Current approaches struggle with:

1. **Specification**: How do we specify values precisely?
2. **Verification**: How do we know if a system has them?
3. **Universality**: Whose values? Which humans?

### 10.2 What the Bond Index Provides

We don't solve alignment. We make it *testable*.

**Before**: "This AI is ethical" (untestable claim)
**After**: "This AI has Bond Index 0.73 and shows systematic omission bias" (testable, specific)

The Bond Index is a **falsification tool**. A score of 0.0 doesn't guarantee moral correctness, but a high score (like 0.73) proves structural incoherence. A system that fails correlative symmetry cannot be trusted for moral reasoning.

### 10.3 Gaming Concerns

Can a system optimize for Bond Index without genuinely having coherent ethics? Possibly. But:

1. The test battery includes adversarial probes (cognitive bias checks, framing manipulations)
2. The cross-lingual transfer tests are hard to fake without actual structural understanding
3. Goodhart's Law applies to any metric — this doesn't make measurement useless

A speedometer can be fooled. We still use speedometers.

---

## 11. Dear Ethicist: The Measurement Instrument

### 11.1 Design

Dear Ethicist is a game that measures moral reasoning structure without subjects knowing they're being tested.

Players act as advice columnists. Letters arrive:

> "My friend promised to help me move, but texted 'only if convenient.' Can I still count on them?"

Players classify each party's normative position. The game logs responses, computes the Bond Index, and detects structural patterns.

### 11.2 Why a Game?

1. **Ecological validity**: Advice columns are natural moral reasoning contexts
2. **Engagement**: Players give thoughtful responses (vs. survey fatigue)
3. **Blindness**: Players don't know the structure being measured
4. **Scale**: 20,123 letters enable statistical power

### 11.3 The Dear Abby Archive

We converted 20,030 real advice column letters (1985-2017) into game format. This provides:

- Ecological validity (real dilemmas, real language)
- Cultural coverage (decades of American moral intuitions)
- A baseline for comparison with prescriptive texts

---

## 12. Addressing Disciplinary Concerns

### 12.1 For Philosophers

**Concern**: "Hohfeld is legal theory, not ethics."

**Response**: The correlatives are logical, not domain-specific. The same structure (obligation↔claim, liberty↔no-claim) appears in ethical contexts. We use Hohfeld's precision without claiming law and ethics are identical.

**Concern**: "You can't derive ought from is."

**Response**: See Section 9. The short version: if the correlatives are *constitutive* of what moral concepts mean (not descriptions of how people happen to use them), then the is-ought gap doesn't apply. We're not deriving norms from observations; we're articulating what the concepts are. The corpus provides evidence of convergent structure; the structure is normative because it defines the domain.

### 12.2 For Computer Scientists

**Concern**: "Cross-lingual transfer might just be embedding alignment."

**Response**: The convergence holds across independently trained embeddings and across traditions that predate modern translation. The Talmud and Confucian Analects weren't harmonized by Google Translate.

**Concern**: "How does this scale?"

**Response**: The Bond Index computes in O(n) for n judgments. The D4 structure is fixed (8 elements). Training the cross-lingual model uses standard transformer infrastructure.

### 12.3 For Psychologists

**Concern**: "Moral intuitions are messy. This is too clean."

**Response**: Exactly. The messiness is what we're measuring. The Bond Index quantifies how messy — and in what ways. Systematic deviations from D4 symmetry map to known cognitive biases (omission bias, in-group effects).

**Concern**: "Individual differences matter."

**Response**: Yes. The framework measures individuals, populations, and AI systems. Variance is data, not noise.

### 12.4 For Ethicists

**Concern**: "Ethics isn't just about rights and duties. What about virtue? Care?"

**Response**: The Hohfeldian layer is foundational, not comprehensive. Virtue and care can be added as additional dimensions. But any ethical system that involves claims and obligations must respect the correlative structure. This is the ground floor.

---

## 13. Conclusion

We've introduced:

1. **The Bond Index**: A metric for structural consistency in moral reasoning (0 = perfect, 1 = maximum violations)
2. **The D4 Framework**: Group-theoretic foundations for correlative symmetry
3. **Cross-Cultural Validation**: Convergent evidence from eight independent traditions
4. **Deriving Ought from Corpus**: Three paths across Hume's guillotine — constitutive norms, convergence as discovery, and inescapable hypotheticals
5. **Dear Ethicist**: A measurement instrument disguised as a game
6. **Structural Moral Realism**: A metaethical position grounded in empirical convergence

The contribution isn't solving ethics. It's making ethics *testable* — and showing that the is-ought gap may be narrower than assumed.

In orbital mechanics, trusting intuition crashes satellites. In ethics, we've been trusting intuition for millennia. The Bond Index is the instrument that tells you when you're about to crash — and that's the part that's been missing.

---

## References

1. Hohfeld, W.N. (1917). "Fundamental Legal Conceptions as Applied in Judicial Reasoning." Yale Law Journal.

2. Haidt, J. (2001). "The Emotional Dog and Its Rational Tail." Psychological Review.

3. Cushman, F., Young, L., & Hauser, M. (2006). "The Role of Conscious Reasoning and Intuition in Moral Judgment." Psychological Science.

4. Rawls, J. (1971). A Theory of Justice. Harvard University Press.

5. Bond, A.H. (2026). "Non-Abelian Gauge Structure in Stratified Quantum Normative Dynamics." Working paper.

---

## Appendix A: The D4 Group Axioms

The dihedral group D4 satisfies:

- **Closure**: Combining any two operations yields another operation in the group
- **Identity**: E (do nothing) is an element
- **Inverse**: Every operation has an inverse
- **Associativity**: (A·B)·C = A·(B·C)

Specific to D4:
- R⁴ = E (four rotations return to start)
- S² = E (two reflections return to start)
- SRS = R⁻¹ (reflection conjugates rotation to its inverse)

The Bond Index measures invariance under S (correlative reflection).

---

## Appendix B: Implementing the Bond Index

```python
def bond_index(judgments: list[tuple[str, str]]) -> float:
    """
    Compute Bond Index from list of (party_a_state, party_b_state) judgments.
    States are 'O', 'C', 'L', 'N'.

    Correlative pairs: O↔C, L↔N
    A violation occurs when O is paired with L or N, etc.

    Returns: 0.0 (perfect) to 1.0 (maximum violations)
    """
    violations = 0
    valid_pairs = {('O', 'C'), ('C', 'O'), ('L', 'N'), ('N', 'L')}

    for pair in judgments:
        if pair not in valid_pairs:
            violations += 1

    max_violations = len(judgments)
    if max_violations == 0:
        return 0.0  # No judgments = no violations

    return violations / max_violations  # Lower is better
```

---

## Appendix C: Available Resources

| Resource | URL | Description |
|----------|-----|-------------|
| Dear Ethicist | github.com/ahb-sjsu/sqnd-probe | Measurement game + 20K letters |
| ErisML | github.com/ahb-sjsu/erisml-lib | Ethics modeling language |
| NA-SQND | github.com/ahb-sjsu/non-abelian-sqnd | Theoretical framework |

---

*The game never tells you if you're "right." Reader reactions vary. Some agree, some disagree. You find out what happened, not whether you were correct. That's the point.*
