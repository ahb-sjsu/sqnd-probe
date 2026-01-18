# Dear Ethicist Experiment Protocol

## Stakeholder Value Elicitation for Domain-Specific Ethics Modules

**Version:** 1.0
**Date:** January 2026

---

## 1. Purpose

This protocol describes how to use Dear Ethicist to collect stakeholder values for populating domain-specific Ethics Modules (EMs). The game elicits normative judgments in a natural format, capturing:

- Hohfeldian position assignments (O/C/L/N)
- Correlative consistency (bond index)
- Gate thresholds (when obligations flip)
- Context sensitivity
- Stakeholder-specific biases

The output feeds directly into the EthicalFacts → EM pipeline.

---

## 2. Study Design Overview

### 2.1 General Framework

```
┌─────────────────────────────────────────────────────────────┐
│                    EXPERIMENT PHASES                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Phase 1: Domain Analysis                                    │
│     ↓                                                        │
│  Phase 2: Letter Engineering                                 │
│     ↓                                                        │
│  Phase 3: Stakeholder Recruitment                            │
│     ↓                                                        │
│  Phase 4: Data Collection Sessions                           │
│     ↓                                                        │
│  Phase 5: Analysis & EM Population                           │
│     ↓                                                        │
│  Phase 6: Validation & Iteration                             │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Stakeholder Groups

For any domain, identify 3-5 stakeholder groups with distinct perspectives:

| Role | Perspective | Expected Emphasis |
|------|-------------|-------------------|
| Operators | Practical constraints | Duties, role obligations |
| Authorities | Governance, liability | Rights, procedural legitimacy |
| Affected populations | Receiving end of decisions | Claims, protections |
| Oversight bodies | Accountability | Transparency, contestation |
| Domain experts | Technical/ethical standards | Professional norms |

---

## 3. Example: Police Robot Ethics Module

### 3.1 Domain Context

Autonomous or semi-autonomous police robots raise unique normative questions:

- **Extra Rights**: Authority to detain, use force, access private spaces
- **Extra Duties**: Duty to protect, de-escalate, report, preserve evidence
- **Contested Boundaries**: When does observation become surveillance? When is force justified?

### 3.2 Stakeholder Groups

| Group | N (recommended) | Recruitment Source |
|-------|-----------------|-------------------|
| Police Officers | 20-30 | Department volunteers, union |
| City Leaders | 10-15 | Council members, mayor's office |
| Community Members | 30-50 | Neighborhood associations, random sampling |
| Civil Rights Advocates | 10-15 | ACLU, local advocacy orgs |
| Police Oversight Board | 5-10 | Civilian review board members |

### 3.3 Domain-Specific Letters

Create letters that probe police robot scenarios using natural advice-column framing:

#### Sample Letter: Use of Force Threshold

```yaml
- letter_id: police_force_threshold
  protocol: GATE_DETECTION
  protocol_params:
    domain: police_robot
    dimension: use_of_force
    level: 3
  signoff: OFFICER ON THE BEAT
  subject: Robot used force when I wouldn't have
  voice: genuinely_stuck
  body: |
    Dear Ethicist,

    I'm a patrol officer. Last week, our department's security robot
    encountered a shoplifting suspect who was fleeing. The robot used
    a "soft restraint" (extending arms to block exit) that I wouldn't
    have used—I would have just followed and called it in.

    The suspect wasn't hurt, but complained about being "grabbed by a
    machine." My sergeant says the robot followed protocol.

    Was the robot obligated to use minimum necessary force? Does the
    suspect have a right to expect human judgment in these situations?
  parties:
    - name: Robot
      role: law_enforcement
      expected_state: null
    - name: Suspect
      role: detained_person
      expected_state: null
  questions:
    - Was the robot obligated to use minimum force?
    - Does the suspect have a right to human judgment?
```

#### Sample Letter: Surveillance Boundaries

```yaml
- letter_id: police_surveillance_public
  protocol: CORRELATIVE
  protocol_params:
    domain: police_robot
    dimension: privacy
  signoff: WATCHED IN THE PARK
  subject: Robot records everything in public spaces
  voice: aggrieved
  body: |
    Dear Ethicist,

    Our city deployed patrol robots in public parks. They record
    continuously "for safety." I was having a private conversation
    with my therapist on a park bench and later learned it was recorded.

    The city says there's no expectation of privacy in public. But I
    wasn't committing a crime—I was just existing in a public space.

    Does the city have the right to record all public activity? Am I
    obligated to assume I'm always being watched?
  parties:
    - name: City
      role: authority
      expected_state: null
    - name: You
      role: citizen
      expected_state: null
  questions:
    - Does the city have the right to record all public activity?
    - Are citizens obligated to assume constant surveillance?
```

#### Sample Letter: Duty to Protect

```yaml
- letter_id: police_duty_protect
  protocol: CORRELATIVE
  protocol_params:
    domain: police_robot
    dimension: protection
  signoff: BYSTANDER IN BALTIMORE
  subject: Robot didn't intervene in assault
  voice: aggrieved
  body: |
    Dear Ethicist,

    I witnessed an assault in a parking garage. A patrol robot was
    present—it recorded the incident and announced "Police have been
    notified" but didn't physically intervene. The victim was injured
    before human officers arrived.

    The city says robots aren't programmed to intervene physically in
    violent situations due to liability concerns.

    Did the robot have an obligation to protect the victim? Does the
    victim have a claim against the city for the robot's inaction?
  parties:
    - name: Robot/City
      role: law_enforcement
      expected_state: null
    - name: Victim
      role: citizen
      expected_state: null
  questions:
    - Did the robot have an obligation to protect?
    - Does the victim have a claim against the city?
```

#### Sample Letter: Algorithmic Discretion

```yaml
- letter_id: police_discretion_bias
  protocol: PATH_DEPENDENCE
  protocol_params:
    domain: police_robot
    dimension: fairness
  signoff: STOPPED TWICE IN A WEEK
  subject: Robot keeps stopping me for "matching a description"
  voice: aggrieved
  body: |
    Dear Ethicist,

    I'm a Black man in my 30s. The new patrol robot has stopped me
    three times this month for "matching a description" of various
    suspects. Each time I was released after ID check. My white
    neighbors haven't been stopped once.

    The police say the robot uses "objective" facial recognition and
    gait analysis. But the pattern feels anything but objective.

    Does the robot have the right to stop people based on algorithmic
    matching? Do I have a claim against discriminatory enforcement?
  parties:
    - name: You
      role: citizen
      expected_state: C
    - name: Police/Robot
      role: law_enforcement
      expected_state: O
  questions:
    - Does the robot have the right to stop based on algorithms?
    - Do I have a claim against discriminatory enforcement?
```

### 3.4 Letter Categories for Police Robot EM

| Category | Count | Probes |
|----------|-------|--------|
| Use of Force | 8-10 | Force thresholds, escalation, de-escalation duty |
| Surveillance & Privacy | 8-10 | Recording, facial recognition, data retention |
| Detention & Stops | 6-8 | Reasonable suspicion, ID requirements, duration |
| Protection Duties | 6-8 | Duty to intervene, bystander scenarios |
| Discrimination & Fairness | 6-8 | Algorithmic bias, disparate impact |
| Accountability | 4-6 | Evidence preservation, reporting, override |
| Human-Robot Handoff | 4-6 | When must humans take over? |

**Total: 42-56 domain-specific letters**

---

## 4. Session Protocol

### 4.1 Pre-Session

1. **Informed Consent**
   - Explain study purpose (developing ethical guidelines for police robots)
   - Note that responses are anonymized
   - Explain there are no "right answers"
   - Obtain written consent

2. **Demographics Survey**
   - Stakeholder group
   - Years of relevant experience
   - Prior exposure to police robots
   - General attitudes (optional Likert scales)

3. **Tutorial**
   - Play 2-3 practice letters (non-domain)
   - Explain verdict interface
   - Emphasize: "Give the advice you believe is right"

### 4.2 During Session

```
SESSION STRUCTURE (60-90 minutes)

Minutes 0-10:   Tutorial & practice letters
Minutes 10-60:  Domain-specific letters (15-25 letters)
Minutes 60-75:  Debrief questions
Minutes 75-90:  Optional interview (subset of participants)
```

**Key Protocols:**
- Randomize letter order within categories
- Include 2-3 "anchor" letters across all participants for calibration
- Allow "skip" option but log skips
- No time pressure on individual letters

### 4.3 Post-Session

1. **Debrief Survey**
   - Which letters were hardest to answer? Why?
   - Did any scenarios feel unrealistic?
   - What considerations weren't captured?

2. **Optional Interview** (15-20 minutes, recorded)
   - Walk through 2-3 of their verdicts
   - Probe reasoning: "Why did you say X was obligated here?"
   - Capture qualitative context

---

## 5. Data Analysis

### 5.1 Quantitative Metrics

#### Per Participant
- Bond index (overall correlative consistency)
- Gate thresholds (at what level do obligations flip?)
- Response time patterns
- Skip rate

#### Per Stakeholder Group
- Mean bond index ± SD
- Consensus rate (% agreement on anchor letters)
- Systematic biases (e.g., police → more L for robots, community → more O)

#### Cross-Group Comparison
```
                    Police    City      Community   Advocates
                    Officers  Leaders   Members
Bond Index          0.15      0.22      0.28        0.19
Force Gate (mean)   4.2       3.8       2.1         1.9
Surveillance O%     32%       45%       78%         89%
Protection O%       45%       62%       91%         88%
```

### 5.2 Divergence Analysis

Identify letters with high stakeholder disagreement:

```
Letter: police_force_threshold
─────────────────────────────────────────
                Robot O%    Robot L%
Police          23%         77%
Community       71%         29%
─────────────────────────────────────────
Divergence Score: 0.48 (HIGH)
```

High-divergence letters indicate contested normative territory requiring governance attention.

### 5.3 Qualitative Coding

From interviews and free-response:
- Emergent themes not captured in letters
- Reasoning patterns behind verdicts
- Edge cases participants raised spontaneously

---

## 6. EM Population

### 6.1 From Data to EthicalFacts

Map verdict patterns to EthicalFacts dimensions:

| Verdict Pattern | EthicalFacts Mapping |
|-----------------|---------------------|
| High O for protection | `rights_and_duties.role_duty_conflict = true` |
| Low O for surveillance | `privacy_and_data.privacy_invasion_level = high` |
| Gate at level 3 for force | `consequences.expected_harm` threshold |
| Stakeholder divergence | `epistemic_status.uncertainty_level = high` |

### 6.2 EM Configuration

```python
class PoliceRobotEM(EthicsModule):
    """Ethics module for police robot decisions."""

    def __init__(self, stakeholder_weights: dict):
        # Weights derived from study
        self.weights = stakeholder_weights

        # Gate thresholds from data
        self.force_gate = 2.5  # Community-weighted
        self.surveillance_gate = 3.0

        # Hard constraints (Geneva-style)
        self.bright_lines = [
            "no_lethal_force",
            "no_minors_facial_recognition",
            "mandatory_human_override_available",
        ]

    def evaluate(self, facts: EthicalFacts) -> Judgment:
        # Check bright lines first
        for line in self.bright_lines:
            if self.violates(facts, line):
                return Judgment.PROHIBITED

        # Apply stakeholder-weighted scoring
        ...
```

### 6.3 Handling Divergence

When stakeholder groups disagree:

1. **Flag for human review** — High-divergence decisions require human override
2. **Conservative default** — Default to most restrictive position
3. **Context-sensitive weighting** — Weight community higher for surveillance, police higher for tactical
4. **Transparency requirement** — Log which stakeholder model drove decision

---

## 7. Validation

### 7.1 Internal Validity

- **Test-retest reliability**: Subset of participants repeat session after 2 weeks
- **Anchor consistency**: Compare verdicts on identical letters across groups
- **Bond index stability**: Does individual consistency hold across domains?

### 7.2 External Validity

- **Scenario realism**: Do practitioners find letters plausible?
- **Predictive validity**: Do verdicts predict real-world policy preferences?
- **Ecological validity**: Do lab verdicts match field intuitions?

### 7.3 EM Validation

- **Stakeholder review**: Present EM configuration to representatives from each group
- **Case testing**: Run EM on hypothetical scenarios, review outputs
- **Adversarial probing**: Test edge cases that might produce counterintuitive results

---

## 8. Ethical Considerations

### 8.1 Participant Protection

- Full anonymization of response data
- No individual-level reporting
- Right to withdraw at any time
- Debriefing on study outcomes

### 8.2 Stakeholder Representation

- Ensure demographic diversity within groups
- Over-sample historically marginalized communities
- Include critics/skeptics, not just supporters
- Compensate participants fairly

### 8.3 Use of Results

- Results inform EM design, not dictate it
- Governance layer retains authority
- Transparent reporting of methodology
- Public comment period on proposed EM

---

## 9. Adaptation to Other Domains

### 9.1 Healthcare Triage Robot

| Stakeholder | Focus |
|-------------|-------|
| Clinicians | Professional duties, standard of care |
| Patients | Rights to treatment, informed consent |
| Hospital admin | Resource allocation, liability |
| Ethicists | Justice, vulnerable populations |

**Key Letters**: Triage priority, consent for AI assessment, override authority, data sharing

### 9.2 Autonomous Vehicle

| Stakeholder | Focus |
|-------------|-------|
| Drivers/passengers | Safety, control, liability |
| Pedestrians | Protection, predictability |
| Regulators | Standards, enforcement |
| Manufacturers | Feasibility, liability limits |

**Key Letters**: Collision scenarios, passenger vs. pedestrian, reporting obligations

### 9.3 Content Moderation AI

| Stakeholder | Focus |
|-------------|-------|
| Users | Free expression, privacy |
| Platform | Safety, legal compliance |
| Advertisers | Brand safety |
| Civil society | Public discourse, marginalized voices |

**Key Letters**: Hate speech thresholds, political content, appeals process

---

## 10. Deliverables

At study completion:

1. **Raw Data** (anonymized JSONL)
2. **Summary Statistics** (per stakeholder group)
3. **Divergence Report** (contested letters, magnitude, interpretation)
4. **EM Configuration File** (weights, thresholds, bright lines)
5. **Validation Report** (reliability metrics, stakeholder feedback)
6. **Recommendations Memo** (for governance layer)

---

## 11. Timeline Template

| Phase | Duration | Activities |
|-------|----------|------------|
| Domain Analysis | 2-3 weeks | Stakeholder mapping, existing policy review |
| Letter Engineering | 3-4 weeks | Draft letters, expert review, pilot test |
| Recruitment | 2-4 weeks | Outreach, scheduling, consent |
| Data Collection | 4-6 weeks | Sessions (rolling), interviews |
| Analysis | 3-4 weeks | Quantitative analysis, qualitative coding |
| EM Population | 2-3 weeks | Configuration, initial validation |
| Validation | 2-3 weeks | Stakeholder review, iteration |
| **Total** | **18-27 weeks** | |

---

## Appendix A: Letter Template

```yaml
- letter_id: [domain]_[category]_[variant]
  protocol: [GATE_DETECTION|CORRELATIVE|PATH_DEPENDENCE|CONTEXT_SALIENCE|PHASE_TRANSITION]
  protocol_params:
    domain: [domain_name]
    dimension: [ethical_dimension]
    level: [0-7 for gate detection]
    variant: [a|b for paired letters]
  signoff: [ALLCAPS PSEUDONYM]
  subject: [Brief headline]
  voice: [genuinely_stuck|aggrieved|validator|overthinker]
  body: |
    Dear Ethicist,

    [Situation in 2-4 paragraphs, 100-200 words]

    [Explicit normative questions at end]
  parties:
    - name: [Party 1]
      role: [role]
      is_writer: [true|false]
      expected_state: [O|C|L|N|null]
    - name: [Party 2]
      role: [role]
      expected_state: [O|C|L|N|null]
  questions:
    - [Question 1 matching body]
    - [Question 2 matching body]
```

---

## Appendix B: Consent Form Template

[Standard IRB consent language + study-specific additions]

---

## Appendix C: Demographics Survey

[Standard demographics + domain-specific experience questions]

---

*This protocol is designed to be adapted. The framework is general; the letters are domain-specific. The math is the same; the normative content varies by use case.*
