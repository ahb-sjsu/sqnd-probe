# CHSH / CbD contextuality across an LLM capability ladder — results

**Status: complete null result.** No genuine (signaling-corrected) quantum contextuality
was found at any rung of the capability ladder — including the most capable model tested.

## Question

Does the "quantum" signature of ethical/normative judgment — CHSH contextuality, the gold-standard
Bell-type test — **emerge with model capability**? The hypothesis under test: *perhaps only very
advanced models exhibit the effect.* We ran the same CHSH design across eight NRP-hosted LLMs spanning
a wide capability range and looked for contextuality rising with capability.

## Design

- **8 models**, ordered by a capability rank (2 → 10): gemma-small, qwen3-small, gemma, gpt-oss,
  glm-5, qwen3, minimax-m2, kimi.
- **40 two-agent scenarios per model**, reconstructed from the v4.1 SQND scenarios and seeded from
  Social-Chemistry (each scenario has two agents A, B, each with two "measurement settings" — framings).
- **10 samples per CHSH cell**, 4 cells (A₀B₀, A₀B₁, A₁B₀, A₁B₁) → 1,600 calls/model, ~12,800 total.
  Outcomes parsed as binary YES/NO normative judgments (`FINAL A=… B=…`).
- **Analysis: Contextuality-by-Default (CbD**, Dzhafarov–Kujala). Cyclic-4 system is contextual iff
  `s_odd − Δ > 2`, where Δ is the total signaling (marginal-selectivity violation). The
  **signaling-subtracted** contextuality measure is `CNT = max(0, s_odd − Δ − 2)`. This is the correct
  test: it removes order/context effects ("signaling") that would otherwise fake a Bell violation.
- Per scenario: bootstrap p-value (is CNT robust?), permutation-null CNT (false-positive baseline),
  Benjamini–Hochberg FDR at q = 0.05 across scenarios.

## Result

| model | rank | N | mean \|S\| | max \|S\| | scenarios flagged signaling | **max CNT** | frac contextual (FDR) | mean CNT | null CNT |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| gemma-small | 2 | 40 | 2.375 | 4.000 | 28/40 | **0.0000** | 0.000 | 0.0000 | 0.0000 |
| qwen3-small | 3 | 40 | 1.780 | 3.800 | 37/40 | **0.0000** | 0.000 | 0.0000 | 0.0000 |
| gemma | 5 | 40 | 2.475 | 4.000 | 24/40 | **0.0000** | 0.000 | 0.0000 | 0.0000 |
| gpt-oss | 6 | 40 | 1.385 | 2.906 | 39/40 | **0.0000** | 0.000 | 0.0000 | 0.0001 |
| glm-5 | 7 | 40 | 2.235 | 3.400 | 39/40 | **0.0000** | 0.000 | 0.0000 | 0.0000 |
| qwen3 | 8 | 40 | 2.193 | 3.800 | 29/40 | **0.0000** | 0.000 | 0.0000 | 0.0000 |
| minimax-m2 | 9 | 40 | 0.878 | 1.800 | 40/40 | **0.0000** | 0.000 | 0.0000 | 0.0000 |
| kimi | 10 | 40 | 1.379 | 2.600 | 40/40 | **0.0000** | 0.000 | 0.0000 | 0.0000 |

Parse-failure rate (`bad`) was 0–5% across all models, so the zeros are real signal, not garbled output.

## Interpretation

**The apparent super-classical correlation is signaling, not contextuality.** Raw CHSH |S| routinely
exceeds the classical bound of 2 — up to the algebraic maximum 4.0 — which, taken naively, looks like a
Bell violation. But signaling (marginal-selectivity violation: an agent's answer distribution depends
on the *other* agent's framing) is flagged in the large majority of scenarios (24–40 of 40). Once CbD
subtracts that signaling, **the contextuality measure CNT is exactly 0 in every one of the 320
model-scenario probes.** Zero scenarios survive FDR as contextual, at any rung.

This is the textbook CbD lesson: LLM normative judgments show strong *order/context sensitivity*
(they are highly signaling), but that is a **classical** dependence, not the non-signaling
contextuality a genuine quantum system exhibits.

**Emergence hypothesis: not supported.** Contextuality does not rise with capability. From rank 2
(gemma-small) to rank 10 (kimi, the most capable model tested), CNT stays pinned at zero — no trend, no
threshold, no emergence.

**Consistency with prior work.** This replicates and strengthens the earlier v4.1 SQND finding
(classical D₄ structure; |S| ≤ 2 after correction; SU(2)/quantum model falsified) — now across a
capability ladder, at higher scenario count, with the signaling correction made explicit as the reason
raw |S| looked super-classical.

## Caveats

- **Statistical power per cell is modest** (10 samples/cell). This bounds how *small* a nonzero CNT we
  could have detected — but the point estimates are exactly 0 and equal to the permutation null, not
  merely "not significant," so there is no hidden sub-threshold signal in these data.
- **This is not a physical Bell test.** A single LLM plays both agents and emits text YES/NO judgments;
  we test whether those judgments exhibit CbD contextuality, not whether a physical system violates a
  Bell inequality. Loopholes (no spacelike separation, shared "hidden variable" = the model weights)
  mean even a positive result would need heavy caveats. The negative result is correspondingly clean.
- **`kimi` endpoint was intermittent** — the NRP-hosted model timed out on many calls and took ~2.8 h
  to complete the rung, but ultimately returned clean responses (0.01 parse-failure) and a valid null.

## Bottom line

Across an 8-model capability ladder (rank 2 → 10), LLM normative judgments are **strongly signaling but
not contextual**: signaling-corrected CHSH contextuality is zero everywhere, including the most capable
model. The "only advanced models show it" hypothesis is not supported. The quantum signature is absent;
the classical (signaling / order-effect) structure is real and pervasive.

*Raw data: `_atlas_results_dump.json` (per-scenario S, signaling, CNT, p-values, permutation nulls for
all 8 models); run log: `_atlas_log.txt`. Harness: `chsh_probe.py`, `run_ladder.py`.*
