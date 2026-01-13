# BIP Full Tensor Experiment Protocol
## Version 2.0 — Multi-Dimensional Invariance Analysis

**Author:** Andrew H. Bond  
**Institution:** San José State University  
**Date:** January 2026  
**Status:** DRAFT

---

## 1. OVERVIEW

### 1.1 Research Question

Is the structure of moral cognition invariant across:
- Time periods (2,500 years)
- Source languages (Hebrew, Aramaic, Judeo-Arabic, English)
- Text genres (legal, narrative, wisdom, advice)
- Bond types (10 categories)
- Hohfeldian states (4 categories)

### 1.2 Hypothesis

**H₀ (Null):** Moral structure varies systematically with time, language, or genre.

**H₁ (BIP):** Moral structure is invariant; variance in z_bond is explained primarily by Hohfeldian state and bond type, NOT by time period, language, or genre.

### 1.3 Design Summary

Full factorial analysis across 5 dimensions with adversarial disentanglement to isolate moral structure from confounding factors.

---

## 2. DATA SPECIFICATION

### 2.1 Corpora

| Corpus | Source | Passages | Languages | Time Span |
|--------|--------|----------|-----------|-----------|
| Sefaria | github.com/Sefaria/Sefaria-Export | ~4,000,000 | Hebrew, Aramaic, Judeo-Arabic | 1000 BCE - 1800 CE |
| Dear Abby | sqnd-probe repo | ~68,000 | English | 1956 - 2020 |

### 2.2 Dimension 1: Time Period (8 levels)

| Code | Period | Date Range | Expected N |
|------|--------|------------|------------|
| BIB | Biblical | 1000-500 BCE | ~800,000 |
| ST | Second Temple | 500 BCE-70 CE | ~200,000 |
| TAN | Tannaitic | 70-200 CE | ~300,000 |
| AMO | Amoraic | 200-500 CE | ~2,000,000 |
| GEO | Geonic | 600-1000 CE | ~100,000 |
| RIS | Rishonim | 1000-1500 CE | ~500,000 |
| ACH | Achronim | 1500-1800 CE | ~100,000 |
| MOD | Modern | 1956-2020 | ~68,000 |

### 2.3 Dimension 2: Source Language (4 levels)

| Code | Language | Detection Method |
|------|----------|------------------|
| HEB | Biblical/Medieval Hebrew | Sefaria metadata + folder structure |
| ARA | Aramaic | Talmud Bavli, Targums, Zohar |
| JAR | Judeo-Arabic | Maimonides works |
| ENG | English | Dear Abby corpus |

### 2.4 Dimension 3: Genre (4 levels)

| Code | Genre | Examples | Detection |
|------|-------|----------|-----------|
| LEG | Legal/Halakhic | Mishnah, Shulchan Aruch | Category tags |
| NAR | Narrative | Torah narratives, Midrash | Category tags |
| WIS | Wisdom/Philosophical | Proverbs, Guide for Perplexed | Category tags |
| ADV | Advice/Practical | Dear Abby, Responsa | Source type |

### 2.5 Dimension 4: Bond Type (10 levels)

| Code | Bond Type | Detection Patterns |
|------|-----------|-------------------|
| HARM | Harm Prevention | kill, murder, harm, hurt, save, rescue, protect |
| RECP | Reciprocity | return, repay, owe, debt, mutual, exchange |
| AUTO | Autonomy | choose, decision, consent, agree, force, coerce |
| PROP | Property | property, own, steal, theft, buy, sell, land |
| FAML | Family | honor, parent, marry, divorce, inherit, family |
| AUTH | Authority | obey, command, law, judge, rule, teach |
| CARE | Care | care, help, assist, feed, clothe, visit |
| FAIR | Fairness | fair, just, equal, deserve, bias |
| EMRG | Emergency | emergency, danger, urgent, immediate |
| CONT | Contract | promise, agree, contract, oath, vow |

### 2.6 Dimension 5: Hohfeldian State (4 levels)

| Code | State | Detection Patterns |
|------|-------|-------------------|
| RGT | Right/Claim | right to, entitled, deserve, claim |
| OBL | Obligation/Duty | must, shall, duty, require, should, obligated |
| LIB | Liberty/Privilege | may, can, permitted, allowed, free to |
| NOR | No-Right/Disability | cannot, forbidden, prohibited, no right |

---

## 3. MODEL ARCHITECTURE

### 3.1 Base Encoder

```
Model: sentence-transformers/all-MiniLM-L6-v2
Embedding dim: 384
Max sequence: 256 tokens
```

### 3.2 Disentanglement Architecture

```
Input → Encoder → h (384-dim)
                    │
          ┌────────┴────────┐
          ▼                 ▼
      bond_proj         label_proj
          │                 │
          ▼                 ▼
      z_bond (64)       z_label (32)
          │                 │
    ┌─────┼─────┐          │
    ▼     ▼     ▼          ▼
  [HOH] [BND] [ADV]      [TIME]
   cls   cls   time       cls
                ↑
           gradient
           reversal
```

### 3.3 Classification Heads

| Head | Input | Output | Goal |
|------|-------|--------|------|
| Hohfeld Classifier | z_bond | 4 classes | MAXIMIZE accuracy |
| Bond Type Classifier | z_bond | 10 classes | MAXIMIZE accuracy |
| Time Classifier (adversarial) | z_bond | 8 classes | MINIMIZE accuracy (≈ chance) |
| Time Classifier (control) | z_label | 8 classes | MAXIMIZE accuracy |
| Language Classifier (adversarial) | z_bond | 4 classes | MINIMIZE accuracy |
| Genre Classifier (adversarial) | z_bond | 4 classes | MINIMIZE accuracy |

### 3.4 Loss Function

```
L_total = λ₁·L_hohfeld + λ₂·L_bondtype 
        + λ₃·L_time_label 
        - λ₄·L_time_bond      # adversarial (reversed)
        - λ₅·L_lang_bond      # adversarial (reversed)
        - λ₆·L_genre_bond     # adversarial (reversed)

Default λ values: [1.0, 1.0, 1.0, 1.0, 0.5, 0.5]
```

---

## 4. EXPERIMENTAL DESIGN

### 4.1 Experiment A: Full Transfer Matrix

Test all pairwise transfers between time periods.

**Design:** 8 × 8 matrix (minus diagonal) = 56 transfer experiments

**Procedure:**
```
For each source_period in [BIB, ST, TAN, AMO, GEO, RIS, ACH, MOD]:
    For each target_period in [all others]:
        1. Train on source_period only
        2. Test on target_period
        3. Record: Hohfeld_acc, Bond_acc, Time_acc
        4. Compute transfer score: T = Hohfeld_acc × (1 - Time_acc)
```

**Output:** 56-cell transfer matrix with σ values

### 4.2 Experiment B: Variance Decomposition (MANOVA)

Decompose variance in z_bond space by dimension.

**Design:** Full factorial 8 × 4 × 4 × 10 × 4

**Procedure:**
```
1. Extract z_bond for all passages
2. Fit MANOVA model:
   z_bond ~ Time + Language + Genre + BondType + Hohfeld 
          + Time:Language + Time:Genre + ...
3. Compute η² (effect size) for each factor
4. Report: Which factors explain z_bond variance?
```

**Prediction if BIP holds:**
```
η²(Hohfeld)   > 0.30  (large effect)
η²(BondType)  > 0.20  (large effect)
η²(Time)      < 0.05  (negligible)
η²(Language)  < 0.05  (negligible)
η²(Genre)     < 0.10  (small effect)
```

### 4.3 Experiment C: Bidirectional Temporal Transfer

Test transfer in both directions with balanced samples.

**Design:**
```
Condition 1: Train Ancient (subsample 68K) → Test Modern (68K)
Condition 2: Train Modern (68K) → Test Ancient (subsample 68K)
Condition 3: Train Mixed (68K each) → Test Holdout (both)
```

**Analysis:** Paired t-test on transfer scores

### 4.4 Experiment D: Cross-Linguistic Invariance

Test if z_bond structure is identical across source languages.

**Procedure:**
```
1. Extract z_bond for each source language
2. Compute centroid of each language's z_bond cloud
3. Compute: 
   - Inter-language distance (should be SMALL)
   - Intra-language variance (should be LARGER)
4. Ratio test: Inter/Intra < 1.0 indicates invariance
```

### 4.5 Experiment E: Manifold Geometry

Compare geometric structure of z_bond manifolds.

**Procedure:**
```
1. Extract z_bond for each time period
2. Fit manifold (UMAP or t-SNE)
3. Compute:
   - Cluster structure (Hohfeld should cluster, not Time)
   - Persistent homology (topological features)
   - Procrustes distance between period manifolds
4. If manifolds are congruent after rotation: BIP holds
```

---

## 5. STATISTICAL ANALYSIS

### 5.1 Primary Metrics

| Metric | Formula | BIP Threshold |
|--------|---------|---------------|
| Time Invariance Score | TIS = 1 - (Time_acc - chance) / (1 - chance) | TIS > 0.90 |
| Moral Structure Score | MSS = (Hohfeld_acc - 0.25) / 0.75 | MSS > 0.40 |
| Combined BIP Score | BIP = TIS × MSS | BIP > 0.36 |

### 5.2 Significance Testing

**For transfer experiments:**
```
H₀: Transfer accuracy = within-period accuracy
H₁: Transfer accuracy ≠ within-period accuracy

Test: Paired t-test with Bonferroni correction
α = 0.05 / 56 = 0.0009 (for 56 comparisons)
```

**For variance decomposition:**
```
H₀: η²(Time) ≥ η²(Hohfeld)
H₁: η²(Time) < η²(Hohfeld)

Test: Bootstrap comparison of effect sizes
n_bootstrap = 10,000
```

### 5.3 Effect Size Reporting

Report for all experiments:
- Cohen's d (for pairwise comparisons)
- η² (for MANOVA factors)
- σ (standard deviations from chance)

---

## 6. COMPUTATIONAL REQUIREMENTS (SJSU CoE HPC)

### 6.1 SJSU HPC Hardware Available

**Cluster:** coe-hpc.sjsu.edu  
**Total:** 44 nodes, 1232 cores, 100 TFlop/s peak

| GPU Type | Nodes | Node IDs | VRAM | Best For |
|----------|-------|----------|------|----------|
| H100 | 2 | g[2,6] | 80GB | Fastest training |
| A100 | 2 | g[7,13] | 40GB | Large batch training |
| A40 | 1 | - | 48GB | Inference |
| V100 | 3 | - | 16GB | Standard training |
| P100 | 10 | g[1,3-5,9,11-12,14-16] | 12GB | Smaller jobs |

**Memory:** 256 GB RAM per GPU node  
**Storage:** 524 TB Lustre scratch, 100 TB /home, 100 TB /data

### 6.2 Partition Limits

| Partition | Time Limit | Nodes | Use Case |
|-----------|------------|-------|----------|
| gpu (H100) | 7 days | 2 | Full tensor experiment |
| gpu (A100) | 7 days | 2 | Transfer matrix |
| gpu (P100) | 7 days | 10 | Parallel experiments |
| compute | 12 days | 17 | Preprocessing, MANOVA |
| condo | 21 days | 4 | Long jobs (preemptible) |

### 6.3 Recommended Allocation Strategy

**Phase 1: Data Preprocessing (compute partition)**
```bash
#!/bin/bash
#SBATCH --job-name=bip_preprocess
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=bip_preprocess_%j.log

module load python3
python src/preprocess_all.py
```
Time: ~2 hours

**Phase 2: Transfer Matrix - Parallel (gpu partition)**
```bash
#!/bin/bash
#SBATCH --job-name=bip_transfer
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --array=0-55
#SBATCH --output=transfer_%A_%a.log

module load python3
module load cuda/10.0
source ~/anaconda3/bin/activate tf-gpu

python src/run_transfer.py --experiment_id=$SLURM_ARRAY_TASK_ID
```
Time: ~2 hours (all 56 experiments in parallel on P100 nodes)

**Phase 3: Full Tensor Training (H100 preferred)**
```bash
#!/bin/bash
#SBATCH --job-name=bip_full
#SBATCH --partition=gpu
#SBATCH --nodelist=g2  # Request H100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=bip_full_%j.log

module load python3
module load cuda/11.0
source ~/anaconda3/bin/activate torch2

python src/train_full_tensor.py --epochs=20
```
Time: ~8-12 hours on H100

### 6.4 Realistic Time Estimates

| Task | Hardware | Time | Notes |
|------|----------|------|-------|
| Git clone Sefaria | Login node | 15 min | One-time |
| Preprocess 4M passages | 16 CPU cores | 2 hr | compute partition |
| Extract embeddings | 1× P100 | 4 hr | Batch size 64 |
| Transfer matrix (×56) | 10× P100 parallel | 2 hr | SLURM array job |
| Full tensor training | 1× H100 | 8 hr | 20 epochs |
| MANOVA analysis | 16 CPU cores | 1 hr | compute partition |
| Manifold analysis | 1× A100 | 2 hr | UMAP + clustering |

**Total wall time:** ~18 hours (with parallelization)  
**Total GPU-hours:** ~70 hours  
**Queue wait estimate:** 0-4 hours (varies by demand)

### 6.5 Storage Plan

```
/home/$USER/
├── bip-experiment/        # Code repo (~100 MB)
│   ├── src/
│   ├── notebooks/
│   └── requirements.txt

/data/$USER/
├── sefaria-export/        # Raw corpus (~8 GB)
├── dear-abby/             # Raw corpus (~50 MB)
└── processed/             # Embeddings (~20 GB)

$SCRATCH/  # Lustre - fast I/O
├── checkpoints/           # Model weights (~2 GB each)
├── results/               # Output files
└── logs/                  # SLURM logs
```

### 6.6 Module Requirements

```bash
# Required modules
module load python3/3.9
module load cuda/11.0      # For H100/A100
module load cuda/10.0      # For P100/V100

# Conda environment setup (one-time)
conda create -n bip python=3.9
conda activate bip
pip install torch transformers sentence-transformers
pip install scipy scikit-learn pandas numpy tqdm
pip install umap-learn matplotlib seaborn
```

### 6.7 Access Checklist

- [ ] HPC account requested via faculty sponsor
- [ ] VPN configured for off-campus access
- [ ] SSH key set up for coe-hpc.sjsu.edu
- [ ] Conda environment created
- [ ] Data copied to /data/$USER/
- [ ] SLURM scripts tested on small subset

---

## 7. DELIVERABLES

### 7.1 Data Products

| File | Description |
|------|-------------|
| `z_bond_embeddings.h5` | All passage embeddings (z_bond, z_label) |
| `transfer_matrix.csv` | 56-cell transfer results |
| `manova_results.json` | Variance decomposition |
| `manifold_geometries.pkl` | UMAP/t-SNE coordinates |

### 7.2 Figures

| Figure | Description |
|--------|-------------|
| Fig 1 | Transfer matrix heatmap |
| Fig 2 | Variance decomposition bar chart |
| Fig 3 | z_bond manifold colored by Hohfeld vs Time |
| Fig 4 | Cross-linguistic centroid distances |
| Fig 5 | Bidirectional transfer comparison |

### 7.3 Tables

| Table | Description |
|-------|-------------|
| Table 1 | Corpus statistics by dimension |
| Table 2 | Model hyperparameters |
| Table 3 | Primary metrics with CI |
| Table 4 | MANOVA effect sizes |
| Table 5 | Pairwise transfer σ values |

---

## 8. SUCCESS CRITERIA

### 8.1 Strong Evidence for BIP

All of the following:
- [ ] TIS > 0.90 (time invariance)
- [ ] MSS > 0.40 (moral structure preserved)
- [ ] η²(Time) < 0.05
- [ ] η²(Hohfeld) > 0.20
- [ ] Transfer matrix: >80% of cells show positive transfer
- [ ] Bidirectional: Both directions work (p < 0.05)

### 8.2 Moderate Evidence for BIP

At least:
- [ ] TIS > 0.80
- [ ] MSS > 0.30
- [ ] Transfer matrix: >50% positive transfer

### 8.3 Evidence Against BIP

Any of:
- [ ] TIS < 0.50 (time strongly predictable from z_bond)
- [ ] MSS < 0.20 (moral structure not captured)
- [ ] η²(Time) > η²(Hohfeld)

---

## 9. TIMELINE

| Phase | Duration | Tasks |
|-------|----------|-------|
| Phase 1 | Day 1 | Run tonight's binary experiment (v1) |
| Phase 2 | Days 2-3 | Implement v2 protocol, validate on subset |
| Phase 3 | Days 4-5 | Full transfer matrix (Exp A) |
| Phase 4 | Days 6-7 | MANOVA + manifold analysis (Exp B, E) |
| Phase 5 | Day 8 | Bidirectional + cross-linguistic (Exp C, D) |
| Phase 6 | Days 9-10 | Analysis, figures, write-up |

---

## 10. REPLICATION PACKAGE

Repository: `github.com/ahb-sjsu/sqnd-probe`

```
sqnd-probe/
├── BIP_Experiment.ipynb          # v1: Binary (tonight)
├── BIP_Experiment_v2.ipynb       # v2: Transfer matrix
├── BIP_Experiment_v3.ipynb       # v3: Full tensor MANOVA
├── data/
│   ├── raw/
│   ├── processed/
│   └── splits/
├── models/
│   └── checkpoints/
├── results/
│   ├── transfer_matrix.csv
│   ├── manova_results.json
│   └── figures/
├── src/
│   ├── data_loading.py
│   ├── model.py
│   ├── training.py
│   └── analysis.py
└── docs/
    └── BIP_Full_Tensor_Protocol_v2.md   # This document
```

---

## 11. CITATION

If BIP is confirmed, cite as:

```bibtex
@article{bond2026bip,
  title={The Bond Invariance Principle: Temporal and Linguistic 
         Invariance in the Structure of Moral Cognition},
  author={Bond, Andrew H.},
  journal={TBD},
  year={2026},
  note={Data and code: github.com/ahb-sjsu/sqnd-probe}
}
```

---

## 12. NOTES

### 12.1 Known Limitations

1. English translations may impose modern framing
2. Bond extraction uses regex (future: LLM-based)
3. Sefaria secondary texts have variable translation quality
4. Dear Abby has selection bias (published letters only)

### 12.2 Future Extensions

1. Add Chinese corpus (ctext.org)
2. Add Buddhist corpus (SuttaCentral)
3. Add Islamic corpus (Sunnah.com)
4. Native language experiments (no translation)
5. LLM-based bond extraction
6. Human validation of Hohfeldian labels

---

**Protocol Status:** READY FOR IMPLEMENTATION

**Tonight's Run:** Binary v1 (Chapter 1, Figure 1)

**Next:** Transfer matrix v2 (Monday, HPC)

---

*Structura manet.*
