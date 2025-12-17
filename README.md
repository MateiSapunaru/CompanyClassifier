# Company → Insurance Taxonomy Classification (Unsupervised / Weakly-Supervised)

## Overview

This project tackles the problem of **automatically mapping companies to an insurance taxonomy** in a setting **without ground-truth labels**. The goal is not to maximize a classic supervised metric (accuracy/F1), but to design a **robust, explainable, and tunable pipeline** that can operate in production with human-in-the-loop validation.

The solution is based on **information retrieval techniques (TF-IDF + cosine similarity)** combined with **carefully designed multi-label assignment rules**, iterative data enrichment, and principled evaluation.

The project was developed iteratively, with each version motivated by empirical analysis of score distributions and error patterns.

---

## Problem Setting

* **Input**: 9,494 companies with free-text descriptions, business tags, and metadata (sector, category, niche)
* **Output**: One or more insurance taxonomy labels per company
* **Constraints**:

  * No labeled training data
  * Taxonomy labels are short and semantically overlapping
  * Output must include confidence handling ("uncertain" cases)

---

## Data Exploration (EDA)

Key findings from EDA:

* Company descriptions are heterogeneous in length and quality
* Business tags and metadata (sector/category/niche) are highly informative
* Taxonomy labels are short (mean ~25 chars), limiting direct lexical matching
* Many companies are ambiguous and should reasonably receive multiple labels

These observations strongly influenced the modeling choices.

---

## Modeling Approach

### Core Architecture

1. **Text Preprocessing**

   * Normalization (lowercasing, whitespace cleanup)
   * Parsing of structured `business_tags`

2. **Vectorization & Retrieval**

   * TF-IDF vectorization
   * Cosine similarity between company texts and taxonomy labels

3. **Candidate Retrieval**

   * Top-K labels retrieved per company

4. **Multi-Label Assignment Logic**

   * Absolute similarity threshold
   * Relative threshold vs. top-1 score
   * Maximum number of labels
   * Explicit `confident` vs `uncertain` status

5. **Evaluation (No Ground Truth)**

   * Score distributions (top-1, gap top1–top2)
   * Percentage of `uncertain` predictions
   * Multi-label rate
   * Manual audit on stratified samples

---

## Methodology

### 1) Text Fields and Preprocessing

Each company comes with:

* `description` (free text)
* `business_tags` (stringified Python list)
* metadata: `sector`, `category`, `niche`

Preprocessing steps:

* Parse `business_tags` with safe parsing (e.g., `ast.literal_eval`) into `tags_list`
* Normalize all text (lowercase, whitespace normalization)
* Build a single retrieval text per company: `company_text`

### 2) Retrieval Model: TF‑IDF + Cosine Similarity

We treat the problem as **retrieval**:

* Vectorize company texts and taxonomy texts using TF‑IDF
* Compute cosine similarity between each company and each label
* Retrieve top‑K candidate labels per company (K=10)

Why retrieval?

* No ground truth labels → supervised training is not possible
* Taxonomy labels are short → matching is primarily lexical / keyword-driven

### 3) Multi‑Label Assignment (from top‑K candidates)

Retrieval produces a ranked list of labels with similarity scores. A separate policy converts this into final labels:

* `min_top1`: if the best score is below this → mark prediction as `uncertain`
* `abs_keep`: keep labels with score ≥ `abs_keep`
* `relative_keep`: keep labels with score ≥ `relative_keep * top1_score`
* `max_labels`: cap how many labels we return

This is the core place where we control **coverage vs quality**.

### 4) Label Enrichment (v3+)

Problem: taxonomy labels are short (e.g., "Accounting Services"), which limits overlap with company texts.

Approach:

* Create/maintain a file `outputs/taxonomy_enrichment_template.csv`
* For each label, optionally provide a **label_description** built from:

  * the original label text
  * domain keywords / synonyms / related terms (manually curated)

Example:

* `Veterinary Clinics` → `vet clinic, animal hospital, vaccinations, surgery, diagnostics`

Implementation detail:

* When enrichment is enabled, we set:

  * `label_text = normalize(label_description)`
* Otherwise:

  * `label_text = normalize(label)`

### 5) Field Weighting (v4+)

Key insight from EDA: `business_tags` and metadata are often more predictive than the long description.

We implemented **field weighting** by repeating high-signal fields when building `company_text`:

* Description × 1
* Tags × 4
* Sector/Category/Niche × 2

This increases their impact in TF‑IDF without changing the model class.

---

## Iterative Experiments (v1 → v4.1)

### v1_bad_thresholds — Baseline TF‑IDF with conservative thresholds

**Goal:** establish a working baseline end‑to‑end.

* Retrieval: TF‑IDF(word) + cosine
* Assignment: very strict (`min_top1=0.20`, `abs_keep=0.15`, `relative_keep=0.75`)

**Outcome:** extremely high `uncertain` rate because TF‑IDF scores are typically low on this dataset.

### v2_good_thresholds — Threshold calibration using score distributions

**Goal:** fix the main failure mode of v1 (thresholds not aligned with score distribution).

* Retrieval unchanged
* Assignment thresholds lowered (data-driven) to reduce `uncertain`

**Outcome:** large drop in `uncertain` without changing the model, proving the bottleneck was assignment policy.

### v3_enriched — Taxonomy enrichment (keywords/synonyms)

**Goal:** increase lexical overlap between company texts and label texts.

* Retrieval: TF‑IDF on enriched `label_text`
* Assignment: same as v2

**Outcome:** small but consistent gains; enrichment helped, but TF‑IDF still limited semantically.

### v4_enriched_weighted — Enrichment + Field weighting

**Goal:** amplify high-signal structured fields (tags + metadata).

* Retrieval input improved by weighted `company_text`
* Label enrichment kept

**Outcome:** biggest improvement overall (higher scores + lower uncertainty).

### v4.1_enriched_weighted_stricter — Quality-first policy

**Goal:** demonstrate the production trade-off between coverage and precision.

* Same retrieval as v4
* Stricter thresholds to reduce noisy multilabel outputs

**Outcome:** similar score distribution but more conservative label assignment.

---

## Results Table (All Versions)

| Version                    | Description                        | Uncertain % |  Top1 Mean | Top1 Median |   Gap Mean | Avg Labels | Multilabel % |
| -------------------------- | ---------------------------------- | ----------: | ---------: | ----------: | ---------: | ---------: | -----------: |
| **v1_bad_thresholds**      | Baseline TF‑IDF, strict thresholds |  **85.35%** |     0.1157 |      0.0879 |     0.0504 |     1.0496 |       4.1289 |
| **v2_good_thresholds**     | Threshold tuning only              |      56.02% |     0.1157 |      0.0879 |     0.0504 |     1.1214 |       9.8378 |
| **v3_enriched**            | Taxonomy enrichment                |      55.48% |     0.1171 |      0.0902 |     0.0516 |     1.1293 |      10.6488 |
| **v4_enriched_weighted**   | Enrichment + field weighting       |  **45.88%** | **0.1379** |  **0.1088** | **0.0614** |     1.1723 |      13.6718 |
| **v4.1_weighted_stricter** | v4 retrieval, stricter assignment  |      54.98% |     0.1379 |      0.1088 |     0.0614 |     1.1031 |       8.5633 |

### What matters most

* **v1 → v2**: assignment policy calibration drives huge improvements
* **v3**: enrichment helps a bit, but TF‑IDF is still lexical
* **v4**: field weighting is the largest quality jump
* **v4 vs v4.1**: shows a clear coverage vs precision trade‑off

------|-----------|-----------|-----------|-------------|----------|------------|--------------|
| **v1_bad_thresholds** | Baseline TF-IDF, poor thresholds | **85.35%** | 0.116 | 0.088 | 0.050 | 1.05 | 4.13 |
| **v2_good_thresholds** | Threshold tuning only | 56.02% | 0.116 | 0.088 | 0.050 | 1.12 | 9.84 |
| **v3_enriched** | Taxonomy enrichment (keywords) | 55.48% | 0.117 | 0.090 | 0.052 | 1.13 | 10.65 |
| **v4_enriched_weighted** | Field weighting + enrichment | **45.88%** | **0.138** | **0.109** | **0.061** | 1.17 | 13.67 |
| **v4.1_weighted_stricter** | Stricter thresholds (quality-first) | 54.98% | 0.138 | 0.109 | 0.061 | 1.10 | 8.56 |

### Key Observations

* **Threshold calibration alone (v1 → v2)** reduced uncertainty by ~30pp without changing the model
* **Taxonomy enrichment (v3)** provided marginal gains
* **Field weighting (v4)** was the most impactful improvement, reducing uncertainty by ~10pp
* **v4.1** demonstrates a quality-vs-coverage trade-off suitable for production settings

---

## Field Weighting (v4)

A major improvement came from explicitly weighting informative fields:

* Description × 1
* Business Tags × 4
* Sector / Category / Niche × 2

This amplifies high-signal metadata during TF-IDF vectorization without increasing model complexity.

---

## Evaluation Strategy (Without Ground Truth)

Because no true labels are available, evaluation is **indirect but methodologically correct**:

* Distribution of similarity scores
* Separation between top-1 and top-2 labels (gap)
* Percentage of `uncertain` predictions
* Multi-label frequency
* Manual audit on stratified samples (human-in-the-loop)

This mirrors how unsupervised taxonomy mapping is evaluated in real-world systems.

---

## Production Considerations

* Explicit handling of uncertainty
* Tunable thresholds depending on business needs
* Human audit pipeline
* Reproducible experiments via `run_name`
* No reliance on opaque black-box models

---

## Future Work

The next natural extension (not required to meet project goals):

* **v5: Ensemble retrieval** using word-level TF-IDF + char n-gram TF-IDF
* Adaptive field weighting based on score gaps
* Lightweight supervised fine-tuning if labels become available

---

## Conclusion

This project demonstrates a **full ML engineering workflow**:

* data understanding
* baseline modeling
* error analysis
* principled iteration
* evaluation without labels

The final solution is **explainable, robust, and production-ready**, and clearly documents trade-offs between coverage and confidence.

---
