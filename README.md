# Temporal Novelty Detection in Scientific Literature

## Project Overview

This project detects and analyzes temporal novelty in scientific papers using knowledge graphs, embedding models, and temporal classifiers. The pipeline spans Steps 1–5, covering KG construction, embedding generation, novelty scoring, temporal modelling, and visualization.

---

## Step 4–5: Temporal Modelling and Disruption Analysis

### 4.1 Temporal Classifier ("Time-Encoded MLP Classifier" / Temporal Feature Classifier with Bochner Time Encoding)

The temporal novelty model is a **2-layer MLP classifier** with a Bochner cosine time encoding concatenated to the input features. It has no graph structure, no adjacency matrix, and no message-passing operations. It is referred to informally as "TGNN" in some notebooks, but it is more accurately described as a **time-encoded MLP classifier** or **temporal feature classifier with Bochner time encoding**. All reported scores and results are unchanged by this clarification.

**Effect size (Mann–Whitney U test):** U / (n₁ × n₂) = **0.664** (rank-biserial AUC)
> Note: the quantity U / (n₁ × n₂) is the rank-biserial AUC, i.e., the probability that a randomly chosen novel paper scores higher than a randomly chosen non-novel paper. The corrected rank-biserial correlation is **r = 2 × AUC − 1 = 0.328**. p-values and significance conclusions are unchanged.

### 4.2 Disruption Index

The disruption score used here is computed as:

```
D = final_novelty_score − structural_novelty
```

This is a **novelty-structural divergence proxy** and is **not** the bibliometric disruption index from Wu et al. (2019), which is citation-based and requires forward citation data. The formula above captures the gap between semantic novelty (embedding-based) and structural novelty (graph-based), serving as an internal proxy measure.

**Effect size (Mann–Whitney U test):** U / (n₁ × n₂) = **0.685** (rank-biserial AUC)
> The corrected rank-biserial correlation is **r = 2 × AUC − 1 = 0.370**. p-values and significance conclusions are unchanged.

### 4.3 Top Disruptive Papers and Negative D Values

The top 20 disruptive papers (see `outputs/final/top20_disruptive_papers.csv`) all have negative disruption index values (range −0.11 to −0.03). This is not a contradiction: classification uses the **dataset median as the threshold** (median disruption index = −0.378), so any paper with a value above −0.378 is classified as *Disruptive* even if its absolute value is negative.

### 4.4 Link Prediction

Link prediction is evaluated using Adamic-Adar and Jaccard similarity across four time windows. Results are taken from `outputs/final/link_prediction_results.csv`.

| Window | AUC (Adamic-Adar) | AUC (Jaccard) |
|------------|-------------------|---------------|
| 2016→2017 | 0.9093 | 0.4138 |
| 2017→2018 | 0.9033 | 0.4574 |
| 2018→2019 | 0.9287 | 0.4014 |
| 2019→2020 | 0.9206 | 0.3695 |

The bar chart below shows both metrics with a 0.5 random-baseline dashed line:

![Link Prediction AUC](outputs/figures/link_prediction_auc.png)

---

## Limitations

**BLOG > NOVEL domain ordering:** In the temporal classifier's normalized novelty scores, BLOG papers score higher than NOVEL papers (0.335 vs. 0.251), contrary to the expected ordering NOVEL > SKG > BLOG. This suggests the model's semantic signal partly reflects academic writing-style divergence rather than genuine conceptual novelty — though alternative explanations (e.g., labelling artefacts or insufficient training data diversity) cannot be ruled out without further investigation. Blog-style text appears more divergent from the training distribution than actual novel papers, leading to inflated scores for BLOG. Addressing this would require domain-matched baseline comparisons, which is left for future work.

---

## Repository Structure

```
notebooks/
  01_KG_Construction/   — Knowledge graph construction
  02_Embedding_Gen/     — Embedding generation
  03_Novelty_Scoring/   — Novelty scoring (semantic, structural, composite)
  04_Temporal/          — Temporal modelling (time-encoded MLP, TransE)
  05_Visualization/     — Visualizations
outputs/
  figures/              — Core figures (link_prediction_auc.png, etc.)
  final/                — Final CSV outputs
  visuals/              — Additional visualizations
```
