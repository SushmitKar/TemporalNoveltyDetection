# Temporal Novelty Detection in Scientific Literature via Knowledge Graphs and Graph Neural Networks

> Can we tell how novel a research paper is — automatically, at the time of publication, without waiting years for citation data? This project builds a system that does exactly that.

---

## Overview

We build **TNoD**, a temporal knowledge graph covering 4,860 NLP papers from 2010 to 2025, and compute multi-signal novelty scores to distinguish genuinely disruptive papers from incremental ones. Three complementary signals are used:

- **Semantic novelty** — how semantically distant a paper is from its nearest neighbours in SciBERT embedding space
- **Structural novelty** — how many of a paper's entity-relation triples are new to the historical knowledge graph
- **Temporal GNN score** — a graph-level score from a TGN-inspired model with GraphSAGE aggregation and Bochner time encoding

All signals are validated against ground-truth labels from the SciND dataset [Gupta et al., 2024].

---

## Key Results

| Metric | NOVEL | SKG | p-value | Effect Size r |
|--------|-------|-----|---------|--------------|
| Semantic KNN (pre-2022) | 0.9545 | 0.9382 | 5.39e-45 | 0.439 (Medium) |
| TGN Novelty Score | 0.793 | 0.285 | 1.67e-289 | 0.927 (V. Large) |
| Disruption Classification | 73.4% Disruptive | 43.9% Disruptive | 4.40e-56 | 0.685 (Large) |
| Link Prediction AUC (Adamic-Adar) | 0.91–0.92 | — | — | Baseline = 0.50 |
| TGN Classifier AUC | 0.9269 | — | — | — |

Semantic KNN is significant across all 5 testable NLP domains (DIA, MT, QA, SA, SUM).

---

## Dataset — TNoD

Built by merging the SciND corpus (2010–2021) with a 2022–2025 OpenAlex extension across seven NLP domains.

| Split | Count | Description |
|-------|-------|-------------|
| NOVEL | 854 | Verified novel papers (ground truth positive) |
| SKG | 3,166 | Standard knowledge graph papers (incremental) |
| BLOG | 840 | Science blog posts (exploratory, excluded from validation) |
| **Total** | **4,860** | |

| Statistic | Value |
|-----------|-------|
| Knowledge edges | 577,024 |
| Entity nodes | 293,149 |
| Unique predicates | 18,495 |
| Papers with embeddings | 4,242 |
| Year range | 2010–2025 |
| NLP domains | 7 (DIA, MT, QA, SA, SUM, NLI, PAR) |

---

## Repository Structure

```
├── 01_KG_Construction/
│   ├── 03_combined_json.ipynb          # Resolve SciND local IDs to global paper identifiers
│   ├── 04_entity_nodes.ipynb           # Extract and hash entity nodes from triplets
│   ├── 05_openalex_integration.ipynb   # Fetch abstracts and citation data from OpenAlex
│   ├── 06_paper_nodes.ipynb            # Build master paper_nodes.csv
│   ├── 07_citation_edges.ipynb         # Build citation edge layer
│   └── 09_knowledge_edges.ipynb        # Build (source, target, predicate, year) edge set
│
├── notebooks_2022_2025/
│   ├── 01_paper_nodes_2022_2025.ipynb  # Parse 2022–2025 OpenAlex JSON structure
│   ├── 02_fetch_openalex_2022_2025.ipynb
│   ├── 03_knowledge_edges_2022_2025.ipynb
│   ├── 04_citation_edges_2022_2025.ipynb
│   └── 05_merge_and_verify.ipynb       # Merge all outputs and deduplicate
│
├── 02_Embedding_Gen/
│   └── 01_EmbeddingGenerator.ipynb     # SciBERT embeddings for 4,242 paper abstracts
│
├── 03_Novelty_Scoring/
│   ├── 01_semantic_novelty_score.ipynb # KNN cosine dissimilarity scoring
│   ├── 02_structural_novelty.ipynb     # Jaccard triplet overlap scoring
│   ├── 03_composite_novelty.ipynb      # Weighted combination (0.70/0.20/0.10)
│   ├── 04_baseline_analysis.ipynb      # Mann-Whitney U validation
│   ├── 05_ablation_study.ipynb         # Per-component ablation
│   └── 06_score_analysis.ipynb         # Domain and temporal breakdowns
│
├── 04_Temporal/
│   ├── 01_TGNN.ipynb                   # TGN-inspired GNN with GraphSAGE + Bochner encoding
│   ├── 02_temporal_analysis.ipynb      # Disruption classification, link prediction
│   └── 03_TransE_modelling.ipynb       # TransE KG embedding novelty signal
│
├── 05_Visualization/
│   └── Visualization.ipynb             # All 10 final figures
│
├── 06_Fixes/
│   ├── 01_entity_embeddings.ipynb      # SciBERT on 553 filtered entity names (node features for TGN)
│   ├── 02_cross_domain_diffusion.ipynb # Concept spread across NLP domains over time
│   ├── 03_concept_evolution.ipynb      # Entity co-occurrence evolution across eras
│   └── 04_semantic_primary_validation.ipynb  # Clean pre-2022 validation (p=5.39e-45)
│
├── outputs/
│   ├── paper_nodes.csv
│   ├── entity_nodes.csv
│   ├── knowledge_edges.csv
│   ├── novelty_feature_matrix_with_score.csv
│   ├── ablation_results.csv
│   ├── summary_statistics.csv
│   ├── top20_novel_papers.csv
│   └── visuals/                        # All generated figures
│
└── README.md
```

---

## Pipeline

```
SciND Raw Data (2010–2021)
        │
        ▼
01_KG_Construction  ──────────────────────────────────────────┐
(paper nodes, entity nodes, knowledge edges, citation edges)  │
        │                                                      │
        ▼                                                      │
notebooks_2022_2025                                           │
(extend to 2022–2025 via OpenAlex)                            │
        │                                                      │
        ▼                                                      │
02_Embedding_Gen                                              │
(SciBERT 768-dim vectors for 4,242 papers)                    │
        │                                                      │
        ▼                                                      │
03_Novelty_Scoring                                            │
(semantic KNN + structural Jaccard + composite score)         │
        │                                                      │
        ▼                                                      │
06_Fixes/01_entity_embeddings  ◄──────────────────────────────┘
(SciBERT on 553 entity names → node features for TGN)
        │
        ▼
04_Temporal
(TGN-inspired GNN + TransE + disruption classification)
        │
        ▼
05_Visualization
(10 final figures)
```

---

## Installation

```bash
git clone https://github.com/yourrepo/TNoD.git
cd TNoD
pip install -r requirements.txt
```

### Requirements

```
torch>=1.12.0
transformers>=4.20.0
scikit-learn>=1.0.0
pandas>=1.4.0
numpy>=1.21.0
scipy>=1.7.0
matplotlib>=3.5.0
seaborn>=0.11.0
networkx>=2.6.0
tqdm
requests
```

---

## Usage

### Step 1 — Build the Knowledge Graph

Run notebooks in `01_KG_Construction/` in order (03 → 04 → 05 → 06 → 07 → 09), then run all notebooks in `notebooks_2022_2025/` (01 → 05).

Outputs: `paper_nodes.csv`, `entity_nodes.csv`, `knowledge_edges.csv`, `citation_edges.csv`

### Step 2 — Generate Embeddings

```bash
# Run 02_Embedding_Gen/01_EmbeddingGenerator.ipynb
# Requires: paper_nodes.csv + openalex_metadata_full.csv
# Output: abstract_embeddings.npy (4242 × 768)
```

Also run `06_Fixes/01_entity_embeddings.ipynb` to generate entity node features for the TGN.

### Step 3 — Compute Novelty Scores

Run notebooks in `03_Novelty_Scoring/` in order (01 → 06).

Key output: `novelty_feature_matrix_with_score.csv` — one row per paper with all component scores and the final composite score.

### Step 4 — Temporal Modelling

Run notebooks in `04_Temporal/` in order. The TGN notebook (`01_TGNN.ipynb`) requires entity embeddings from Step 2 (`06_Fixes/01_entity_embeddings.ipynb`).

Key outputs: `tgnn_novelty_scores.csv`, `disruption_classification.csv`

### Step 5 — Visualisations

Run `05_Visualization/Visualization.ipynb` to generate all figures.

---

## Method

### Composite Novelty Score

```
s(P) = 0.70 × f_sem(P) + 0.20 × f_str(P) + 0.10 × f_cit(P)
```

- **f_sem**: cosine dissimilarity between paper P and its 5 nearest neighbours in the pre-publication SciBERT embedding space
- **f_str**: 1 − Jaccard(triplets of P, historical KG triples before year T)
- **f_cit**: normalised inverse citation density of P's reference list

Weights selected by ablation over five configurations.

### TGN-Inspired Temporal GNN

For each paper P at year t:
1. Retrieve SciBERT embeddings of all entities connected to P in the KG
2. Mean-pool via GraphSAGE aggregation → 32-dim neighbourhood vector
3. Concatenate with paper features (year, citation count, reference count, structural score) + Bochner time encoding
4. Pass through 2-layer MLP with sigmoid output → novelty score ∈ [0, 1]

Training: BCEWithLogitsLoss with class-weight correction for SKG/NOVEL imbalance, 40 epochs.

---

## Known Limitations

- **Composite score not significant on full dataset (p = 1.0)** due to vocabulary mismatch between 2010–2021 and 2022–2025 entity extraction pipelines. Primary validation is restricted to year ≤ 2021.
- **BLOG > NOVEL on semantic metrics** — informal writing style is interpreted as semantic novelty. BLOG split excluded from all statistical tests.
- **Original SciND BLOG papers not recovered** during KG construction. 2010–2021 subset contains NOVEL and SKG only.
- **TGN uses static entity embeddings** rather than full persistent memory modules. This trades some temporal expressivity for computational tractability.
- **Disruption index is a novelty-structural divergence proxy**, not the standard Wu et al. citation D-index, which requires multi-year citation data.

---

## Team

| Member | Contributions |
|--------|--------------|
| **Sparsh Kulkarni** | KG construction (Steps 1), 2022–2025 data integration |
| **Sushmit Kar** | SciBERT embeddings, novelty scoring (Step 3), entity embeddings, concept diffusion, semantic validation |
| **Ishant Kumar Jangra** | Temporal GNN (Step 4), TransE modelling, all visualisations (Step 5) |

Supervised by **Dr. Komal Gupta**, Department of Computer Science and Engineering.

---

## Citation

If you use TNoD or this codebase in your work, please cite:

```bibtex
@inproceedings{kar2026temporal,
  title     = {Temporal Novelty Detection in Scientific Literature via Knowledge Graphs and Graph Neural Networks},
  author    = {Kar, Sushmit and Kulkarni, Sparsh and Jangra, Ishant Kumar},
  booktitle = {Proceedings of IEEE},
  year      = {2026}
}
```

---

## References

1. Gupta et al., "SciND: A knowledge graph dataset for scientific novelty detection in NLP," 2024.
2. Rossi et al., "Temporal graph networks for deep learning on dynamic graphs," ICML Workshop, 2020.
3. Hamilton et al., "Inductive representation learning on large graphs," NeurIPS, 2017.
4. Beltagy et al., "SciBERT: A pretrained language model for scientific text," EMNLP, 2019.
5. Bordes et al., "Translating embeddings for modeling multi-relational data," NeurIPS, 2013.
6. Priem et al., "OpenAlex: A fully-open index of the world's research works," arXiv:2205.01833, 2022.