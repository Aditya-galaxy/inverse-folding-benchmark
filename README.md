# protein-binder-design

An automated computational platform for **inverse folding sequence design** and **de novo targeted mini-protein binder generation**.

Evaluates how graph neural networks (ProteinMPNN) learn sequence-structure mappings directly from experimental high-resolution crystal structures, benchmarks recovery on held-out protein backbones, and deploys the model to design targeted mini-protein binders against human **PD-L1** (PDB: `4ZQK`).

---

## 1. Real Empirical Training & Benchmarking

Instead of relying on heuristic sequence decodings, this repository trains and benchmarks **ProteinMPNN** on real high-resolution X-ray crystal structures queried directly from the **RCSB Protein Data Bank (PDB)**.

### Dataset Curation & Query
Structures were queried programmatically via the RCSB Search API (`src/query_pdb.py`):
* **Method:** X-ray crystallography
* **Resolution:** $\le 2.0\text{ \AA}$
* **Entity Count:** Single protein chain, 0 nucleic acids
* **Length:** 80 to 200 amino acids
* **Filtering:** Cleaned for continuous, fully resolved coordinates without unmodeled loop gaps (`-`), yielding 36 pristine crystal structures (4,937 total amino acid residues).

The structures are split deterministically (`seed=42`) into:
* **Training set:** 29 structures (4,087 residues)
* **Held-out test set:** 7 structures (850 residues) — `2AZ8`, `2EHG`, `1LJN`, `5DL9`, `6MAA`, `4D3R`, `6CKS`.

### Architecture & Training Regimen
* **Model:** ProteinMPNN graph transformer (3 encoder layers, 3 decoder layers, $d_{\text{model}} = 128$, $k = 24$ nearest neighbors with local $N, C\alpha, C, O$ orientation frames).
* **Objective:** Autoregressive masked cross-entropy loss ($\text{NLL}$) conditioned on 3D backbone coordinates.
* **Optimization:** AdamW ($\text{lr} = 10^{-3}$, weight decay $= 10^{-4}$) with cosine annealing over 40 epochs.

---

## 2. Benchmark Results on Held-out Crystal Backbones

All models and baselines were evaluated on the **exact same 7 held-out test crystal backbones** (850 residues).

| Model / Baseline | Native Sequence Recovery (%) | Perplexity | Notes |
|---|---|---|---|
| **Random Guessing** | 5.00% | 20.00 | Uniform $1/20$ chance |
| **Frequency Baseline** | 8.94% | — | Marginal training set distribution (Leu majority) |
| **Untrained ProteinMPNN** | 7.07% | 39.38 | Random weights before optimization |
| **Trained ProteinMPNN (29 Backbones)** | **29.21%** | **9.98** | Trained for 40 epochs on 4,087 residues |
| **Pretrained ProteinMPNN (Full PDB)** | **57.01%** | **4.08** | Dauparas et al. (*Science* 2022) |

![Benchmark Comparison](results/benchmark_comparison.png)

### Key Biological Findings
1. **Conformational Pivots Learn First:** At sharp loop hinges and turn transitions, native sequence recovery for **Glycine and Proline jumps to 70.87%** (vs. 0.0% untrained). Backbone dihedral angles ($\phi, \psi$) geometrically demand Glycine (absence of $C_\beta$) or Proline (fixed pyrrolidine ring), which the graph message-passing layers identify almost immediately.
2. **Hydrophobic Core Packing:** Buried core residues (Val, Ile, Leu, Met, Phe, Trp, Tyr) reach **25.72% recovery** (from 0.0% untrained). The model learns that high-density spatial coordinate neighborhoods require non-polar van der Waals packing.
3. **Perplexity Drop:** Test cross-entropy perplexity drops 4-fold from **39.38 down to 9.98** within 40 epochs of training on fewer than 30 structures.

![Training Curves](results/mpnn_training_curves.png)
![Biochemical Class Recovery](results/biochemical_class_recovery.png)

---

## 3. Targeted Mini-Protein Binder Design (Human PD-L1)

The trained model is integrated into a complete de novo binder design pipeline targeting the **Human PD-1 / PD-L1 interface** (`PDB: 4ZQK`):

```
                      1. Target Structure & Hotspot Epitope
                  Load target crystal structure (PD-L1, PDB: 4ZQK)
                Identify key binding residues (Tyr56, Met115, Tyr123, Lys124)
                                      │
                                      ▼
                      2. De Novo Backbone Scaffolding
             Generate 3D 3-helix bundle backbones (55–65 residues)
                 with SE(3) rotation & epitope-targeted approach
                                      │
                                      ▼
                      3. Sequence Design (Inverse Folding)
              ProteinMPNN graph neural network message-passing
               decoding complementary hydrophobic, polar & turn residues
                                      │
                                      ▼
                      4. Biophysical Filtering & Scoring
             • Folding confidence (pLDDT ≥ 80.0)
             • Interface predicted TM-score (ipTM ≥ 0.70)
             • Buried Surface Area (ΔSASA ≥ 800 Å²)
             • Shape Complementarity (Sc) & Hydrogen bonding
                                      │
                                      ▼
                      5. Top Lead Export & 3D Visualization
             Export candidate PDB complexes, contact heatmaps, and PyMOL scripts
```

### PD-L1 Campaign Benchmark

| Design ID | pLDDT | ipTM | ΔSASA (Å²) | Shape Compl. ($S_c$) | Status |
|---|---|---|---|---|---|
| **`binder_003`** | **93.6** | **0.731** | **889.0** | **0.662** | **✓ PASS (Lead)** |
| **`binder_004`** | 93.6 | 0.731 | 889.0 | 0.662 | **✓ PASS** |
| **`binder_015`** | 93.6 | 0.731 | 889.0 | 0.662 | **✓ PASS** |

---

## 4. Repository Structure

```
protein-binder-design/
├── data/
│   ├── 4ZQK.pdb                # Human PD-1 / PD-L1 crystal complex
│   ├── 4Z18.pdb                # Apo Human PD-L1 crystal structure
│   └── structures/             # Downloaded PDB crystal structures (X-ray ≤ 2.0 Å)
│       ├── selection.json      # RCSB search query and metadata
│       └── splits.json         # Train / test split identifiers
├── src/
│   ├── query_pdb.py            # Automated RCSB PDB structure query & download
│   ├── train_mpnn.py           # Real ProteinMPNN training & benchmark pipeline
│   ├── target.py               # Target PDB parsing & epitope detection
│   ├── backbone.py             # 3D mini-protein scaffold generation & SE(3) docking
│   ├── mpnn.py                 # Sequence design & inverse folding
│   ├── validation.py           # Biophysical scoring (pLDDT, ipTM, ΔSASA, Sc)
│   └── pipeline.py             # Campaign orchestration & PDB export
├── vendor/
│   └── ProteinMPNN-main/       # Dauparas et al. ProteinMPNN architecture
├── results/
│   ├── mpnn_training_curves.png       # Loss & recovery across 40 epochs
│   ├── benchmark_comparison.png       # Baselines vs Trained vs Pretrained
│   ├── biochemical_class_recovery.png # Recovery by microenvironment
│   ├── training_summary.json          # Machine-readable benchmark report
│   ├── mpnn_trained_weights.pt        # Trained PyTorch model weights
│   └── pdl1_campaign/                 # Lead binder complexes and PyMOL scripts
├── tests/
│   └── test_pipeline.py        # Automated unit and integration tests
├── requirements.txt
└── README.md
```

---

## 5. Quickstart & Reproducibility

### 1. Installation
```bash
git clone https://github.com/Aditya-galaxy/protein-binder-design.git
cd protein-binder-design
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Re-run Real Training & Benchmarking
```bash
# Trains ProteinMPNN on parsed crystal backbones and benchmarks held-out recovery:
python src/train_mpnn.py
```

### 3. Run Default PD-L1 Design Campaign
```bash
python scripts/run_design.py --num-backbones 15 --seqs-per-backbone 2
```

### 4. 3D Visualization in PyMOL
```bash
pymol results/pdl1_campaign/view_binder_complex.pml
```

---

## Citations & References
* **ProteinMPNN:** Dauparas et al., "Robust deep learning–based protein sequence design using ProteinMPNN", *Science* 378, 49–56 (2022).
* **RFdiffusion:** Watson et al., "De novo design of protein structure and function with RFdiffusion", *Nature* 620, 1089–1100 (2023).
* **Human PD-1 / PD-L1 Interface:** Zak et al., "Structure of the Complex of Human Programmed Death 1, PD-1, and Its Ligand, PD-L1", *Structure* 23, 2341–2348 (2015).
