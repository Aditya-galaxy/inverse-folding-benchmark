# protein-binder-design

**How much of a protein's native sequence can a graph neural network recover from its backbone alone?**

Trains a ProteinMPNN-style inverse folding model on real X-ray crystal structures from the RCSB PDB and benchmarks native sequence recovery on held-out backbones, against a random baseline, an amino-acid frequency baseline, an untrained network, and the published pretrained weights run on the same held-out residues.

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

All models and baselines were evaluated on the **same 7 held-out crystal backbones** (850 residues).

| Model / Baseline | Recovery | 95% CI over structures | Perplexity |
|---|---|---|---|
| Random guessing | 5.00% | — | 20.00 |
| Frequency baseline (always Leu) | 8.94% | — | — |
| Untrained ProteinMPNN | 7.07% | — | 39.38 |
| **Trained here (29 structures)** | **29.21%** | **[23.5%, 36.2%]** | 9.98 |
| Pretrained, full PDB (Dauparas et al. 2022) | 56.28% | [52.6%, 59.2%] | 4.08 |

**Seven structures is very little evidence, and the interval says so.** Residues inside one protein
share a fold, a composition and a crystallography run, so 850 residues are nowhere near 850
independent observations. The intervals above resample *structures*, which is the real unit, and
per-structure recovery for the trained model ranges from 18.4% (`4D3R`) to 48.0% (`2AZ8`).

**The pretrained row is not a held-out comparison.** ProteinMPNN's published weights were trained on
the PDB up to roughly mid-2021. Every one of these seven structures was deposited between 2002 and
2018, so all seven were almost certainly in its training set. Its score is a training-set number
placed beside a genuine held-out one, and the gap between 29% and 56% is correspondingly overstated.
It is kept in the table as a ceiling reference, not as a fair baseline.

**Recovery is an average, not a single number.** ProteinMPNN draws a new decoding order on every
forward pass, so one evaluation is one sample. Each figure above averages 10 decoding orders per
structure; the spread across orders is 0.69 points for the trained model and 1.54 for the pretrained
one, which is why a single-draw number can move by a point or so between runs.

Reproduce with `python src/evaluate.py`, which writes `results/evaluation.json` including
per-structure recovery and deposition years.

![Benchmark Comparison](results/benchmark_comparison.png)

### What the model learns first
1. **Conformational pivots.** Recovery for glycine and proline reaches **70.9%**, against 0% untrained.
   Backbone dihedrals geometrically demand glycine (no C-beta) or proline (fixed ring), and the
   message-passing layers pick this up almost immediately.
2. **Hydrophobic core packing.** Buried residues (V, I, L, M, F, W, Y) reach **25.7%**, from 0%
   untrained: dense neighbourhoods call for non-polar packing.
3. **Perplexity.** Held-out perplexity falls from 39.4 to 10.0 in 40 epochs on fewer than 30 structures.

These three come from a single training run on 7 test structures and are offered as observations,
not as established facts about what inverse folding models learn.

![Training Curves](results/mpnn_training_curves.png)
![Biochemical Class Recovery](results/biochemical_class_recovery.png)

---

## 3. What this does not do

The trained model designs sequences for a backbone. It does **not** predict whether those sequences fold, or whether they bind anything.

Validating a designed binder needs structure prediction (AlphaFold2, ESMFold) and ultimately wet-lab binding assays. None of that is run here, so no folding confidence (pLDDT), interface score (ipTM) or binding claim is reported. `src/validation.py` computes geometric descriptors only (contact counts, buried area estimates); they are not folding or binding predictions and should not be read as any.

An earlier version of this README reported pLDDT and ipTM values for designed PD-L1 binders. Those numbers came from hand-written formulas, not from any structure predictor, and have been removed.

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
│   ├── evaluate.py             # Held-out scoring with intervals over structures
│   ├── target.py               # Target PDB parsing & epitope detection
│   ├── backbone.py             # 3D mini-protein scaffold generation & SE(3) docking
│   ├── mpnn.py                 # Sequence design & inverse folding
│   ├── validation.py           # Geometric descriptors only (see section 3)
│   └── pipeline.py             # Campaign orchestration & PDB export
├── vendor/
│   └── ProteinMPNN-main/       # Dauparas et al. ProteinMPNN architecture
├── results/
│   ├── mpnn_training_curves.png       # Loss & recovery across 40 epochs
│   ├── benchmark_comparison.png       # Baselines vs Trained vs Pretrained
│   ├── biochemical_class_recovery.png # Recovery by microenvironment
│   ├── training_summary.json          # Machine-readable benchmark report
│   ├── evaluation.json                # Per-structure recovery, intervals, deposition years
│   ├── mpnn_trained_weights.pt        # Trained PyTorch model weights
│   └── designs/                       # Candidate sequences (not validated, see section 3)
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

### 3. Design sequences for a target backbone
```bash
python scripts/run_design.py --num-backbones 15 --seqs-per-backbone 2
```
This emits candidate sequences and geometric descriptors. It does not validate them: see section 3.

---

## Citations & References
* **ProteinMPNN:** Dauparas et al., "Robust deep learning–based protein sequence design using ProteinMPNN", *Science* 378, 49–56 (2022).
* **Human PD-1 / PD-L1 Interface:** Zak et al., "Structure of the Complex of Human Programmed Death 1, PD-1, and Its Ligand, PD-L1", *Structure* 23, 2341–2348 (2015).
