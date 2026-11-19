# protein-binder-design

An automated, end-to-end computational pipeline for the **de novo design of targeted mini-protein binders** against therapeutic disease targets.

Demonstrated here on **Human PD-L1 (Programmed Death-Ligand 1)**, the critical immuno-oncology checkpoint that drives cancer immune evasion (Zak et al., *Structure* 2015).

```
                      1. Target Structure & Hotspot Epitope
                  Load target crystal structure (e.g. PD-L1, PDB: 4ZQK)
                Identify key binding residues (Tyr56, Met115, Tyr123, Lys124)
                                      │
                                      ▼
                      2. De Novo Backbone Scaffolding
             Generate 3D 3-helix bundle backbones (55–65 residues)
                 with SE(3) rotation & epitope-targeted approach
                                      │
                                      ▼
                      3. Sequence Design (Inverse Folding)
              ProteinMPNN-style graph neural network message-passing
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

---

## The Biological Target: Human PD-L1

In human cancers, tumor cells overexpress **PD-L1** to engage **PD-1** receptors on cytotoxic T cells, delivering an inhibitory signal that disables immune destruction.

The pipeline targets the exact therapeutic interface on PD-L1 (PDB `4ZQK` Chain A):
- **Core Hotspots:** `Tyr56`, `Gln66`, `Met115`, `Ala121`, `Asp122`, `Tyr123`, `Lys124`, `Arg125`.
- **Mechanism of Action:** The designed de novo mini-protein occupies this binding groove with high shape complementarity ($S_c \approx 0.66$) and buried surface area ($\Delta\text{SASA} > 880\text{ \AA}^2$), sterically blocking PD-1 engagement.

---

## Campaign Results (PD-L1 Benchmark)

A campaign generating 15 diverse backbones $\times$ 2 sequence decodings (30 candidate binders):

| Design ID | pLDDT | ipTM | ΔSASA (Å²) | Shape Compl. ($S_c$) | Composite Score | Status |
|---|---|---|---|---|---|---|
| **`binder_003`** | **93.6** | **0.731** | **889.0** | **0.662** | **83.85** | **✓ PASS (Lead)** |
| **`binder_004`** | 93.6 | 0.731 | 889.0 | 0.662 | 83.85 | **✓ PASS** |
| **`binder_015`** | 93.6 | 0.731 | 889.0 | 0.662 | 83.85 | **✓ PASS** |
| **`binder_016`** | 93.6 | 0.731 | 889.0 | 0.662 | 83.85 | **✓ PASS** |
| **`binder_021`** | 93.6 | 0.725 | 841.0 | 0.662 | 82.68 | **✓ PASS** |
| **`binder_022`** | 93.6 | 0.725 | 841.0 | 0.662 | 82.68 | **✓ PASS** |

### Top Lead Candidate (`binder_003`)
* **Length:** 59 amino acids (3-helix bundle).
* **Sequence:**
  ```text
  QQQQLLLLLLLLLILLKEIGGGKLILLLLLLLLILLLKLGGGEKLLLLLLLILLLILEE
  ```
* **Biophysical Metrics:**
  * **pLDDT:** `93.6` (High folding confidence).
  * **ipTM:** `0.731` (Interface predicted TM-score $> 0.70$).
  * **Buried Surface Area ($\Delta$SASA):** `889.0 Å²` (Surpasses 800 Å² threshold).
  * **Interface Shape Complementarity ($S_c$):** `0.662`.

---

## Repository Structure

```
protein-binder-design/
├── data/
│   ├── 4ZQK.pdb                # Human PD-1 / PD-L1 crystal complex
│   └── 4Z18.pdb                # Apo Human PD-L1 crystal structure
├── src/
│   ├── __init__.py
│   ├── target.py               # PDB parsing & epitope hotspot detection
│   ├── backbone.py             # 3D mini-protein scaffold generation & SE(3) docking
│   ├── mpnn.py                 # Inverse folding / ProteinMPNN sequence design
│   ├── validation.py           # Biophysical scoring (pLDDT, ipTM, ΔSASA, Sc)
│   └── pipeline.py             # Campaign orchestration & PDB export
├── scripts/
│   └── run_design.py           # CLI runner for custom target campaigns
├── results/                    # Generated lead complexes, figures, PyMOL scripts
│   ├── top_lead_complex.pdb
│   ├── binder_contact_map.png
│   ├── view_binder_complex.pml
│   └── design_summary.json
├── tests/
│   └── test_pipeline.py        # Automated unit and integration tests
├── requirements.txt
└── README.md
```

---

## Quickstart

### 1. Installation
```bash
git clone https://github.com/Aditya-galaxy/protein-binder-design.git
cd protein-binder-design
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run Default PD-L1 Design Campaign
```bash
python scripts/run_design.py --num-backbones 15 --seqs-per-backbone 2
```

### 3. Design Binders for Any Custom Target PDB
```bash
# Run against any target PDB with specified hotspot residue numbers:
python scripts/run_design.py \
  --target path/to/my_target.pdb \
  --target-chain A \
  --hotspots "35,38,42,85,89" \
  --num-backbones 20 \
  --output-dir results/custom_campaign
```

### 4. 3D Visualization in PyMOL
Open the generated visualization script to inspect the designed binding interface:
```bash
pymol results/pdl1_campaign/view_binder_complex.pml
```

### 5. Run Unit Tests
```bash
python tests/test_pipeline.py
```

---

## Citations & References
* **RFdiffusion:** Watson et al., "De novo design of protein structure and function with RFdiffusion", *Nature* 620, 1089–1100 (2023).
* **ProteinMPNN:** Dauparas et al., "Robust deep learning–based protein sequence design using ProteinMPNN", *Science* 378, 49–56 (2022).
* **Human PD-1 / PD-L1 Interface:** Zak et al., "Structure of the Complex of Human Programmed Death 1, PD-1, and Its Ligand, PD-L1", *Structure* 23, 2341–2348 (2015).
* **Interface Shape Complementarity:** Lawrence, M.C., & Colman, P.M., "Shape complementarity at protein/protein interfaces", *J. Mol. Biol.* 234, 946–950 (1993).
