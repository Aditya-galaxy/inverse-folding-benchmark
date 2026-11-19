"""Unit and integration tests for De Novo Protein Binder Design."""

import os
import sys
import numpy as np

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.target import TargetHandler
from src.backbone import BackboneGenerator
from src.mpnn import ProteinMPNNDesigner
from src.validation import ComplexValidator
from src.pipeline import DeNovoBinderPipeline


def test_target_handler():
    handler = TargetHandler("data/4ZQK.pdb", target_chain="A")
    assert len(handler.residues) > 50
    assert len(handler.ca_coords) == len(handler.res_ids)

    # Test hotspot identification
    hotspots = handler.find_hotspots_from_complex(partner_chain="B", cutoff=4.5)
    assert len(hotspots) > 5
    hotspot_nums = [h["res_num"] for h in hotspots]
    assert 56 in hotspot_nums or 123 in hotspot_nums  # Known PD-L1 interface residues


def test_backbone_generator():
    gen = BackboneGenerator(seed=42)
    scaffold = gen.generate_three_helix_bundle(helix_lens=(18, 16, 18), loop_len=3)
    assert scaffold.length == (18 + 16 + 18 + 6)
    assert scaffold.ca_coords.shape == (scaffold.length, 3)

    # Verify minimum inter-residue bond distances
    for i in range(scaffold.length - 1):
        d = np.linalg.norm(scaffold.ca_coords[i + 1] - scaffold.ca_coords[i])
        # Consecutive C-alphas should be roughly 3.8 Å apart
        assert 1.0 < d < 6.5


def test_mpnn_designer():
    designer = ProteinMPNNDesigner(seed=42)
    gen = BackboneGenerator(seed=42)
    scaffold = gen.generate_three_helix_bundle()

    target_coords = np.random.randn(20, 3) * 10.0
    target_names = ["TYR"] * 10 + ["ASP"] * 10

    seq, roles = designer.design_sequence(scaffold, target_coords, target_names, temperature=0.2)
    assert len(seq) == scaffold.length
    assert len(roles) == scaffold.length
    assert all(aa in "ACDEFGHIKLMNPQRSTVWY" for aa in seq)


def test_validator():
    validator = ComplexValidator()
    gen = BackboneGenerator(seed=42)
    scaffold = gen.generate_three_helix_bundle()
    target_ca = np.random.randn(30, 3) * 15.0
    hotspot_ca = target_ca[:5]

    seq = "A" * scaffold.length
    roles = ["core"] * 20 + ["surface"] * 20 + ["interface"] * (scaffold.length - 40)

    metrics = validator.evaluate_complex(scaffold, target_ca, hotspot_ca, seq, roles)
    assert "plddt" in metrics
    assert "iptm" in metrics
    assert "delta_sasa" in metrics
    assert 0.0 <= metrics["iptm"] <= 1.0
    assert 0.0 <= metrics["plddt"] <= 100.0


def test_end_to_end_small():
    pipeline = DeNovoBinderPipeline(target_pdb="data/4ZQK.pdb", target_chain="A", seed=42)
    lead = pipeline.run_campaign(
        hotspot_res_nums=[56, 66, 115, 122, 123, 124],
        num_backbones=2,
        seqs_per_backbone=1,
        output_dir="results/test_run",
    )
    assert lead is not None
    assert "sequence" in lead
    assert os.path.exists("results/test_run/top_lead_complex.pdb")
    assert os.path.exists("results/test_run/design_summary.json")


if __name__ == "__main__":
    print("Running test_target_handler...")
    test_target_handler()
    print("Running test_backbone_generator...")
    test_backbone_generator()
    print("Running test_mpnn_designer...")
    test_mpnn_designer()
    print("Running test_validator...")
    test_validator()
    print("Running test_end_to_end_small...")
    test_end_to_end_small()
    print("All tests passed successfully!")
