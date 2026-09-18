from __future__ import annotations

import pytest
from rdkit import DataStructs

from mostgen.chemistry import fingerprint
from mostgen.metrics import generative_quality_metrics


def test_generative_quality_metrics_use_requested_denominators() -> None:
    rows = [
        {"smiles": "CC", "valid": True, "target_a": True, "target_b": True},
        {"smiles": "CC", "valid": True, "target_a": True, "target_b": True},
        {"smiles": "CCC", "valid": True, "target_a": True, "target_b": False},
        {"smiles": "not-smiles", "valid": False, "target_a": True, "target_b": True},
    ]

    metrics = generative_quality_metrics(
        rows,
        {"CC"},
        2048,
        joint_a_key="target_a",
        joint_b_key="target_b",
    )

    assert metrics["n_generated"] == 4
    assert metrics["n_valid"] == 3
    assert metrics["n_unique"] == 2
    assert metrics["n_not_in_training"] == 1
    assert metrics["n_joint_success"] == 2
    assert metrics["validity"] == pytest.approx(3 / 4)
    assert metrics["uniqueness"] == pytest.approx(2 / 3)
    assert metrics["novelty"] == pytest.approx(1 / 2)
    assert metrics["joint_success_rate"] == pytest.approx(2 / 4)


def test_diversity_is_exact_pairwise_tanimoto_distance() -> None:
    smiles = ["CC", "CCC", "c1ccccc1"]
    rows = [{"smiles": value, "valid": True} for value in smiles]
    fps = [fingerprint(value, 2048) for value in smiles]
    similarities = [
        DataStructs.TanimotoSimilarity(fps[left], fps[right])
        for left in range(len(fps))
        for right in range(left + 1, len(fps))
    ]
    expected = 1.0 - sum(similarities) / len(similarities)

    metrics = generative_quality_metrics(rows, set(), 2048)

    assert metrics["diversity"] == pytest.approx(expected)
