from __future__ import annotations

from pathlib import Path

from mostgen.phototoxicity import fit_phototoxicity_ensemble, load_experimental_phototoxicity


ROOT = Path(__file__).resolve().parents[1]


def test_qdb_workbook_supplies_all_structures_and_binary_labels():
    frame = load_experimental_phototoxicity(
        ROOT / "datasets" / "QDB_2011TIV324_phototoxicity_normalized.xlsx",
        ROOT / "datasets" / "PhotoChem_phototoxicity_251_verified.xlsx",
    )
    assert len(frame) == 53
    assert frame.smiles.nunique() == 53
    assert set(frame.phototoxic) == {0, 1}
    assert frame.photochem_label.notna().sum() >= 25


def test_ensemble_returns_bounded_probabilities_and_exact_positive_veto():
    frame = load_experimental_phototoxicity(
        ROOT / "datasets" / "QDB_2011TIV324_phototoxicity_normalized.xlsx"
    )
    model = fit_phototoxicity_ensemble(frame, members=2, trees_per_member=12, seed=7)
    positive = frame.loc[frame.phototoxic.eq(1), "smiles"].iloc[0]
    result = model.predict([positive, "CCO"])
    assert result.phototoxic_probability.between(0, 1).all()
    assert result.phototoxic_upper_confidence.between(0, 1).all()
    assert bool(result.loc[0, "known_phototoxic_match"])
    assert not bool(result.loc[0, "toxicity_pass"])
