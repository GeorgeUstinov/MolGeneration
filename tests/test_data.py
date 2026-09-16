from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

from mostgen.data import read_csv
from mostgen.real_data import parse_m13, parse_u07_irritation, parse_u13_permeation


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_m13_preserves_m11_m12_constituent_provenance(tmp_path):
    import pandas as pd

    path = tmp_path / "uvvisml.csv"
    pd.DataFrame([
        {"smiles": "CCO", "peakwavs_max": 310.0, "solvent": "water", "source": "deep4chem"},
        {"smiles": "CCN", "peakwavs_max": 350.0, "solvent": "water", "source": "cdex"},
        {"smiles": "CCC", "peakwavs_max": 390.0, "solvent": "water", "source": "chemfluor"},
    ]).to_csv(path, index=False)
    rows = parse_m13(path)
    assert {row["original_source"]: row["source_id"] for row in rows} == {
        "deep4chem": "M11", "cdex": "M12", "chemfluor": "M13",
    }


def test_training_data_has_provenance_units_and_semantics(smoke_experiment):
    rows = read_csv(smoke_experiment["root"] / "data" / "reviewer_training.csv")
    required = {
        "source_id", "source_kind", "evidence_tier", "state", "temperature_k",
        "medium", "label_level", "energy_kj_mol", "specific_energy_wh_kg",
        "log_half_life_h", "kp_log_cm_s", "phototoxicity_probability",
    }
    assert required <= set(rows[0])
    assert {row["evidence_tier"] for row in rows} == {"synthetic_smoke_only"}
    assert all(float(row["temperature_k"]) == 305.0 for row in rows)


def test_no_duplicate_molecule_identity(smoke_experiment):
    rows = read_csv(smoke_experiment["root"] / "data" / "reviewer_training.csv")
    keys = [(row["family"], row["smiles"]) for row in rows]
    assert len(keys) == len(set(keys))


def test_no_scaffold_leakage(smoke_experiment):
    rows = read_csv(smoke_experiment["root"] / "data" / "reviewer_training.csv")
    split_scaffolds = defaultdict(set)
    for row in rows:
        split_scaffolds[row["split"]].add(row["scaffold"])
    names = list(split_scaffolds)
    for index, first in enumerate(names):
        for second in names[index + 1:]:
            assert not (split_scaffolds[first] & split_scaffolds[second])


def test_raw_files_were_only_hashed(smoke_experiment):
    manifest = json.loads((smoke_experiment["root"] / "data" / "data_manifest.json").read_text())
    assert manifest["raw_inputs_mutated"] is False
    assert all(len(item["sha256"]) == 64 for item in manifest["local_inputs"])


def test_spectrum_grid_is_complete(smoke_experiment):
    rows = read_csv(smoke_experiment["root"] / "data" / "reviewer_training.csv")
    expected = {f"abs_{w}" for w in range(290, 401, 5)}
    assert expected <= set(rows[0])


def test_u07_irritation_uses_only_exact_dtxsid_structure_join():
    rows = parse_u07_irritation(
        PROJECT_ROOT / "data/raw/U07_nice/Skin_Irritation_Corrosion.xlsx",
        PROJECT_ROOT / "data/raw/U09_hppt/hppt_database_14feb2023.xlsx",
    )
    assert len(rows) == 763
    assert len({row["smiles"] for row in rows}) == 48
    assert {float(row["target"]) for row in rows} == {0.0, 1.0}
    assert all(row["dtxsid"] for row in rows)


def test_u13_kp_is_explicitly_converted_from_cm_per_hour_to_cm_per_second():
    rows = parse_u13_permeation(
        PROJECT_ROOT / "data/raw/U13_mendeley/Table_1.xlsx",
        PROJECT_ROOT / "data/raw/U09_hppt/hppt_database_14feb2023.xlsx",
        PROJECT_ROOT / "data/raw/U12_skinpix/20230620_cleanedDB.xlsx",
    )
    assert len(rows) == 154
    assert len({row["smiles"] for row in rows}) == 56
    for row in rows:
        assert float(row["target"]) == float(row["original_value"]) - math.log10(3600.0)
        assert row["original_unit"] == "log10(cm/h)"
