from __future__ import annotations

import json
from collections import defaultdict

from mostgen.data import read_csv


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

