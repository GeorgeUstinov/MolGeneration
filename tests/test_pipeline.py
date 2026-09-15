from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

from mostgen.data import read_csv
from mostgen.reviewers import ReviewerBundle


def _truth(value):
    return str(value).lower() in {"true", "1"}


def test_three_methods_use_exact_matched_budget(smoke_experiment):
    rows = read_csv(smoke_experiment["root"] / "generated.csv")
    counts = Counter((row["method_id"], row["seed"]) for row in rows)
    assert set(method for method, _ in counts) == {"prior_random", "weighted_retraining", "libinvent_rl"}
    assert set(counts.values()) == {smoke_experiment["config"]["execution"]["reviewer_budget_per_run"]}
    assert smoke_experiment["metrics"]["matched_reviewer_budget"]


def test_generated_schema_and_uniqueness(smoke_experiment):
    rows = read_csv(smoke_experiment["root"] / "generated.csv")
    required = {
        "smiles", "charged_smiles", "family", "method_id", "seed", "uvb_auc",
        "uva_auc", "lambda_c_nm", "energy_kj_mol", "specific_energy_wh_kg",
        "half_life_h", "kp_log_cm_s", "phototoxicity_probability",
        "uvb_auc_uncertainty", "energy_uncertainty", "similarity_D_A",
        "similarity_D_B", "ad_spectral", "ad_most", "sa_score",
        "psoralen_alert", "joint_pass", "selected", "not_iso_certified",
    }
    assert required <= set(rows[0])
    for key in Counter((row["method_id"], row["seed"]) for row in rows):
        members = [row for row in rows if (row["method_id"], row["seed"]) == key]
        assert len(members) == len({row["smiles"] for row in members})


def test_hard_psoralen_acceptance_gate(smoke_experiment):
    rows = read_csv(smoke_experiment["root"] / "generated.csv")
    assert not any(_truth(row["psoralen_alert"]) for row in rows)
    shortlist = read_csv(smoke_experiment["root"] / "review" / "shortlist.csv")
    assert not any(_truth(row["psoralen_alert"]) for row in shortlist)


def test_reward_is_dense_and_diversity_diagnostics_exist(smoke_experiment):
    rows = read_csv(smoke_experiment["root"] / "generated.csv")
    assert all(float(row["reward_pre_diversity"]) > 0.0 for row in rows)
    diagnostics = read_csv(smoke_experiment["root"] / "metrics" / "reward_diagnostics.csv")
    assert all(float(row["nonzero_fraction"]) > 0.95 for row in diagnostics)
    assert all(float(row["effective_sample_size"]) > 0 for row in diagnostics)


def test_family_coverage(smoke_experiment):
    rows = read_csv(smoke_experiment["root"] / "generated.csv")
    for method in {row["method_id"] for row in rows}:
        assert {row["family"] for row in rows if row["method_id"] == method} == {
            "nbd_qc", "dewar_pyrimidinone", "spiropyran"
        }


def test_reward_and_evaluator_models_are_independent(smoke_experiment):
    root = smoke_experiment["root"]
    reward = ReviewerBundle.load(root / "models" / "reward" / "reviewers.pkl")
    evaluator = ReviewerBundle.load(root / "models" / "evaluator" / "reviewers.pkl")
    assert reward.purpose == "reward"
    assert evaluator.purpose == "evaluator"
    assert type(reward.spectrum_models[0]) is not type(evaluator.spectrum_models[0])


def test_fail_closed_physical_oracle(smoke_experiment):
    summary = smoke_experiment["review"]
    if summary["physical_oracle"]["status"] != "complete":
        assert summary["selected_after_physical_oracle"] == 0
        rows = read_csv(smoke_experiment["root"] / "generated.csv")
        assert not any(_truth(row["selected"]) for row in rows)


def test_metrics_ablations_cards_and_reports_exist(smoke_experiment):
    root = smoke_experiment["root"]
    for relative in (
        "metrics/metrics_runs.csv", "metrics/metrics_summary.csv", "metrics/ablations.csv",
        "review/reviewed_top.csv", "review/provisional_shortlist.csv",
        "review/physical_oracle/oracle_queue.csv", "report/report.md", "report/presentation_7min.md",
    ):
        assert (root / relative).exists(), relative
    cards = list((root / "review" / "cards").glob("*.json"))
    assert len(cards) == smoke_experiment["config"]["execution"]["shortlist_size"]
    card = json.loads(cards[0].read_text())
    assert {"spectrum", "most", "safety_triage", "applicability_domain", "decision", "evidence"} <= set(card)
    assert len((root / "report" / "report.md").read_text().split()) >= 1_400


def test_reinvent_handoff_configs_are_valid_toml(smoke_experiment):
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib
    configs = sorted((smoke_experiment["root"] / "generator").glob("*.toml"))
    assert len(configs) == 3
    for path in configs:
        with path.open("rb") as handle:
            parsed = tomllib.load(handle)
        assert parsed["run_type"] == "staged_learning"
        assert len(parsed["stage"]) == 4
        assert parsed["parameters"]["prior_file"] == "PRIOR_PATH_REQUIRED"


def test_artifact_manifest_checksums(smoke_experiment):
    import hashlib
    root = smoke_experiment["root"]
    manifest = json.loads((root / "artifact_manifest.json").read_text())
    assert manifest["artifacts"]
    for item in manifest["artifacts"]:
        assert hashlib.sha256((root / item["path"]).read_bytes()).hexdigest() == item["sha256"]


def test_machine_readable_acceptance_verification(smoke_experiment):
    assert smoke_experiment["verification"]["passed"]
    assert all(smoke_experiment["verification"]["checks"].values())


def test_full_configuration_meets_acceptance_count():
    from mostgen.config import load_config
    config = load_config(mode="full")
    assert len(config["execution"]["seeds"]) == 3
    assert config["execution"]["n_per_run"] >= 1_000
    assert config["execution"]["reviewer_budget_per_run"] == config["execution"]["n_per_run"]


def test_iso_claim_is_always_negative(smoke_experiment):
    rows = read_csv(smoke_experiment["root"] / "generated.csv")
    assert all(_truth(row["not_iso_certified"]) for row in rows)
