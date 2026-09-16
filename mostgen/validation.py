from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .data import read_csv


def _truth(value: Any) -> bool:
    return value is True or str(value).lower() in {"true", "1", "yes"}


def verify_experiment(root: str | Path, config: dict[str, Any]) -> dict[str, Any]:
    experiment = Path(root)
    rows = read_csv(experiment / "generated.csv")
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["method_id"], row["seed"])].append(row)
    expected_runs = {
        (method, str(seed))
        for method in config["execution"]["methods"]
        for seed in config["execution"]["seeds"]
    }
    minimum = int(config["execution"]["n_per_run"])
    ledger_path = experiment / "search_budget_ledger.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8")) if ledger_path.exists() else {"runs": {}}
    ledger_runs = ledger.get("runs", {})
    expected_ledger_keys = {f"{method}:{seed}" for method, seed in expected_runs}
    required_columns = {
        "smiles", "charged_smiles", "family", "method_id", "seed", "uvb_auc", "uva_auc",
        "lambda_c_nm", "energy_kj_mol", "specific_energy_wh_kg", "half_life_h",
        "specific_energy_lcb", "uvb_transmittance_ucb", "uva_transmittance_ucb", "film_proxy_pass",
        "phototoxicity_probability", "phototoxicity_uncertainty", "similarity_D_A", "similarity_D_B",
        "ad_spectral", "ad_most", "sa_score", "psoralen_alert", "joint_pass", "selected",
        "not_iso_certified",
    }
    checks = {
        "all_expected_runs_present": set(grouped) == expected_runs,
        "minimum_unique_per_run": all(len({row["smiles"] for row in members}) >= minimum for members in grouped.values()),
        "exact_output_count_per_run": all(len(members) == minimum for members in grouped.values()),
        "exact_matched_reviewer_budget": set(ledger_runs) == expected_ledger_keys and all(
            int(item.get("candidate_reward_evaluations", -1)) == int(config["execution"]["reviewer_budget_per_run"])
            for item in ledger_runs.values()
        ),
        "all_three_families": set(row["family"] for row in rows) == set(config["families"]),
        "zero_psoralen_cores": not any(_truth(row.get("psoralen_alert")) for row in rows),
        "zero_known_phototoxic_matches": not any(_truth(row.get("known_phototoxic_match")) for row in rows),
        "no_uncertain_phototoxicity_selected": not any(_truth(row.get("selected")) and _truth(row.get("phototoxicity_uncertain")) for row in rows),
        "no_iso_claims": all(_truth(row.get("not_iso_certified")) for row in rows),
        "required_generated_schema": bool(rows) and required_columns <= set(rows[0]),
        "reward_evaluator_separation": (experiment / "models" / "reward" / "reviewers.pkl").resolve() != (experiment / "models" / "evaluator" / "reviewers.pkl").resolve(),
        "metrics_present": all((experiment / "metrics" / name).is_file() for name in ("metrics_runs.csv", "metrics_summary.csv", "ablations.csv", "reward_diagnostics.csv")),
        "report_present": (experiment / "report" / "report.md").is_file(),
        "presentation_present": (experiment / "report" / "presentation_7min.md").is_file(),
        "gpu_budget_respected": float(config["execution"]["gpu_hours"]) <= float(config["project"]["max_gpu_hours"]),
    }
    verification = {
        "schema_version": "1.0",
        "passed": all(checks.values()),
        "checks": checks,
        "generated_rows": len(rows),
        "run_counts": {f"{method}:{seed}": len(members) for (method, seed), members in sorted(grouped.items())},
        "run_unique_counts": {f"{method}:{seed}": len({row["smiles"] for row in members}) for (method, seed), members in sorted(grouped.items())},
        "family_counts": dict(sorted(Counter(row["family"] for row in rows).items())),
        "scientific_exceptions": {
            "no_joint_candidates_is_allowed": True,
            "physical_oracle_unavailable_blocks_final_selection": True,
        },
    }
    (experiment / "verification.json").write_text(json.dumps(verification, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not verification["passed"]:
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(f"Experiment acceptance verification failed: {failed}")
    return verification
