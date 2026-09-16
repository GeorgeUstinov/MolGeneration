from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from .data import read_csv, write_csv
from .experimental import write_experimental_package
from .oracle import prepare_oracle_queue
from .reviewers import ReviewerBundle
from .scoring import ScoringContext


def _truth(value: Any) -> bool:
    return value is True or str(value).lower() in {"true", "1", "yes"}


def _balanced_top(rows: list[dict[str, str]], count: int) -> list[dict[str, str]]:
    deduplicated: dict[str, dict[str, str]] = {}
    for row in sorted(rows, key=lambda item: -float(item.get("reward", 0.0))):
        deduplicated.setdefault(row["smiles"], row)
    by_family: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in deduplicated.values():
        if not _truth(row.get("safety_veto")):
            by_family[row["family"]].append(row)
    families = sorted(by_family)
    base, remainder = divmod(count, max(1, len(families)))
    chosen = []
    for index, family in enumerate(families):
        chosen.extend(by_family[family][: base + (1 if index < remainder else 0)])
    if len(chosen) < count:
        used = {row["smiles"] for row in chosen}
        extras = [row for row in sorted(deduplicated.values(), key=lambda item: -float(item.get("reward", 0.0))) if row["smiles"] not in used]
        chosen.extend(extras[: count - len(chosen)])
    return chosen[:count]


def _card(row: dict[str, Any], oracle_status: str) -> dict[str, Any]:
    try:
        spectrum = json.loads(row.get("spectrum_json", "[]"))
    except json.JSONDecodeError:
        spectrum = []
    return {
        "candidate": {
            "candidate_id": row.get("candidate_id"), "smiles": row["smiles"],
            "charged_smiles": row["charged_smiles"], "family": row["family"],
            "method_id": row["method_id"], "seed": row["seed"],
        },
        "spectrum": {
            "curve": spectrum, "uvb_auc": row.get("uvb_auc"), "uva_auc": row.get("uva_auc"),
            "lambda_c_proxy_nm": row.get("lambda_c_nm"), "uncertainty": {
                "uvb_auc": row.get("uvb_auc_uncertainty"), "uva_auc": row.get("uva_auc_uncertainty"),
                "lambda_c": row.get("lambda_c_uncertainty"),
            },
            "conditional_film_proxy": {
                "loading_scale": row.get("beer_lambert_loading_scale"),
                "uvb_transmittance": row.get("uvb_transmittance"),
                "uvb_transmittance_ucb": row.get("uvb_transmittance_ucb"),
                "uva_transmittance": row.get("uva_transmittance"),
                "uva_transmittance_ucb": row.get("uva_transmittance_ucb"),
                "pass": row.get("film_proxy_pass"),
            },
        },
        "most": {
            "delta_h_kj_mol": row.get("energy_kj_mol"), "specific_energy_wh_kg": row.get("specific_energy_wh_kg"),
            "specific_energy_lcb_wh_kg": row.get("specific_energy_lcb"),
            "half_life_h_at_305k": row.get("half_life_h"), "uncertainty": {
                "delta_h": row.get("energy_uncertainty"), "specific_energy": row.get("specific_energy_uncertainty"),
                "log_half_life": row.get("half_life_uncertainty_log"),
            },
        },
        "safety_triage": {
            "kp_log_cm_s": row.get("kp_log_cm_s"), "phototoxicity_probability": row.get("phototoxicity_probability"),
            "phototoxicity_uncertain": row.get("phototoxicity_uncertain"), "psoralen_alert": row.get("psoralen_alert"),
            "known_phototoxic_match": row.get("known_phototoxic_match"), "reactive_alerts": row.get("reactive_alerts"),
            "sa_score_proxy": row.get("sa_score"),
            "sensitization_probability": row.get("sensitization_probability"),
            "irritation_probability": row.get("irritation_probability"),
        },
        "applicability_domain": {
            "spectral_similarity_D_A": row.get("similarity_D_A"), "most_similarity_D_B": row.get("similarity_D_B"),
            "spectral_inside": row.get("ad_spectral"), "most_inside": row.get("ad_most"),
        },
        "decision": {
            "uv_pass": row.get("joint_uv_pass"), "most_pass": row.get("most_pass"),
            "safety_pass": row.get("safety_pass"), "joint_pass": row.get("joint_pass"),
            "failure_reasons": row.get("failure_reasons"), "physical_oracle": oracle_status,
            "selected": row.get("selected", False),
        },
        "evidence": {
            "reviewer": "independent evaluator, separate from reward models",
            "data_tier": row.get("reviewer_data_mode", "synthetic_smoke_only"),
            "endpoint_evidence": row.get("evidence_json", "{}"),
            "physical_oracle": oracle_status,
            "claim_level": "research screening candidate only",
            "not_iso_certified": True,
            "experimental_phototoxicity_required": "OECD TG 432 or suitable successor method",
        },
    }


def review_generated(
    config: dict[str, Any],
    generated_path: str | Path,
    evaluator_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    generated = read_csv(generated_path)
    top = _balanced_top(generated, int(config["execution"]["shortlist_size"]))
    evaluators = ReviewerBundle.load(evaluator_path)
    context = ScoringContext(config, evaluators)
    reviewed = []
    for row in top:
        evaluated = context.review_candidate(row, row["method_id"], int(row["seed"]))
        evaluated["reward_model_reward"] = float(row.get("reward", 0.0))
        evaluated["preselection_rank"] = len(reviewed) + 1
        reviewed.append(evaluated)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    oracle_manifest = prepare_oracle_queue(reviewed, destination / "physical_oracle", config)
    oracle_complete = oracle_manifest["status"] == "complete"
    oracle_results_path = destination / "physical_oracle" / "oracle_results.csv"
    oracle_by_id = {row["candidate_id"]: row for row in read_csv(oracle_results_path)} if oracle_results_path.exists() else {}
    for row in reviewed:
        physical = oracle_by_id.get(row.get("candidate_id"), {})
        row.update(physical)
        row["physical_oracle_pass"] = bool(
            physical
            and float(physical.get("gfn2_delta_e_kj_mol", 0.0)) > 0.0
            and float(physical.get("gfn2_specific_energy_wh_kg", 0.0)) >= float(config["reviewers"].get("specific_energy_min_wh_kg", 0.0))
            and float(physical.get("stda_lambda_c_nm", 0.0)) >= float(config["reviewers"]["lambda_c_min_nm"])
            and float(physical.get("stda_uvb_transmittance", 1.0)) <= float(config["reviewers"].get("uvb_transmittance_max", 1.0))
            and float(physical.get("stda_uva_transmittance", 1.0)) <= float(config["reviewers"].get("uva_transmittance_max", 1.0))
        )
    provisional = [
        row for row in reviewed
        if _truth(row.get("joint_pass"))
        and not _truth(row.get("phototoxicity_uncertain"))
        and not _truth(row.get("psoralen_alert"))
        and not _truth(row.get("known_phototoxic_match"))
    ]
    selected = [row for row in provisional if _truth(row.get("physical_oracle_pass"))] if oracle_complete else []
    selected_keys = {(row["smiles"], row["method_id"], str(row["seed"])) for row in selected}
    for row in reviewed:
        row["physical_oracle_status"] = oracle_manifest["status"]
        row["selected"] = (row["smiles"], row["method_id"], str(row["seed"])) in selected_keys
        row["selection_status"] = "selected_research_candidate" if row["selected"] else (
            "provisional_pending_physical_oracle" if row in provisional else "failed_independent_review"
        )
    for row in generated:
        row["selected"] = (row["smiles"], row["method_id"], str(row["seed"])) in selected_keys
    write_csv(generated_path, generated)
    write_csv(destination / "reviewed_top.csv", reviewed)
    write_csv(destination / "provisional_shortlist.csv", provisional, fieldnames=list(reviewed[0]) if reviewed else [])
    write_csv(destination / "shortlist.csv", selected, fieldnames=list(reviewed[0]) if reviewed else [])
    experimental_package = write_experimental_package(
        destination / "shortlist.csv", destination / "experimental_package",
    )
    cards_dir = destination / "cards"
    cards_dir.mkdir(parents=True, exist_ok=True)
    for row in reviewed:
        card = _card(row, oracle_manifest["status"])
        (cards_dir / f"{row['candidate_id']}.json").write_text(json.dumps(card, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = {
        "reviewed": len(reviewed), "independent_joint_pass": len(provisional),
        "selected_after_physical_oracle": len(selected), "physical_oracle": oracle_manifest,
        "experimental_package": experimental_package,
        "selection_policy": "Fail closed: no final selection until independent physical oracle results are complete.",
    }
    (destination / "review_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary
