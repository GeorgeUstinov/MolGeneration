from __future__ import annotations

import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any, Callable

from rdkit import DataStructs

from .chemistry import fingerprint, murcko_scaffold
from .data import read_csv, write_csv


def _truth(value: Any) -> bool:
    return value is True or str(value).lower() in {"true", "1", "yes"}


def _float(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def _internal_diversity(rows: list[dict[str, Any]], bits: int, seed: int = 701) -> float:
    unique = sorted({row["smiles"] for row in rows})
    if len(unique) < 2:
        return 0.0
    rng = random.Random(seed)
    pairs = []
    max_pairs = min(500, len(unique) * (len(unique) - 1) // 2)
    seen = set()
    while len(pairs) < max_pairs:
        left, right = sorted(rng.sample(range(len(unique)), 2))
        if (left, right) in seen:
            continue
        seen.add((left, right))
        pairs.append((left, right))
    fps = [fingerprint(smiles, bits) for smiles in unique]
    return fmean(1.0 - float(DataStructs.TanimotoSimilarity(fps[left], fps[right])) for left, right in pairs)


def _effective_sample_size(weights: list[float]) -> float:
    total = sum(weights)
    squares = sum(value * value for value in weights)
    return total * total / squares if squares else 0.0


def _run_metrics(rows: list[dict[str, Any]], training_smiles: set[str], bits: int) -> dict[str, Any]:
    n = len(rows)
    unique = {row["smiles"] for row in rows}
    valid = [row for row in rows if _truth(row.get("valid"))]
    scaffolds = {murcko_scaffold(row["smiles"]) for row in valid}
    families = Counter(row["family"] for row in valid)
    rewards = [_float(row, "reward") for row in rows]
    return {
        "evaluated": n,
        "validity": len(valid) / n if n else 0.0,
        "uniqueness": len(unique) / n if n else 0.0,
        "novelty": sum(row["smiles"] not in training_smiles for row in valid) / len(valid) if valid else 0.0,
        "internal_diversity": _internal_diversity(valid, bits),
        "scaffold_diversity": len(scaffolds) / len(valid) if valid else 0.0,
        "mean_sa_score": fmean(_float(row, "sa_score", 10.0) for row in valid) if valid else 10.0,
        "joint_success": sum(_truth(row.get("joint_pass")) for row in rows) / n if n else 0.0,
        "both_ad_fraction": sum(_truth(row.get("ad_spectral")) and _truth(row.get("ad_most")) for row in rows) / n if n else 0.0,
        "family_coverage": sum(1 for family in families if families[family] > 0) / 3.0,
        "nbd_qc_count": families["nbd_qc"],
        "dewar_pyrimidinone_count": families["dewar_pyrimidinone"],
        "spiropyran_count": families["spiropyran"],
        "nonzero_reward_fraction": sum(value > 0.0 for value in rewards) / n if n else 0.0,
        "effective_sample_size": _effective_sample_size(rewards),
        "unique_ecfp_clusters": len({row.get("ecfp_cluster") for row in rows}),
    }


def _bootstrap(values: list[float], samples: int, seed: int) -> tuple[float, float, float]:
    if not values:
        return math.nan, math.nan, math.nan
    if len(values) == 1:
        return values[0], values[0], values[0]
    rng = random.Random(seed)
    means = []
    for _ in range(samples):
        means.append(fmean(rng.choice(values) for _ in values))
    means.sort()
    low = means[max(0, int(0.025 * len(means)) - 1)]
    high = means[min(len(means) - 1, int(0.975 * len(means)))]
    return fmean(values), low, high


def _ablation_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["method_id"]].append(row)
    for method, members in sorted(grouped.items()):
        gates: dict[str, Callable[[dict[str, Any]], bool]] = {
            "full": lambda r: _truth(r.get("joint_pass")),
            "without_uncertainty_penalty": lambda r: (
                _float(r, "uvb_auc") >= _float(r, "reference_uvb_median")
                and _float(r, "uva_auc") >= _float(r, "reference_uva_median")
                and _float(r, "lambda_c_nm") >= 370.0
                and _float(r, "energy_kj_mol") >= _float(r, "reference_energy_median")
                and 4.0 <= _float(r, "half_life_h") <= 24.0
                and _truth(r.get("safety_pass")) and _truth(r.get("ad_spectral")) and _truth(r.get("ad_most"))
            ),
            "without_safety_gate": lambda r: _truth(r.get("joint_uv_pass")) and _truth(r.get("most_pass")) and _truth(r.get("ad_spectral")) and _truth(r.get("ad_most")),
            "without_ad_gate": lambda r: _truth(r.get("joint_uv_pass")) and _truth(r.get("most_pass")) and _truth(r.get("safety_pass")),
        }
        for setting, predicate in gates.items():
            passed = sum(predicate(row) for row in members)
            result.append({"method_id": method, "ablation": setting, "evaluated": len(members), "eligible": passed, "success_fraction": passed / len(members) if members else 0.0})
        top_adjusted = sorted(members, key=lambda row: -_float(row, "reward"))[: min(100, len(members))]
        top_raw = sorted(members, key=lambda row: -_float(row, "reward_pre_diversity"))[: min(100, len(members))]
        result.append({
            "method_id": method, "ablation": "without_diversity_filter",
            "evaluated": len(members), "eligible": sum(_truth(row.get("joint_pass")) for row in top_raw),
            "success_fraction": sum(_truth(row.get("joint_pass")) for row in top_raw) / len(top_raw) if top_raw else 0.0,
            "top100_cluster_diversity_full": len({row.get("ecfp_cluster") for row in top_adjusted}) / len(top_adjusted) if top_adjusted else 0.0,
            "top100_cluster_diversity_ablated": len({row.get("ecfp_cluster") for row in top_raw}) / len(top_raw) if top_raw else 0.0,
        })
    return result


def compute_metrics(
    generated_path: str | Path,
    training_path: str | Path,
    output_dir: str | Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    rows = read_csv(generated_path)
    training_smiles = {row["smiles"] for row in read_csv(training_path)}
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["method_id"], row["seed"])].append(row)
    run_rows = []
    bits = int(config["reviewers"]["fingerprint_bits"])
    for (method, seed), members in sorted(grouped.items()):
        run_rows.append({"method_id": method, "seed": int(seed), **_run_metrics(members, training_smiles, bits)})
    summary_rows = []
    metric_names = ("validity", "uniqueness", "novelty", "internal_diversity", "scaffold_diversity", "mean_sa_score", "joint_success", "both_ad_fraction", "family_coverage")
    for method in sorted({row["method_id"] for row in run_rows}):
        members = [row for row in run_rows if row["method_id"] == method]
        summary: dict[str, Any] = {"method_id": method, "seed_runs": len(members)}
        for metric in metric_names:
            mean, low, high = _bootstrap([float(row[metric]) for row in members], int(config["execution"]["bootstrap_samples"]), int(config["project"]["default_seed"]) ^ sum(map(ord, method + metric)))
            summary[f"{metric}_mean"] = mean
            summary[f"{metric}_ci_low"] = low
            summary[f"{metric}_ci_high"] = high
        summary_rows.append(summary)
    diagnostics = []
    for (method, seed), members in sorted(grouped.items()):
        component_values: dict[str, list[float]] = defaultdict(list)
        for row in members:
            try:
                parsed = json.loads(row.get("reward_components_json", "{}"))
            except json.JSONDecodeError:
                parsed = {}
            for key, value in parsed.items():
                component_values[key].append(float(value))
        rewards = [_float(row, "reward") for row in members]
        diagnostics.append({
            "method_id": method, "seed": seed,
            "nonzero_fraction": sum(value > 0 for value in rewards) / len(rewards) if rewards else 0.0,
            "effective_sample_size": _effective_sample_size(rewards),
            "largest_cluster_fraction": max(Counter(row.get("ecfp_cluster") for row in members).values()) / len(members) if members else 0.0,
            **{f"component_{key}_mean": fmean(values) for key, values in sorted(component_values.items())},
        })
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    write_csv(destination / "metrics_runs.csv", run_rows)
    write_csv(destination / "metrics_summary.csv", summary_rows)
    write_csv(destination / "ablations.csv", _ablation_rows(rows))
    write_csv(destination / "reward_diagnostics.csv", diagnostics)
    ledger_path = destination.parent / "search_budget_ledger.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8")) if ledger_path.exists() else {"runs": {}}
    call_counts = {
        int(item.get("candidate_reward_evaluations", -1))
        for item in ledger.get("runs", {}).values()
    }
    result = {
        "run_metrics": run_rows, "summary": summary_rows,
        "matched_final_set_size": len({len(group) for group in grouped.values()}) == 1,
        "matched_reviewer_budget": bool(call_counts) and len(call_counts) == 1,
        "reviewer_call_counts": {
            key: int(item.get("candidate_reward_evaluations", -1))
            for key, item in sorted(ledger.get("runs", {}).items())
        },
    }
    (destination / "metrics.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result
