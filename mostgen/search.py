from __future__ import annotations

import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from rdkit import DataStructs

from .chemistry import fingerprint
from .data import read_csv, write_csv
from .provenance import stable_hash
from .reviewers import ReviewerBundle
from .scoring import ScoringContext


METHODS = ("prior_random", "weighted_retraining", "libinvent_rl")


def _family_quotas(total: int, families: list[str]) -> dict[str, int]:
    base, remainder = divmod(total, len(families))
    return {family: base + (1 if index < remainder else 0) for index, family in enumerate(families)}


def _weighted_choice(rng: random.Random, rows: list[dict[str, Any]], weights: list[float]) -> dict[str, Any]:
    total = sum(weights)
    if total <= 0.0:
        return rng.choice(rows)
    point = rng.random() * total
    cumulative = 0.0
    for row, weight in zip(rows, weights):
        cumulative += weight
        if cumulative >= point:
            return row
    return rows[-1]


def _diversity_clusters(rows: list[dict[str, Any]], bits: int, threshold: float = 0.58) -> None:
    centroids = []
    counts: Counter[int] = Counter()
    for row in rows:
        fp = fingerprint(row["smiles"], bits)
        cluster = None
        if centroids:
            similarities = DataStructs.BulkTanimotoSimilarity(fp, centroids)
            best = max(range(len(similarities)), key=similarities.__getitem__)
            if similarities[best] >= threshold:
                cluster = best
        if cluster is None:
            cluster = len(centroids)
            centroids.append(fp)
        prior_count = counts[cluster]
        counts[cluster] += 1
        row["ecfp_cluster"] = cluster
        row["cluster_prior_count"] = prior_count


def _apply_diversity_penalty(rows: list[dict[str, Any]], config: dict[str, Any]) -> None:
    _diversity_clusters(rows, int(config["reviewers"]["fingerprint_bits"]))
    penalty = float(config["reward"]["diversity_penalty"])
    for row in rows:
        base = float(row.get("reward_pre_diversity", 0.0))
        row["reward"] = base / (1.0 + penalty * int(row["cluster_prior_count"]))


def _prior_candidates(rng: random.Random, pool: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    return rng.sample(pool, min(count, len(pool)))


def _adaptive_candidates(
    method: str,
    rng: random.Random,
    pool: list[dict[str, Any]],
    count: int,
    context: ScoringContext,
    seed: int,
) -> list[dict[str, Any]]:
    remaining = list(pool)
    synthon_value: dict[str, float] = defaultdict(lambda: 0.5)
    synthon_visits: Counter[str] = Counter()
    selected: list[dict[str, Any]] = []
    warmup = min(max(8, count // 8), count)
    stage_names = [entry["name"] for entry in context.config["reward"]["stages"]]
    batch_size = min(24, max(4, count // 20))
    while remaining and len(selected) < count:
        pending = []
        pending_stages = []
        for _ in range(min(batch_size, count - len(selected), len(remaining))):
            progress = (len(selected) + len(pending)) / max(1, count)
            stage = stage_names[min(len(stage_names) - 1, int(progress * len(stage_names)))]
            if len(selected) + len(pending) < warmup:
                candidate = rng.choice(remaining)
            else:
                weights = []
                for row in remaining:
                    q = 0.5 * (synthon_value[row["synthon_a"]] + synthon_value[row["synthon_b"]])
                    if method == "weighted_retraining":
                        weight = 0.05 + q**3
                    else:
                        visits = 1 + synthon_visits[row["synthon_a"]] + synthon_visits[row["synthon_b"]]
                        bonus = math.sqrt(math.log(2 + len(selected) + len(pending)) / visits)
                        weight = 0.03 + math.exp(min(4.0, 2.2 * q + 0.35 * bonus))
                    weights.append(weight)
                candidate = _weighted_choice(rng, remaining, weights)
            remaining.remove(candidate)
            pending.append(candidate)
            pending_stages.append(stage)
        reviewed_batch = context.review_candidates(pending, method, seed)
        for reviewed, stage, candidate in zip(reviewed_batch, pending_stages, pending):
            reviewed["curriculum_stage"] = stage
            stage_rewards = json.loads(reviewed.get("stage_rewards_json", "{}") or "{}")
            signal = float(stage_rewards.get(stage, reviewed.get("reward_pre_diversity", 0.0)))
            for synthon in (candidate["synthon_a"], candidate["synthon_b"]):
                visits = synthon_visits[synthon]
                synthon_value[synthon] = (synthon_value[synthon] * visits + signal) / (visits + 1)
                synthon_visits[synthon] += 1
            selected.append(reviewed)
    return selected


def run_method(
    config: dict[str, Any],
    library_path: str | Path,
    reviewer_path: str | Path,
    method: str,
    seed: int,
    budget: int | None = None,
) -> list[dict[str, Any]]:
    if method not in METHODS:
        raise ValueError(f"Unknown search method: {method}")
    pool = read_csv(library_path)
    reviewers = ReviewerBundle.load(reviewer_path)
    context = ScoringContext(config, reviewers)
    calls = int(budget or config["execution"]["reviewer_budget_per_run"])
    if calls < int(config["execution"]["n_per_run"]):
        raise ValueError("Reviewer call budget cannot produce the requested unique count")
    families = sorted(config["families"])
    quotas = _family_quotas(calls, families)
    rng = random.Random(seed ^ int(stable_hash(method, 8), 16))
    results: list[dict[str, Any]] = []
    for family in families:
        members = [row for row in pool if row["family"] == family]
        quota = quotas[family]
        if len(members) < quota:
            raise ValueError(f"Family {family} has only {len(members)} candidates for quota {quota}")
        if method == "prior_random":
            candidates = _prior_candidates(rng, members, quota)
            results.extend(context.review_candidates(candidates, method, seed))
        else:
            results.extend(_adaptive_candidates(method, rng, members, quota, context, seed))
    _apply_diversity_penalty(results, config)
    results.sort(key=lambda row: (row["family"], -float(row.get("reward", 0.0)), row["smiles"]))
    for rank, row in enumerate(results, start=1):
        row["method_rank"] = rank
        row["reviewer_call"] = rank
    if len(results) != calls or len({row["smiles"] for row in results}) != len(results):
        raise RuntimeError("Search failed the exact-budget/uniqueness invariant")
    return results


def run_methods(
    config: dict[str, Any],
    library_path: str | Path,
    reviewer_path: str | Path,
    output_path: str | Path,
    methods: list[str],
) -> dict[str, Any]:
    all_rows: list[dict[str, Any]] = []
    run_counts: dict[str, int] = {}
    for method in methods:
        for seed in config["execution"]["seeds"]:
            rows = run_method(config, library_path, reviewer_path, method, int(seed))
            key = f"{method}:{seed}"
            run_counts[key] = len(rows)
            all_rows.extend(rows)
    write_csv(output_path, all_rows)
    return {
        "path": str(Path(output_path).resolve()),
        "rows": len(all_rows),
        "run_counts": run_counts,
        "methods": methods,
        "seeds": config["execution"]["seeds"],
    }


def write_generator_manifests(config: dict[str, Any], output_dir: str | Path) -> dict[str, Any]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    manifests = []
    toml_configs = []
    root = destination.parent
    scorer = Path(__file__).resolve().parents[1] / "scripts" / "reinvent_external_score.py"
    prior = config["production"]["reaction_prior_path"] or "PRIOR_PATH_REQUIRED"
    per_family_budget = math.ceil(int(config["execution"]["reviewer_budget_per_run"]) / len(config["families"]))
    steps_per_stage = 2
    batch_size = max(1, math.ceil(per_family_budget / (4 * steps_per_stage)))
    for family, family_config in config["families"].items():
        scaffold = family_config["ground_template"].format(r1="[*:1]", r2="[*:2]")
        scaffold_path = destination / f"scaffold_{family}.smi"
        scaffold_path.write_text(scaffold + "\n", encoding="utf-8")
        model_path = root / "models" / "reward" / "reviewers.pkl"
        library_path = root / "data" / "reaction_library.csv"
        lines = [
            'run_type = "staged_learning"',
            f'device = "{"cpu" if config["execution"]["mode"] == "smoke" else "cuda:0"}"',
            f'tb_logdir = "{(destination / ("tb_" + family)).as_posix()}"',
            f'json_out_config = "{(destination / ("resolved_" + family + ".json")).as_posix()}"',
            "", "[parameters]",
            f'prior_file = "{prior}"', f'agent_file = "{prior}"',
            f'smiles_file = "{scaffold_path.resolve().as_posix()}"',
            f'summary_csv_prefix = "{(destination / ("rl_" + family)).as_posix()}"',
            f"batch_size = {batch_size}", "randomize_smiles = true", "use_checkpoint = false", "purge_memories = false",
            "", "[learning_strategy]", 'type = "dap"', "sigma = 128", "rate = 0.0001",
            "", "[diversity_filter]", 'type = "PenalizeSameSmiles"', "bucket_size = 25", "minscore = 0.35", "penalty_multiplier = 0.5",
        ]
        for index, stage in enumerate(config["reward"]["stages"], start=1):
            stage_name = stage["name"]
            args = (
                f'{scorer.resolve().as_posix()} --config {Path(config["_config_path"]).resolve().as_posix()} '
                f'--model {model_path.resolve().as_posix()} --library {library_path.resolve().as_posix()} '
                f'--family {family} --stage {stage_name}'
            )
            lines.extend([
                "", "[[stage]]", f'chkpt_file = "{(destination / (family + "_stage" + str(index) + ".chkpt")).as_posix()}"',
                'termination = "simple"', "max_score = 0.999", f"min_steps = {steps_per_stage}", f"max_steps = {steps_per_stage}",
                "", "[stage.scoring]", 'type = "geometric_mean"',
                "", "[[stage.scoring.component]]", "[stage.scoring.component.ExternalProcess]",
                "[[stage.scoring.component.ExternalProcess.endpoint]]", f'name = "MOSTGen {stage_name}"', "weight = 1.0",
                f'params.executable = "{Path(sys.executable).resolve().as_posix()}"', f'params.args = "{args}"',
                'params.property = "stage_score"',
            ])
        toml_path = destination / f"reinvent4_{family}.toml"
        toml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        toml_configs.append(str(toml_path.resolve()))
        manifest = {
            "schema_version": "1.0",
            "family": family,
            "generator": "REINVENT4 LibInvent",
            "backend_status": "surrogate_smoke" if config["execution"]["backend"] == "surrogate_smoke" else "external_required",
            "repository": config["production"]["reinvent4_repository"],
            "revision": config["production"]["reinvent4_revision"],
            "version": config["production"]["reinvent4_version"],
            "prior": config["production"]["reaction_prior_path"] or "NOT_CONFIGURED",
            "reaction_smarts": family_config["reaction_smarts"],
            "allowed_positions": family_config["allowed_positions"],
            "curriculum": config["reward"]["stages"],
            "reward_weights": config["reward"]["weights"],
            "diversity_filter": {"type": "ECFP centroid", "penalty": config["reward"]["diversity_penalty"]},
            "seed_runs": config["execution"]["seeds"],
            "reviewer_budget_per_run": config["execution"]["reviewer_budget_per_run"],
            "toml_config": str(toml_path.resolve()),
            "scaffold_file": str(scaffold_path.resolve()),
            "external_scorer": str(scorer.resolve()),
        }
        path = destination / f"reinvent4_{family}.json"
        path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        manifests.append(str(path.resolve()))
    policy = {
        "status": "manifest_only" if config["execution"]["backend"] == "surrogate_smoke" else "ready_for_external_validation",
        "family_manifests": manifests,
        "reinvent_toml_configs": toml_configs,
        "reinvent_command": "reinvent -l <family>.log <family>.toml",
        "note": "The smoke policy is an adaptive reaction-library search, not a trained REINVENT neural prior.",
    }
    (destination / "generator_policy.json").write_text(json.dumps(policy, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return policy
