from __future__ import annotations

import json
import math
import random
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from rdkit import DataStructs

from .chemistry import fingerprint, parse_smiles
from .data import read_csv, write_csv
from .provenance import stable_hash
from .reviewers import ReviewerBundle
from .scoring import ScoringContext


METHODS = ("prior_random", "weighted_retraining", "libinvent_rl")


def _update_budget_ledger(directory: str | Path, entries: dict[str, dict[str, Any]]) -> Path:
    """Atomically merge measured reward-evaluation counts for method/seed runs."""
    path = Path(directory) / "search_budget_ledger.json"
    payload: dict[str, Any] = {"schema_version": "1.0", "runs": {}}
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
    payload.setdefault("runs", {}).update(entries)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)
    return path


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
    if len(results) != calls or len({row["smiles"] for row in results}) != len(results):
        raise RuntimeError("Search failed the exact-budget/uniqueness invariant")
    for reviewer_call, row in enumerate(results, start=1):
        row["reviewer_call"] = reviewer_call
    _apply_diversity_penalty(results, config)
    retained: list[dict[str, Any]] = []
    output_quotas = _family_quotas(int(config["execution"]["n_per_run"]), families)
    for family in families:
        members = [row for row in results if row["family"] == family]
        members.sort(key=lambda row: (-float(row.get("reward", 0.0)), row["smiles"]))
        retained.extend(members[:output_quotas[family]])
    retained.sort(key=lambda row: (row["family"], -float(row.get("reward", 0.0)), row["smiles"]))
    for rank, row in enumerate(retained, start=1):
        row["method_rank"] = rank
        row["candidate_reward_evaluations_in_run"] = calls
    if len(retained) != int(config["execution"]["n_per_run"]):
        raise RuntimeError("Search failed to retain the requested family-balanced output quota")
    return retained


def run_methods(
    config: dict[str, Any],
    library_path: str | Path,
    reviewer_path: str | Path,
    output_path: str | Path,
    methods: list[str],
) -> dict[str, Any]:
    all_rows: list[dict[str, Any]] = []
    run_counts: dict[str, int] = {}
    ledger_entries: dict[str, dict[str, Any]] = {}
    for method in methods:
        for seed in config["execution"]["seeds"]:
            rows = run_method(config, library_path, reviewer_path, method, int(seed))
            key = f"{method}:{seed}"
            run_counts[key] = len(rows)
            ledger_entries[key] = {
                "method_id": method, "seed": int(seed),
                "candidate_reward_evaluations": int(config["execution"]["reviewer_budget_per_run"]),
                "retained_candidates": len(rows),
                "source": "reaction_library_search",
            }
            all_rows.extend(rows)
    write_csv(output_path, all_rows)
    ledger_path = _update_budget_ledger(Path(output_path).parent, ledger_entries)
    return {
        "path": str(Path(output_path).resolve()),
        "rows": len(all_rows),
        "run_counts": run_counts,
        "reviewer_call_counts": {key: value["candidate_reward_evaluations"] for key, value in ledger_entries.items()},
        "budget_ledger": str(ledger_path.resolve()),
        "methods": methods,
        "seeds": config["execution"]["seeds"],
    }


def write_generator_manifests(config: dict[str, Any], output_dir: str | Path) -> dict[str, Any]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    manifests = []
    toml_configs = []
    root = destination.parent
    scoring_config = destination / "scoring_config.resolved.json"
    scoring_config.write_text(
        json.dumps({key: value for key, value in config.items() if not key.startswith("_")}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    scorer = Path(__file__).resolve().parents[1] / "scripts" / "reinvent_external_score.py"
    prior_value = config["production"]["reaction_prior_path"] or "PRIOR_PATH_REQUIRED"
    prior_path = Path(prior_value)
    if not prior_path.is_absolute():
        prior_path = Path(__file__).resolve().parents[1] / prior_path
    prior = str(prior_path.resolve())
    per_family_budget = math.ceil(int(config["execution"]["reviewer_budget_per_run"]) / len(config["families"]))
    steps_per_stage = int(config["production"].get("rl_steps_per_stage", 2))
    batch_size = int(config["production"].get("rl_batch_size", max(1, math.ceil(per_family_budget / (4 * steps_per_stage)))))
    library_rows = read_csv(root / "data" / "reaction_library.csv") if (root / "data" / "reaction_library.csv").exists() else []
    synthons = {item["id"]: item["smiles"] for item in config["synthons"]}
    supported = set(config["production"].get("libinvent_supported_elements", config["elements"]))
    compatible_ids = {identifier for identifier, smiles in synthons.items()
                      if {atom.GetSymbol() for atom in parse_smiles(smiles).GetAtoms()} <= supported}
    for family, family_config in config["families"].items():
        scaffold = family_config["libinvent_scaffold"]
        scaffold_path = destination / f"scaffold_{family}.smi"
        scaffold_path.write_text(scaffold + "\n", encoding="utf-8")
        tl_rows = []
        for row in library_rows:
            if row["family"] != family:
                continue
            if row["synthon_a"] not in compatible_ids or row["synthon_b"] not in compatible_ids:
                continue
            first = "*" + synthons[row["synthon_a"]]
            second = "*" + synthons[row["synthon_b"]]
            tl_scaffold = scaffold.replace("[*:0]", "[*]").replace("[*:1]", "[*]")
            tl_rows.append(f"{tl_scaffold}\t{first}|{second}")
        tl_train = destination / f"tl_{family}.csv"
        tl_validation = destination / f"tl_{family}_validation.csv"
        split = max(1, int(len(tl_rows) * 0.9)) if tl_rows else 0
        tl_train.write_text("\n".join(tl_rows[:split]) + "\n", encoding="utf-8")
        tl_validation.write_text("\n".join(tl_rows[split:] or tl_rows[-1:]) + "\n", encoding="utf-8")
        tl_agent = destination / f"libinvent_{family}.agent"
        tl_lines = [
            'run_type = "transfer_learning"', 'device = "cpu"',
            f'tb_logdir = "{(destination / ("tb_tl_" + family)).resolve().as_posix()}"',
            f'json_out_config = "{(destination / ("tl_resolved_" + family + ".json")).resolve().as_posix()}"',
            "", "[parameters]", f"num_epochs = {int(config['production'].get('transfer_learning_epochs', 1))}",
            f"save_every_n_epochs = {int(config['production'].get('transfer_learning_epochs', 1))}",
            "batch_size = 96", "num_refs = 0", "sample_batch_size = 100", "tb_isim = false",
            f'input_model_file = "{prior}"', f'smiles_file = "{tl_train.resolve().as_posix()}"',
            f'validation_smiles_file = "{tl_validation.resolve().as_posix()}"',
            f'output_model_file = "{tl_agent.resolve().as_posix()}"',
        ]
        tl_config = destination / f"transfer_learning_{family}.toml"
        tl_config.write_text("\n".join(tl_lines) + "\n", encoding="utf-8")
        model_path = root / "models" / "reward" / "reviewers.pkl"
        library_path = root / "data" / "reaction_library.csv"
        family_toml_configs = []
        for run_seed in config["execution"]["seeds"]:
            run_seed = int(run_seed)
            lines = [
                'run_type = "staged_learning"', 'device = "cpu"', f"seed = {run_seed}",
                f'tb_logdir = "{(destination / ("tb_" + family + "_" + str(run_seed))).resolve().as_posix()}"',
                f'json_out_config = "{(destination / ("resolved_" + family + "_" + str(run_seed) + ".json")).resolve().as_posix()}"',
                "", "[parameters]",
                f'prior_file = "{prior}"', f'agent_file = "{tl_agent.resolve().as_posix()}"',
                f'smiles_file = "{scaffold_path.resolve().as_posix()}"',
                f'summary_csv_prefix = "{(destination / ("rl_" + family + "_" + str(run_seed))).resolve().as_posix()}"',
                f"batch_size = {batch_size}", "randomize_smiles = true", "use_checkpoint = false", "purge_memories = false",
                "", "[learning_strategy]", 'type = "dap"', "sigma = 128", "rate = 0.0001",
                "", "[diversity_filter]", 'type = "PenalizeSameSmiles"', "bucket_size = 25", "minscore = 0.35", "penalty_multiplier = 0.5",
            ]
            for index, stage in enumerate(config["reward"]["stages"], start=1):
                stage_name = stage["name"]
                args = (
                    f'{scorer.resolve().as_posix()} --config {scoring_config.resolve().as_posix()} '
                    f'--model {model_path.resolve().as_posix()} --library {library_path.resolve().as_posix()} '
                    f'--family {family} --stage {stage_name}'
                )
                lines.extend([
                    "", "[[stage]]",
                    f'chkpt_file = "{(destination / (family + "_" + str(run_seed) + "_stage" + str(index) + ".chkpt")).resolve().as_posix()}"',
                    'termination = "simple"', "max_score = 0.0", f"min_steps = {max(0, steps_per_stage - 2)}", f"max_steps = {steps_per_stage + 2}",
                    "", "[stage.scoring]", 'type = "geometric_mean"',
                    "", "[[stage.scoring.component]]", "[stage.scoring.component.ExternalProcess]",
                    "[[stage.scoring.component.ExternalProcess.endpoint]]", f'name = "MOSTGen {stage_name}"', "weight = 1.0",
                    f'params.executable = "{Path(sys.executable).absolute().as_posix()}"', f'params.args = "{args}"',
                    'params.property = "stage_score"',
                ])
            toml_path = destination / f"reinvent4_{family}_{run_seed}.toml"
            toml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            family_toml_configs.append(str(toml_path.resolve()))
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
            "toml_configs": family_toml_configs,
            "scaffold_file": str(scaffold_path.resolve()),
            "external_scorer": str(scorer.resolve()),
            "transfer_learning_config": str(tl_config.resolve()),
            "transfer_learning_rows": len(tl_rows),
            "trained_agent": str(tl_agent.resolve()),
        }
        path = destination / f"reinvent4_{family}.json"
        path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        manifests.append(str(path.resolve()))
    policy = {
        "status": "manifest_only" if config["execution"]["mode"] == "smoke" else "ready_to_train_reinvent4",
        "family_manifests": manifests,
        "reinvent_toml_configs": toml_configs,
        "reinvent_command": "reinvent -l <family>.log <family>.toml",
        "note": "The smoke policy is an adaptive reaction-library search, not a trained REINVENT neural prior.",
    }
    (destination / "generator_policy.json").write_text(json.dumps(policy, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return policy


def execute_generator_training(config: dict[str, Any], output_dir: str | Path) -> dict[str, Any]:
    """Run real LibInvent transfer learning with the pinned public prior."""
    destination = Path(output_dir)
    executable = Path(config["production"].get("reinvent_executable", "reinvent"))
    if not executable.is_absolute():
        executable = Path(__file__).resolve().parents[1] / executable
    if not executable.is_file():
        raise FileNotFoundError(f"REINVENT4 executable is unavailable: {executable}")
    runs = []
    for family in sorted(config["families"]):
        toml = destination / f"transfer_learning_{family}.toml"
        log = destination / f"transfer_learning_{family}.log"
        completed = subprocess.run([str(executable), "-l", str(log), str(toml)], cwd=Path(__file__).resolve().parents[1],
                                   text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
        agent = destination / f"libinvent_{family}.agent"
        runs.append({"family": family, "returncode": completed.returncode, "agent": str(agent.resolve()),
                     "agent_exists": agent.exists(), "log": str(log.resolve())})
        if completed.returncode != 0 or not agent.exists():
            tail = completed.stdout[-1200:] if completed.stdout else ""
            raise RuntimeError(f"LibInvent transfer learning failed for {family}; inspect {log}; {tail}")
    result = {"status": "trained", "generator": "REINVENT4 LibInvent", "runs": runs}
    (destination / "training_result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def execute_reinvent_curriculum(config: dict[str, Any], output_dir: str | Path) -> dict[str, Any]:
    """Execute the real four-stage REINVENT4 curriculum for each family."""
    destination = Path(output_dir)
    executable = Path(config["production"].get("reinvent_executable", "reinvent"))
    if not executable.is_absolute():
        executable = Path(__file__).resolve().parents[1] / executable
    runs = []
    for run_seed in config["execution"]["seeds"]:
        run_seed = int(run_seed)
        for family in sorted(config["families"]):
            toml = destination / f"reinvent4_{family}_{run_seed}.toml"
            log = destination / f"curriculum_{family}_{run_seed}.log"
            completed = subprocess.run([str(executable), "-l", str(log), str(toml)], cwd=Path(__file__).resolve().parents[1],
                                       text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
            stage_files = sorted(destination.glob(f"rl_{family}_{run_seed}_*.csv"))
            evaluation_rows = sum(max(0, sum(1 for _ in path.open(encoding="utf-8")) - 1) for path in stage_files)
            run = {"family": family, "seed": run_seed, "returncode": completed.returncode,
                   "log": str(log.resolve()), "stage_files": [str(path.resolve()) for path in stage_files],
                   "optimization_reviewer_evaluations": evaluation_rows}
            runs.append(run)
            if completed.returncode != 0 or len(stage_files) != len(config["reward"]["stages"]):
                tail = completed.stdout[-1200:] if completed.stdout else ""
                raise RuntimeError(f"REINVENT4 curriculum failed for {family}, seed {run_seed}; {tail}")
    result = {"status": "complete", "backend": "REINVENT4 LibInvent staged_learning", "runs": runs,
              "optimization_reviewer_evaluations": sum(run["optimization_reviewer_evaluations"] for run in runs)}
    (destination / "curriculum_result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def collect_reinvent_curriculum(config: dict[str, Any], generator_dir: str | Path,
                                 library_path: str | Path, reviewer_path: str | Path,
                                 output_path: str | Path) -> dict[str, Any]:
    """Map actual LibInvent outputs back to the allowed commercial library.

    REINVENT is allowed to propose arbitrary decorators, but only exact members
    of the versioned synthon library are eligible for reviewer scoring.
    """
    import csv

    destination = Path(generator_dir)
    library = read_csv(library_path)
    lookup = {row["smiles"]: row for row in library}
    reviewers = ReviewerBundle.load(reviewer_path)
    context = ScoringContext(config, reviewers)
    raw_count = valid_count = exact_count = 0
    unique: dict[tuple[str, str], tuple[dict[str, Any], dict[str, Any]]] = {}
    run_seed = int(config["project"]["default_seed"])
    for family in sorted(config["families"]):
        for stage_index, stage in enumerate(config["reward"]["stages"], start=1):
            path = destination / f"rl_{family}_{run_seed}_{stage_index}.csv"
            if not path.exists():
                continue
            with path.open(encoding="utf-8", newline="") as handle:
                for generated in csv.DictReader(handle):
                    raw_count += 1
                    if str(generated.get("SMILES_state")) == "0":
                        continue
                    valid_count += 1
                    try:
                        from .chemistry import standardize_smiles
                        canonical = standardize_smiles(generated["SMILES"])
                    except Exception:
                        continue
                    candidate = lookup.get(canonical)
                    if candidate is None or candidate["family"] != family:
                        continue
                    exact_count += 1
                    key = (family, canonical)
                    metadata = {"reinvent_stage": stage["name"], "reinvent_stage_index": stage_index,
                                "reinvent_raw_score": generated.get("Score"), "reinvent_step": generated.get("step"),
                                "reinvent_r_groups": generated.get("R-groups")}
                    previous = unique.get(key)
                    if previous is None or float(generated.get("Score") or 0.0) > float(previous[1].get("reinvent_raw_score") or 0.0):
                        unique[key] = (candidate, metadata)
    candidates = [item[0] for item in unique.values()]
    reviewed = context.review_candidates(candidates, "libinvent_rl", int(config["project"]["default_seed"]))
    metadata_by_key = {key: item[1] for key, item in unique.items()}
    for row in reviewed:
        row.update(metadata_by_key[(row["family"], row["smiles"])])
    _apply_diversity_penalty(reviewed, config)
    write_csv(output_path, reviewed)
    result = {"raw_reinvent_evaluations": raw_count, "valid_reinvent_outputs": valid_count,
              "commercial_library_matches": exact_count, "unique_eligible": len(reviewed),
              "path": str(Path(output_path).resolve()),
              "acceptance_1000_unique": len(reviewed) >= 1000}
    (destination / "curriculum_collection.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def sample_reinvent_candidates(config: dict[str, Any], generator_dir: str | Path,
                               library_path: str | Path, reviewer_path: str | Path,
                               output_path: str | Path, seed: int | None = None) -> dict[str, Any]:
    """Sample real LibInvent agents until the requested unique quota is met.

    The curriculum checkpoint is sampled first.  The less-collapsed TL agent is
    then used as a diversity backstop.  Structures outside the exact versioned
    commercial-synthon library are recorded but never returned or scored.
    """
    import csv

    destination = Path(generator_dir)
    executable = Path(config["production"].get("reinvent_executable", "reinvent"))
    if not executable.is_absolute():
        executable = Path(__file__).resolve().parents[1] / executable
    seed = int(seed if seed is not None else config["project"]["default_seed"])
    total_requested = int(config["execution"]["n_per_run"])
    quotas = _family_quotas(total_requested, sorted(config["families"]))
    library = read_csv(library_path)
    by_family = {family: {row["smiles"]: row for row in library if row["family"] == family}
                 for family in config["families"]}
    context = ScoringContext(config, ReviewerBundle.load(reviewer_path))
    selected: list[dict[str, Any]] = []
    audit = []
    for family in sorted(config["families"]):
        eligible: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
        agents = [
            ("rl_checkpoint", destination / f"{family}_{seed}_stage4.chkpt"),
            ("transfer_learning_agent", destination / f"libinvent_{family}.agent"),
        ]
        for agent_kind, agent in agents:
            if len(eligible) >= quotas[family]:
                break
            if not agent.exists():
                continue
            for round_index in range(int(config["production"].get("sampling_rounds", 3))):
                if len(eligible) >= quotas[family]:
                    break
                sample_count = max(2000, quotas[family] * int(config["production"].get("sampling_oversample", 12)))
                sample_file = destination / f"sample_{family}_{seed}_{agent_kind}_{round_index}.csv"
                sample_config = destination / f"sample_{family}_{seed}_{agent_kind}_{round_index}.toml"
                sample_log = destination / f"sample_{family}_{seed}_{agent_kind}_{round_index}.log"
                sample_config.write_text("\n".join([
                    'run_type = "sampling"', 'device = "cpu"', f"seed = {seed + 1009 * round_index}", "",
                    "[parameters]", f'model_file = "{agent.resolve().as_posix()}"',
                    f'smiles_file = "{(destination / ("scaffold_" + family + ".smi")).resolve().as_posix()}"',
                    f'output_file = "{sample_file.resolve().as_posix()}"', f"num_smiles = {sample_count}",
                    "unique_molecules = true", "randomize_smiles = true", "",
                ]), encoding="utf-8")
                completed = subprocess.run([str(executable), "-l", str(sample_log), str(sample_config)],
                                           cwd=Path(__file__).resolve().parents[1], text=True,
                                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
                if completed.returncode != 0 or not sample_file.exists():
                    raise RuntimeError(f"LibInvent sampling failed for {family}/{agent_kind}; inspect {sample_log}")
                raw = valid = matches = 0
                with sample_file.open(encoding="utf-8", newline="") as handle:
                    for record in csv.DictReader(handle):
                        raw += 1
                        if str(record.get("SMILES_state")) == "0":
                            continue
                        valid += 1
                        try:
                            from .chemistry import standardize_smiles
                            canonical = standardize_smiles(record["SMILES"])
                        except Exception:
                            continue
                        candidate = by_family[family].get(canonical)
                        if candidate is None:
                            continue
                        matches += 1
                        eligible.setdefault(canonical, (candidate, {
                            "reinvent_sampling_agent": agent_kind, "reinvent_sampling_round": round_index,
                            "reinvent_nll": record.get("NLL"), "reinvent_r_groups": record.get("R-groups"),
                        }))
                audit.append({"family": family, "agent": agent_kind, "round": round_index,
                              "requested": sample_count, "returned_unique": raw, "valid": valid,
                              "commercial_matches": matches, "eligible_cumulative": len(eligible)})
        if len(eligible) < quotas[family]:
            raise RuntimeError(f"LibInvent underfilled {family}: {len(eligible)}/{quotas[family]} unique allowed molecules")
        # Select without another reviewer call. RL-checkpoint samples are
        # preferred, followed by lower-NLL samples; the stable hash resolves
        # ties reproducibly. Exactly the final quota is then evaluated once.
        eligible_items = list(eligible.items())
        eligible_items.sort(key=lambda item: (
            item[1][1].get("reinvent_sampling_agent") != "rl_checkpoint",
            float(item[1][1].get("reinvent_nll") or float("inf")),
            stable_hash(f"{seed}:{item[0]}", 16),
        ))
        eligible_items = eligible_items[:quotas[family]]
        family_candidates = [value[0] for _, value in eligible_items]
        family_metadata = {key: value[1] for key, value in eligible.items()}
        family_reviewed = context.review_candidates(family_candidates, "libinvent_rl", seed)
        for row in family_reviewed:
            row.update(family_metadata[row["smiles"]])
        selected.extend(family_reviewed)
    _apply_diversity_penalty(selected, config)
    write_csv(output_path, selected)
    result = {"seed": seed, "rows": len(selected), "unique": len({row["smiles"] for row in selected}),
              "reviewer_calls": len(selected), "quota_met": len(selected) == total_requested,
              "audit": audit, "path": str(Path(output_path).resolve())}
    audit_path = destination / f"sampling_audit_{seed}.json"
    audit_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def run_reinvent_generation(config: dict[str, Any], generator_dir: str | Path,
                             library_path: str | Path, reviewer_path: str | Path,
                             output_path: str | Path) -> dict[str, Any]:
    """Run real curriculum learning and sample one exact-budget set per seed."""
    curriculum = execute_reinvent_curriculum(config, generator_dir)
    destination = Path(generator_dir)
    all_rows: list[dict[str, Any]] = []
    run_counts: dict[str, int] = {}
    sampling = []
    optimization_by_seed: dict[int, int] = defaultdict(int)
    for run in curriculum["runs"]:
        optimization_by_seed[int(run["seed"])] += int(run["optimization_reviewer_evaluations"])
    ledger_entries: dict[str, dict[str, Any]] = {}
    for run_seed in config["execution"]["seeds"]:
        run_seed = int(run_seed)
        seed_path = destination / f"generated_libinvent_{run_seed}.csv"
        result = sample_reinvent_candidates(
            config, destination, library_path, reviewer_path, seed_path, seed=run_seed,
        )
        rows = read_csv(seed_path)
        all_rows.extend(rows)
        key = f"libinvent_rl:{run_seed}"
        run_counts[key] = len(rows)
        total_calls = optimization_by_seed[run_seed] + int(result["reviewer_calls"])
        expected_calls = int(config["execution"]["reviewer_budget_per_run"])
        if total_calls != expected_calls:
            raise RuntimeError(
                f"LibInvent reviewer budget mismatch for seed {run_seed}: {total_calls} != {expected_calls}"
            )
        for row in rows:
            row["candidate_reward_evaluations_in_run"] = total_calls
        ledger_entries[key] = {
            "method_id": "libinvent_rl", "seed": run_seed,
            "candidate_reward_evaluations": total_calls,
            "optimization_evaluations": optimization_by_seed[run_seed],
            "final_scoring_evaluations": int(result["reviewer_calls"]),
            "retained_candidates": len(rows),
            "source": "REINVENT4_curriculum_plus_final_scoring",
        }
        sampling.append(result)
    write_csv(output_path, all_rows)
    ledger_path = _update_budget_ledger(Path(output_path).parent, ledger_entries)
    return {
        "path": str(Path(output_path).resolve()), "rows": len(all_rows),
        "run_counts": run_counts, "methods": ["libinvent_rl"],
        "seeds": list(config["execution"]["seeds"]), "curriculum": curriculum,
        "sampling": sampling, "backend": "REINVENT4 LibInvent staged_learning",
        "reviewer_call_counts": {key: value["candidate_reward_evaluations"] for key, value in ledger_entries.items()},
        "budget_ledger": str(ledger_path.resolve()),
    }
