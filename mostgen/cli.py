from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

from .config import apply_override, dump_resolved_config, load_config
from .data import prepare_data, read_csv, write_csv
from .metrics import compute_metrics
from .oracle import availability
from .provenance import append_event, sha256_file, write_artifact_manifest, write_manifest
from .reporting import build_reports
from .review import review_generated
from .reviewers import train_reviewers
from .search import execute_generator_training, run_methods, run_reinvent_generation, write_generator_manifests
from .validation import verify_experiment


ROOT = Path(__file__).resolve().parents[1]


def _config(args: argparse.Namespace) -> dict[str, Any]:
    return apply_override(load_config(args.config, args.mode), args.override)


def _layout(output: str | Path) -> dict[str, Path]:
    root = Path(output).resolve()
    return {
        "root": root, "data": root / "data", "models": root / "models",
        "generator": root / "generator", "generated": root / "generated.csv",
        "metrics": root / "metrics", "review": root / "review",
    }


def _initialize(config: dict[str, Any], paths: dict[str, Path]) -> None:
    paths["root"].mkdir(parents=True, exist_ok=True)
    dump_resolved_config(config, paths["root"] / "config.resolved.json")
    inputs = [config["_config_path"], ROOT / "pyproject.toml", ROOT / "requirements.txt"]
    inputs.extend(sorted((ROOT / "mostgen").glob("*.py")))
    inputs.append(ROOT / "scripts" / "reinvent_external_score.py")
    inputs.extend(ROOT / name for name in ("database_matrix_MOST_UV_skin.xlsx", "Задание.docx", "Солнцезащитная плёнка с молекулярным накоплением солнечной энергии.pptx"))
    write_manifest(paths["root"] / "experiment_manifest.json", config, inputs)


def _require(path: Path, hint: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}; run `{hint}` first")


def _production_checks(config: dict[str, Any]) -> None:
    production = config["production"]
    errors = []
    if production["reinvent4_revision"].startswith("PIN_REQUIRED"):
        errors.append("REINVENT4 revision is not pinned")
    prior = Path(production["reaction_prior_path"]) if production["reaction_prior_path"] else None
    if prior is not None and not prior.is_absolute():
        prior = ROOT / prior
    if prior is None or not prior.is_file():
        errors.append("reaction prior path is not a file")
    elif not production.get("reaction_prior_sha256"):
        errors.append("reaction prior SHA-256 is not configured")
    elif sha256_file(prior) != production["reaction_prior_sha256"]:
        errors.append("reaction prior SHA-256 mismatch")
    reinvent_value = production.get("reinvent_executable", "reinvent")
    reinvent_path = Path(reinvent_value)
    if not reinvent_path.is_absolute():
        reinvent_path = ROOT / reinvent_path
    if not (reinvent_path.is_file() or shutil.which(reinvent_value)):
        errors.append("REINVENT4 `reinvent` executable is unavailable")
    if production.get("require_physical_oracle") and not availability(config)["ready"]:
        errors.append("xTB/sTDA toolchain is unavailable")
    if errors:
        raise RuntimeError("Production preflight failed: " + "; ".join(errors))


def command_prepare(args: argparse.Namespace) -> dict[str, Any]:
    config, paths = _config(args), _layout(args.output)
    _initialize(config, paths)
    result = prepare_data(config, paths["data"], ROOT)
    append_event(paths["root"] / "events.jsonl", "prepare-data", result)
    return result


def command_train_reviewers(args: argparse.Namespace) -> dict[str, Any]:
    config, paths = _config(args), _layout(args.output)
    _initialize(config, paths)
    training = paths["data"] / ("reviewer_training.csv" if config["execution"]["mode"] == "smoke" else "reviewer_endpoints.csv")
    _require(training, "prepare-data")
    result = train_reviewers(config, training, paths["models"])
    append_event(paths["root"] / "events.jsonl", "train-reviewers", {"rows": result["rows"]})
    return result


def command_train_generator(args: argparse.Namespace) -> dict[str, Any]:
    config, paths = _config(args), _layout(args.output)
    _initialize(config, paths)
    if config["execution"]["mode"] == "production":
        _production_checks(config)
    result = write_generator_manifests(config, paths["generator"])
    if config["execution"]["mode"] != "smoke":
        result["training"] = execute_generator_training(config, paths["generator"])
    append_event(paths["root"] / "events.jsonl", "train-generator", result)
    return result


def command_search(args: argparse.Namespace, methods: list[str]) -> dict[str, Any]:
    config, paths = _config(args), _layout(args.output)
    _initialize(config, paths)
    _require(paths["data"] / "reaction_library.csv", "prepare-data")
    _require(paths["models"] / "reward" / "reviewers.pkl", "train-reviewers")
    output_name = "generated.csv" if args.command in {"generate", "run-all"} else "baselines.csv"
    if methods == ["libinvent_rl"] and config["execution"]["mode"] != "smoke":
        _production_checks(config)
        for family in config["families"]:
            _require(paths["generator"] / f"libinvent_{family}.agent", "train-generator")
        result = run_reinvent_generation(
            config, paths["generator"], paths["data"] / "reaction_library.csv",
            paths["models"] / "reward" / "reviewers.pkl", paths["root"] / output_name,
        )
    else:
        result = run_methods(
            config, paths["data"] / "reaction_library.csv", paths["models"] / "reward" / "reviewers.pkl",
            paths["root"] / output_name, methods,
        )
    append_event(paths["root"] / "events.jsonl", args.command, result)
    return result


def command_review(args: argparse.Namespace) -> dict[str, Any]:
    config, paths = _config(args), _layout(args.output)
    _initialize(config, paths)
    _require(paths["generated"], "generate")
    _require(paths["models"] / "evaluator" / "reviewers.pkl", "train-reviewers")
    result = review_generated(config, paths["generated"], paths["models"] / "evaluator" / "reviewers.pkl", paths["review"])
    append_event(paths["root"] / "events.jsonl", "review", {"reviewed": result["reviewed"], "selected": result["selected_after_physical_oracle"]})
    return result


def command_run_all(args: argparse.Namespace) -> dict[str, Any]:
    config, paths = _config(args), _layout(args.output)
    _initialize(config, paths)
    data_result = prepare_data(config, paths["data"], ROOT)
    append_event(paths["root"] / "events.jsonl", "prepare-data", data_result)
    training_path = paths["data"] / ("reviewer_training.csv" if config["execution"]["mode"] == "smoke" else "reviewer_endpoints.csv")
    model_result = train_reviewers(config, training_path, paths["models"])
    append_event(paths["root"] / "events.jsonl", "train-reviewers", {"rows": model_result["rows"]})
    if config["execution"]["mode"] != "smoke":
        _production_checks(config)
    generator_result = write_generator_manifests(config, paths["generator"])
    if config["execution"]["mode"] != "smoke":
        generator_result["training"] = execute_generator_training(config, paths["generator"])
    if config["execution"]["mode"] == "smoke":
        search_result = run_methods(
            config, paths["data"] / "reaction_library.csv", paths["models"] / "reward" / "reviewers.pkl",
            paths["generated"], list(config["execution"]["methods"]),
        )
    else:
        baseline_path = paths["root"] / "baselines.csv"
        baseline_result = run_methods(
            config, paths["data"] / "reaction_library.csv", paths["models"] / "reward" / "reviewers.pkl",
            baseline_path, ["prior_random", "weighted_retraining"],
        )
        rl_path = paths["root"] / "libinvent_generated.csv"
        rl_result = run_reinvent_generation(
            config, paths["generator"], paths["data"] / "reaction_library.csv",
            paths["models"] / "reward" / "reviewers.pkl", rl_path,
        )
        combined = read_csv(baseline_path) + read_csv(rl_path)
        write_csv(paths["generated"], combined)
        search_result = {
            "path": str(paths["generated"]), "rows": len(combined),
            "run_counts": {**baseline_result["run_counts"], **rl_result["run_counts"]},
            "methods": list(config["execution"]["methods"]),
            "seeds": list(config["execution"]["seeds"]),
            "libinvent_backend": rl_result["backend"],
        }
    metrics_result = compute_metrics(paths["generated"], training_path, paths["metrics"], config)
    review_result = review_generated(config, paths["generated"], paths["models"] / "evaluator" / "reviewers.pkl", paths["review"])
    reports = build_reports(paths["root"], config)
    verification = verify_experiment(paths["root"], config)
    result = {
        "experiment": str(paths["root"]), "data_rows": data_result["training_rows"],
        "generated_rows": search_result["rows"], "run_counts": search_result["run_counts"],
        "matched_budget": metrics_result["matched_reviewer_budget"],
        "independent_reviewed": review_result["reviewed"],
        "selected": review_result["selected_after_physical_oracle"], "reports": reports,
        "generator_status": generator_result["status"],
        "acceptance_verified": verification["passed"],
    }
    append_event(paths["root"] / "events.jsonl", "run-all-complete", result)
    (paths["root"] / "run_summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_artifact_manifest(paths["root"])
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mostgen", description="Reproducible UV/MOST screening pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)
    help_text = {
        "prepare-data": "prepare immutable-source derivatives and scaffold splits",
        "train-reviewers": "train separate reward and evaluator ensembles",
        "sample-baselines": "run matched-budget prior and weighted baselines",
        "train-generator": "write family-specific REINVENT4/LibInvent manifests",
        "generate": "run real REINVENT4 LibInvent curriculum generation",
        "review": "independently review top candidates and queue physical oracle",
        "run-all": "execute the entire reproducible workflow",
    }
    for name, description in help_text.items():
        child = subparsers.add_parser(name, help=description)
        child.add_argument("--config", default=str(ROOT / "config" / "default.json"))
        child.add_argument("--override", help="JSON configuration patch")
        child.add_argument("--mode", choices=("smoke", "full", "production"), default="full")
        child.add_argument("--output", default="runs/latest")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare-data":
            result = command_prepare(args)
        elif args.command == "train-reviewers":
            result = command_train_reviewers(args)
        elif args.command == "sample-baselines":
            result = command_search(args, ["prior_random", "weighted_retraining"])
        elif args.command == "train-generator":
            result = command_train_generator(args)
        elif args.command == "generate":
            result = command_search(args, ["libinvent_rl"])
        elif args.command == "review":
            result = command_review(args)
        else:
            result = command_run_all(args)
        print(json.dumps(result, indent=2, sort_keys=True, default=str))
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
