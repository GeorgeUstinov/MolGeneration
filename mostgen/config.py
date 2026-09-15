from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config" / "default.json"


class ConfigError(ValueError):
    """Raised when an experiment configuration violates the contract."""


def _deep_update(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path: str | Path | None = None, mode: str = "full") -> dict[str, Any]:
    config_path = Path(path) if path else DEFAULT_CONFIG
    with config_path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    config["_config_path"] = str(config_path.resolve())
    if mode == "smoke":
        smoke = copy.deepcopy(config["smoke"])
        config["execution"].update({k: v for k, v in smoke.items() if k != "training_rows_per_family"})
        config["execution"]["mode"] = "smoke"
        config["training_rows_per_family"] = smoke["training_rows_per_family"]
    elif mode == "full":
        config["execution"]["mode"] = "full"
        config["training_rows_per_family"] = 180
    elif mode == "production":
        config["execution"]["mode"] = "production"
        config["execution"]["backend"] = "reinvent4"
        config["training_rows_per_family"] = 180
    else:
        raise ConfigError(f"Unknown mode: {mode}")
    validate_config(config)
    return config


def apply_override(config: dict[str, Any], override_path: str | Path | None) -> dict[str, Any]:
    if not override_path:
        return config
    with Path(override_path).open(encoding="utf-8") as handle:
        patch = json.load(handle)
    merged = _deep_update(copy.deepcopy(config), patch)
    validate_config(merged)
    return merged


def validate_config(config: dict[str, Any]) -> None:
    required_families = {"nbd_qc", "dewar_pyrimidinone", "spiropyran"}
    missing = required_families - set(config.get("families", {}))
    if missing:
        raise ConfigError(f"Missing family configurations: {sorted(missing)}")
    project = config.get("project", {})
    if int(project.get("max_training_structures", 0)) > 30_000:
        raise ConfigError("max_training_structures exceeds the 30,000 structure contract")
    execution = config.get("execution", {})
    if float(execution.get("gpu_hours", 0.0)) > float(project.get("max_gpu_hours", 8.0)):
        raise ConfigError("Configured GPU budget exceeds project maximum")
    if int(execution.get("reviewer_budget_per_run", 0)) < int(execution.get("n_per_run", 0)):
        raise ConfigError("reviewer_budget_per_run must be at least n_per_run")
    if len(config.get("synthons", [])) < 2:
        raise ConfigError("At least two synthons are required")
    ids = [item["id"] for item in config["synthons"]]
    if len(ids) != len(set(ids)):
        raise ConfigError("Synthon identifiers must be unique")
    if not 0.0 < float(config["reward"]["floor"]) < 1.0:
        raise ConfigError("Reward floor must be between zero and one")


def dump_resolved_config(config: dict[str, Any], path: str | Path) -> None:
    clean = {k: v for k, v in config.items() if not k.startswith("_")}
    Path(path).write_text(json.dumps(clean, indent=2, sort_keys=True) + "\n", encoding="utf-8")

