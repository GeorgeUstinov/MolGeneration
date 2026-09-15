from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from . import __version__


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_hash(text: str, length: int = 16) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:length]


def package_versions(names: Iterable[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for name in names:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = "not-installed"
    return result


def git_state(root: Path) -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, check=True,
            text=True, capture_output=True, timeout=5,
        ).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "status", "--porcelain"], cwd=root, check=True,
            text=True, capture_output=True, timeout=5,
        ).stdout.strip())
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.SubprocessError):
        return {"commit": "not-a-git-worktree", "dirty": None}


def write_manifest(
    path: str | Path,
    config: dict[str, Any],
    inputs: Iterable[str | Path] = (),
    command: list[str] | None = None,
) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    input_records = []
    for item in inputs:
        source = Path(item)
        if source.exists() and source.is_file():
            input_records.append({
                "path": str(source.resolve()),
                "bytes": source.stat().st_size,
                "sha256": sha256_file(source),
            })
    manifest = {
        "schema_version": "1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mostgen_version": __version__,
        "python": sys.version,
        "platform": platform.platform(),
        "executable": sys.executable,
        "command": command or sys.argv,
        "cwd": str(Path.cwd()),
        "timezone": os.environ.get("TZ", "system-default"),
        "random_seeds": list(config["execution"]["seeds"]),
        "budget": {
            "max_training_structures": config["project"]["max_training_structures"],
            "max_gpu_hours": config["project"]["max_gpu_hours"],
            "configured_gpu_hours": config["execution"]["gpu_hours"],
            "reviewer_calls_per_run": config["execution"]["reviewer_budget_per_run"],
        },
        "backend": config["execution"]["backend"],
        "packages": package_versions(["rdkit", "numpy", "pandas", "scikit-learn", "PyYAML"]),
        "git": git_state(root),
        "inputs": input_records,
        "sources": [
            {
                **source,
                "version": source.get("version", "REFERENCE_UNPINNED"),
                "checksum": source.get("checksum", "NOT_FETCHED_NO_LOCAL_ARTIFACT"),
            }
            for source in config["sources"]
        ],
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def write_artifact_manifest(root: str | Path) -> dict[str, Any]:
    experiment = Path(root).resolve()
    destination = experiment / "artifact_manifest.json"
    artifacts = []
    for path in sorted(experiment.rglob("*")):
        if not path.is_file() or path == destination:
            continue
        artifacts.append({
            "path": str(path.relative_to(experiment)),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    manifest = {"schema_version": "1.0", "root": str(experiment), "artifacts": artifacts}
    destination.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def append_event(path: str | Path, event: str, payload: dict[str, Any]) -> None:
    record = {
        "time_utc": datetime.now(timezone.utc).isoformat(),
        "event": event,
        **payload,
    }
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
