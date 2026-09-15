from __future__ import annotations

import json
from pathlib import Path

import pytest

from mostgen.config import load_config
from mostgen.data import prepare_data
from mostgen.metrics import compute_metrics
from mostgen.provenance import write_artifact_manifest
from mostgen.reporting import build_reports
from mostgen.review import review_generated
from mostgen.reviewers import train_reviewers
from mostgen.search import run_methods, write_generator_manifests
from mostgen.validation import verify_experiment


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def smoke_experiment(tmp_path_factory):
    root = tmp_path_factory.mktemp("mostgen-smoke")
    config = load_config(mode="smoke")
    prepare_data(config, root / "data", PROJECT_ROOT)
    train_reviewers(config, root / "data" / "reviewer_training.csv", root / "models")
    write_generator_manifests(config, root / "generator")
    run_methods(
        config, root / "data" / "reaction_library.csv", root / "models" / "reward" / "reviewers.pkl",
        root / "generated.csv", list(config["execution"]["methods"]),
    )
    metrics = compute_metrics(root / "generated.csv", root / "data" / "reviewer_training.csv", root / "metrics", config)
    review = review_generated(config, root / "generated.csv", root / "models" / "evaluator" / "reviewers.pkl", root / "review")
    reports = build_reports(root, config)
    verification = verify_experiment(root, config)
    write_artifact_manifest(root)
    return {"root": root, "config": config, "metrics": metrics, "review": review, "reports": reports, "verification": verification}
