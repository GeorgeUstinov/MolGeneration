from __future__ import annotations

import csv

from mostgen.experimental import SCHEMAS, validate_experimental_results, write_experimental_package


def test_blank_experimental_package_is_explicitly_blocked(tmp_path):
    package = tmp_path / "package"
    result = write_experimental_package(tmp_path / "missing_shortlist.csv", package)
    assert result["status"] == "blocked_no_computational_candidates"
    for filename, columns in SCHEMAS.items():
        with (package / filename).open(encoding="utf-8", newline="") as handle:
            assert next(csv.reader(handle)) == columns


def test_blank_package_cannot_be_misread_as_completed_experiment(tmp_path):
    package = tmp_path / "package"
    write_experimental_package(tmp_path / "missing.csv", package)
    validation = validate_experimental_results(package)
    assert validation["valid"]
    assert not validation["experiment_complete"]
    assert validation["status"] == "blocked_no_candidates"
    assert validation["candidate_count"] == 0
    assert all(count == 0 for count in validation["row_counts"].values())
