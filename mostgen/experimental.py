from __future__ import annotations

"""Schemas and fail-closed validation for the external wet-lab campaign.

This module does not pretend to perform experiments.  It turns a computational
shortlist into versioned blank result tables and validates files returned by an
authorized laboratory before those measurements can be joined to model data.
"""

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from .data import read_csv, write_csv


SCHEMAS: dict[str, list[str]] = {
    "candidate_registry.csv": [
        "candidate_id", "smiles", "charged_smiles", "family", "batch_id",
        "identity_confirmed", "nmr_confirmed", "lcms_confirmed", "purity_hplc_fraction",
    ],
    "solution_spectra.csv": [
        "candidate_id", "batch_id", "replicate", "state", "solvent", "temperature_k",
        "concentration_mol_l", "path_length_cm", "wavelength_nm", "absorbance",
        "dark_corrected", "instrument_id", "raw_file_sha256",
    ],
    "photokinetics.csv": [
        "candidate_id", "batch_id", "replicate", "irradiation_nm", "photon_flux_mol_s",
        "initial_state", "pss_charged_fraction", "quantum_yield", "thermal_temperature_k",
        "half_life_h", "fit_r2", "raw_file_sha256",
    ],
    "calorimetry.csv": [
        "candidate_id", "batch_id", "replicate", "method", "charged_fraction",
        "delta_h_kj_mol", "specific_energy_wh_kg", "standard_uncertainty_wh_kg",
        "instrument_id", "raw_file_sha256",
    ],
    "cycling.csv": [
        "candidate_id", "batch_id", "replicate", "cycle_count",
        "remaining_capacity_fraction", "photoproduct_fraction", "analytical_method",
        "raw_file_sha256",
    ],
    "film_spectra.csv": [
        "formulation_id", "candidate_id", "batch_id", "replicate", "matrix",
        "loading_wt_fraction", "thickness_um", "state", "wavelength_nm",
        "transmittance_fraction", "instrument_id", "raw_file_sha256",
    ],
    "oecd_tg432.csv": [
        "candidate_id", "batch_id", "laboratory", "study_id", "glp_status",
        "cell_line", "replicate", "uva_dose_j_cm2", "ic50_minus_uv_mg_ml",
        "ic50_plus_uv_mg_ml", "photo_irritation_factor", "mean_photo_effect",
        "classification", "report_sha256",
    ],
}


def write_experimental_package(shortlist_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    shortlist = read_csv(shortlist_path) if Path(shortlist_path).exists() else []
    registry = []
    seen = set()
    for row in shortlist:
        candidate_id = row.get("candidate_id")
        if not candidate_id or candidate_id in seen:
            continue
        seen.add(candidate_id)
        registry.append({
            "candidate_id": candidate_id, "smiles": row["smiles"],
            "charged_smiles": row["charged_smiles"], "family": row["family"],
            "batch_id": "", "identity_confirmed": "", "nmr_confirmed": "",
            "lcms_confirmed": "", "purity_hplc_fraction": "",
        })
    write_csv(destination / "candidate_registry.csv", registry, SCHEMAS["candidate_registry.csv"])
    for filename, columns in SCHEMAS.items():
        if filename != "candidate_registry.csv":
            write_csv(destination / filename, [], columns)
    manifest = {
        "schema_version": "1.0", "candidate_count": len(registry),
        "status": "ready_for_authorized_laboratory" if registry else "blocked_no_computational_candidates",
        "tables": SCHEMAS,
        "minimum_replicates": 3,
        "claim_boundary": "Blank templates are not experimental evidence. OECD TG 432 must be run by a competent laboratory.",
        "protocol": "docs/EXPERIMENTAL_VALIDATION.md",
    }
    (destination / "experimental_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    return manifest


def _truth(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _finite(row: dict[str, str], key: str, low: float | None = None,
            high: float | None = None) -> bool:
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError):
        return False
    return math.isfinite(value) and (low is None or value >= low) and (high is None or value <= high)


def validate_experimental_results(input_dir: str | Path) -> dict[str, Any]:
    """Validate identity, ranges, coverage and replication; never infer blanks."""
    source = Path(input_dir)
    errors: list[str] = []
    tables: dict[str, list[dict[str, str]]] = {}
    for filename, required in SCHEMAS.items():
        path = source / filename
        if not path.is_file():
            errors.append(f"missing_table:{filename}")
            continue
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            missing = set(required) - set(reader.fieldnames or [])
            if missing:
                errors.append(f"missing_columns:{filename}:{','.join(sorted(missing))}")
            tables[filename] = list(reader)
    registry = tables.get("candidate_registry.csv", [])
    candidates = {row.get("candidate_id", "") for row in registry if row.get("candidate_id")}
    for row in registry:
        cid = row.get("candidate_id", "")
        if not (_truth(row.get("identity_confirmed")) and _truth(row.get("nmr_confirmed"))
                and _truth(row.get("lcms_confirmed"))):
            errors.append(f"identity_not_confirmed:{cid}")
        if not _finite(row, "purity_hplc_fraction", 0.95, 1.0):
            errors.append(f"purity_below_0.95_or_invalid:{cid}")
    for filename, rows in tables.items():
        if filename == "candidate_registry.csv":
            continue
        for row in rows:
            cid = row.get("candidate_id", "")
            if cid not in candidates:
                errors.append(f"unknown_candidate:{filename}:{cid}")
    # Each curve must cover the complete requested molecular or film interval.
    for filename, wavelength_key, group_keys in (
        ("solution_spectra.csv", "wavelength_nm", ("candidate_id", "batch_id", "replicate", "state", "solvent")),
        ("film_spectra.csv", "wavelength_nm", ("formulation_id", "replicate", "state")),
    ):
        groups: dict[tuple[str, ...], list[float]] = defaultdict(list)
        for row in tables.get(filename, []):
            if _finite(row, wavelength_key, 200.0, 900.0):
                groups[tuple(row.get(key, "") for key in group_keys)].append(float(row[wavelength_key]))
        for key, wavelengths in groups.items():
            if min(wavelengths) > 290.0 or max(wavelengths) < 400.0:
                errors.append(f"incomplete_290_400_coverage:{filename}:{key}")
    # Scalar assays need three independent replicate identifiers per candidate.
    for filename in ("photokinetics.csv", "calorimetry.csv", "cycling.csv", "oecd_tg432.csv"):
        replicates: dict[str, set[str]] = defaultdict(set)
        for row in tables.get(filename, []):
            replicates[row.get("candidate_id", "")].add(row.get("replicate", ""))
        for cid in candidates:
            if len(replicates.get(cid, set()) - {""}) < 3:
                errors.append(f"fewer_than_3_replicates:{filename}:{cid}")
    result = {
        "schema_version": "1.0", "schema_valid": not errors,
        "experiment_complete": bool(candidates) and not errors,
        "status": ("complete" if candidates and not errors else
                   "blocked_no_candidates" if not candidates else "invalid"),
        "valid": not errors, "errors": sorted(set(errors)),
        "candidate_count": len(candidates),
        "row_counts": {filename: len(rows) for filename, rows in tables.items()},
    }
    (source / "experimental_validation.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    return result
