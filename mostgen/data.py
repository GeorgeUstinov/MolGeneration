from __future__ import annotations

import csv
import json
import math
import random
from pathlib import Path
from typing import Any, Iterable

from .chemistry import build_pair, descriptors, murcko_scaffold, parse_smiles, safety_veto
from .numerics import critical_wavelength, trapezoid_auc
from .provenance import sha256_file, stable_hash


WAVELENGTHS = tuple(range(290, 401, 5))


def write_csv(path: str | Path, rows: Iterable[dict[str, Any]], fieldnames: list[str] | None = None) -> int:
    materialized = list(rows)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(materialized[0]) if materialized else []
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        if fieldnames:
            writer.writeheader()
            writer.writerows(materialized)
    return len(materialized)


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _noise(key: str, magnitude: float) -> float:
    integer = int(stable_hash(key, 12), 16)
    return magnitude * (2.0 * (integer / float(16**12 - 1)) - 1.0)


def _gaussian(x: float, centre: float, width: float, amplitude: float) -> float:
    return amplitude * math.exp(-0.5 * ((x - centre) / width) ** 2)


def reference_labels(smiles: str, charged_smiles: str, family: str) -> dict[str, float]:
    """Deterministic fixture labels used only by the smoke backend.

    Values are smooth functions of RDKit descriptors plus deterministic noise,
    which makes leakage/error tests meaningful without pretending that the
    records are experimental observations.
    """
    d = descriptors(smiles)
    q = descriptors(charged_smiles)
    family_peak = {"nbd_qc": (314.0, 366.0), "dewar_pyrimidinone": (307.0, 354.0), "spiropyran": (325.0, 382.0)}[family]
    family_energy = {"nbd_qc": 91.0, "dewar_pyrimidinone": 55.0, "spiropyran": 32.0}[family]
    family_log_half = {"nbd_qc": 1.02, "dewar_pyrimidinone": 0.78, "spiropyran": 0.62}[family]
    shift = 3.5 * d["aromatic_rings"] + 0.065 * d["tpsa"] + 1.7 * d["logp"]
    donor_acceptor = min(2.0, d["hba"] / 3.0 + d["hbd"] / 2.0)
    peak_a = family_peak[0] + 0.30 * shift + _noise(smiles + "p1", 4.0)
    peak_b = family_peak[1] + 0.65 * shift + _noise(smiles + "p2", 6.0)
    amp_a = 0.45 + 0.065 * d["aromatic_rings"] + 0.025 * donor_acceptor
    amp_b = 0.22 + 0.095 * donor_acceptor + 0.035 * d["aromatic_rings"]
    spectrum = [
        max(0.0, _gaussian(w, peak_a, 17.0, amp_a) + _gaussian(w, peak_b, 29.0, amp_b) + _noise(f"{smiles}:{w}", 0.012))
        for w in WAVELENGTHS
    ]
    uvb_auc = trapezoid_auc(WAVELENGTHS, spectrum, 290.0, 320.0)
    uva_auc = trapezoid_auc(WAVELENGTHS, spectrum, 320.0, 400.0)
    lambda_c = critical_wavelength(WAVELENGTHS, spectrum)
    delta_descriptor = abs(q["fraction_csp3"] - d["fraction_csp3"]) + abs(q["charge_separation"] - d["charge_separation"]) * 0.25
    energy = max(3.0, family_energy + 7.0 * delta_descriptor + 1.2 * d["aromatic_rings"] - 0.035 * d["mol_wt"] + _noise(smiles + "dh", 8.0))
    specific = energy * 277.7777778 / max(1.0, d["mol_wt"])
    log_half = family_log_half + 0.10 * d["aromatic_rings"] + 0.04 * d["logp"] - 0.025 * d["rotatable_bonds"] + _noise(smiles + "t", 0.24)
    kp = -5.30 + 0.42 * d["logp"] - 0.012 * d["mol_wt"] - 0.018 * d["tpsa"] + _noise(smiles + "kp", 0.30)
    photo_logit = -2.2 + 0.58 * d["aromatic_rings"] + 0.20 * max(0.0, d["logp"] - 3.0) + 0.15 * donor_acceptor
    phototoxicity = 1.0 / (1.0 + math.exp(-photo_logit))
    phototoxicity = min(0.98, max(0.02, phototoxicity + _noise(smiles + "pt", 0.08)))
    result = {
        "uvb_auc": uvb_auc,
        "uva_auc": uva_auc,
        "lambda_c_nm": lambda_c,
        "energy_kj_mol": energy,
        "specific_energy_wh_kg": specific,
        "log_half_life_h": log_half,
        "kp_log_cm_s": kp,
        "phototoxicity_probability": phototoxicity,
    }
    result.update({f"abs_{w}": value for w, value in zip(WAVELENGTHS, spectrum)})
    return result


def scaffold_split(scaffold: str) -> str:
    bucket = int(stable_hash(scaffold, 8), 16) % 10
    if bucket < 7:
        return "train"
    if bucket < 9:
        return "validation"
    return "test"


def enumerate_library(config: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    prior_elements = set(config.get("production", {}).get("libinvent_supported_elements", config["elements"]))
    synthon_ids = [
        entry["id"] for entry in config["synthons"]
        if {atom.GetSymbol() for atom in parse_smiles(entry["smiles"]).GetAtoms()} <= prior_elements
    ]
    for family in sorted(config["families"]):
        for first in synthon_ids:
            for second in synthon_ids:
                try:
                    pair = build_pair(config, family, first, second)
                except ValueError:
                    continue
                veto = safety_veto(pair.smiles, config)
                if veto.veto:
                    continue
                rows.append({
                    "candidate_id": stable_hash(f"{family}|{first}|{second}", 20),
                    "family": family,
                    "synthon_a": first,
                    "synthon_b": second,
                    "smiles": pair.smiles,
                    "charged_smiles": pair.charged_smiles,
                    "formula": pair.formula,
                    "scaffold": murcko_scaffold(pair.smiles),
                })
    # Canonical structure identity wins over alternate synthon encodings.
    unique: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        unique.setdefault((row["family"], row["smiles"]), row)
    return list(unique.values())


def _ensure_split_coverage(rows: list[dict[str, Any]]) -> None:
    """Keep scaffolds intact while ensuring every split is populated globally."""
    scaffolds = sorted({row["scaffold"] for row in rows})
    assignments = {scaffold: scaffold_split(scaffold) for scaffold in scaffolds}
    present = set(assignments.values())
    for wanted, index in (("validation", -2), ("test", -1)):
        if wanted not in present and len(scaffolds) >= abs(index):
            assignments[scaffolds[index]] = wanted
    for row in rows:
        row["split"] = assignments[row["scaffold"]]


def prepare_data(config: dict[str, Any], output_dir: str | Path, root: str | Path) -> dict[str, Any]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    root_path = Path(root)
    library = enumerate_library(config)
    if config["execution"].get("mode") != "smoke":
        # Import lazily: the smoke fixture remains lightweight, while full and
        # production modes are prohibited from manufacturing endpoint labels.
        from .real_data import prepare_real_data

        manifest = prepare_real_data(config, destination, root_path, library)
        manifest["training_rows"] = sum(manifest["endpoint_rows"].values())
        manifest["library_rows"] = manifest["reaction_library_rows"]
        return manifest
    per_family = int(config["training_rows_per_family"])
    rng = random.Random(int(config["project"]["default_seed"]))
    selected: list[dict[str, Any]] = []
    for family in sorted(config["families"]):
        members = [row for row in library if row["family"] == family]
        rng.shuffle(members)
        selected.extend(members[:per_family])
    if len(selected) > int(config["project"]["max_training_structures"]):
        raise ValueError("Prepared training corpus exceeds configured 30,000-row cap")
    _ensure_split_coverage(selected)
    source_by_family = {"nbd_qc": "M01", "dewar_pyrimidinone": "M03/M04", "spiropyran": "M01/M04"}
    training_rows: list[dict[str, Any]] = []
    for row in selected:
        labels = reference_labels(row["smiles"], row["charged_smiles"], row["family"])
        training_rows.append({
            **row,
            "source_id": source_by_family[row["family"]],
            "source_kind": "deterministic_fixture",
            "evidence_tier": "synthetic_smoke_only",
            "state": "ground_and_charged_pair",
            "temperature_k": 305.0,
            "medium": "conditional_standard_medium",
            "label_level": "molecule_state_condition",
            **labels,
        })
    training_path = destination / "reviewer_training.csv"
    write_csv(training_path, training_rows)
    library_path = destination / "reaction_library.csv"
    write_csv(library_path, library)
    local_inputs = []
    for name in ("database_matrix_MOST_UV_skin.xlsx", "Задание.docx", "Солнцезащитная плёнка с молекулярным накоплением солнечной энергии.pptx"):
        path = root_path / name
        if path.exists():
            local_inputs.append({"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    provenance = {
        "schema_version": "1.0",
        "raw_inputs_mutated": False,
        "derivative_data": str(training_path.resolve()),
        "training_rows": len(training_rows),
        "library_rows": len(library),
        "local_inputs": local_inputs,
        "source_registry": config["sources"],
        "warning": "Fixture labels are synthetic and cannot support scientific, efficacy, or safety claims.",
    }
    (destination / "data_manifest.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return provenance
