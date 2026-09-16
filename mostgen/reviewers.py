from __future__ import annotations

import json
import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any, Iterable

import numpy as np
from rdkit import DataStructs
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

from .chemistry import descriptors, fingerprint, max_similarity
from .data import WAVELENGTHS, read_csv
from .provenance import stable_hash


DESCRIPTOR_NAMES = (
    "mol_wt", "logp", "tpsa", "hbd", "hba", "rings", "aromatic_rings",
    "rotatable_bonds", "hetero_fraction", "fraction_csp3", "formal_charge",
    "charge_separation", "heavy_atoms",
)
FAMILIES = ("nbd_qc", "dewar_pyrimidinone", "spiropyran")
SPECTRAL_TARGETS = tuple(f"abs_{w}" for w in WAVELENGTHS)
MOST_TARGETS = ("energy_kj_mol", "specific_energy_wh_kg", "log_half_life_h")
SAFETY_TARGETS = ("kp_log_cm_s", "phototoxicity_probability")


def feature_vector(smiles: str, family: str, bits: int) -> np.ndarray:
    fp = fingerprint(smiles, bits)
    fp_array = np.zeros(bits, dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, fp_array)
    desc = descriptors(smiles)
    scaled = np.asarray([
        desc["mol_wt"] / 500.0,
        desc["logp"] / 6.0,
        desc["tpsa"] / 150.0,
        desc["hbd"] / 5.0,
        desc["hba"] / 10.0,
        desc["rings"] / 8.0,
        desc["aromatic_rings"] / 6.0,
        desc["rotatable_bonds"] / 12.0,
        desc["hetero_fraction"],
        desc["fraction_csp3"],
        desc["formal_charge"] / 3.0,
        desc["charge_separation"] / 4.0,
        desc["heavy_atoms"] / 60.0,
    ], dtype=np.float32)
    one_hot = np.asarray([1.0 if family == item else 0.0 for item in FAMILIES], dtype=np.float32)
    return np.concatenate([fp_array, scaled, one_hot])


@dataclass
class Prediction:
    mean: dict[str, float]
    std: dict[str, float]


class ReviewerBundle:
    """Serializable independent ensemble of spectrum, MOST, and safety models."""

    def __init__(
        self,
        purpose: str,
        bits: int,
        spectrum_models: list[Any],
        safety_models: list[Any],
        most_models: dict[str, list[Any]],
        references: dict[str, Any],
        metrics: dict[str, Any],
    ) -> None:
        self.purpose = purpose
        self.bits = bits
        self.spectrum_models = spectrum_models
        self.safety_models = safety_models
        self.most_models = most_models
        self.references = references
        self.metrics = metrics

    @staticmethod
    def _ensemble_predict(models: list[Any], vector: np.ndarray, targets: tuple[str, ...]) -> Prediction:
        member_values = []
        sample = vector.reshape(1, -1)
        for model in models:
            member_values.append(np.asarray(model.predict(sample)[0], dtype=float).reshape(-1))
        values = np.vstack(member_values)
        return Prediction(
            mean={key: float(value) for key, value in zip(targets, values.mean(axis=0))},
            std={key: float(value) for key, value in zip(targets, values.std(axis=0, ddof=0))},
        )

    def predict(self, smiles: str, family: str) -> dict[str, Any]:
        return self.predict_many([(smiles, family)])[0]

    def predict_many(self, molecules: list[tuple[str, str]]) -> list[dict[str, Any]]:
        if not molecules:
            return []
        x = np.vstack([feature_vector(smiles, family, self.bits) for smiles, family in molecules])

        def batch(models: list[Any], targets: tuple[str, ...], indices: list[int] | None = None):
            selected_x = x if indices is None else x[indices]
            values = np.asarray([model.predict(selected_x) for model in models], dtype=float)
            means, stds = values.mean(axis=0), values.std(axis=0, ddof=0)
            return [
                Prediction(
                    mean={key: float(value) for key, value in zip(targets, means[index].reshape(-1))},
                    std={key: float(value) for key, value in zip(targets, stds[index].reshape(-1))},
                )
                for index in range(len(selected_x))
            ]

        spectrum = batch(self.spectrum_models, SPECTRAL_TARGETS)
        safety = batch(self.safety_models, SAFETY_TARGETS)
        most: list[Prediction | None] = [None] * len(molecules)
        for family in FAMILIES:
            indices = [index for index, (_, item_family) in enumerate(molecules) if item_family == family]
            if not indices:
                continue
            predictions = batch(self.most_models[family], MOST_TARGETS, indices)
            for index, prediction in zip(indices, predictions):
                most[index] = prediction
        results = []
        spectral_refs = self.references["spectral_smiles"]
        for index, (smiles, family) in enumerate(molecules):
            results.append({
                "spectrum": spectrum[index], "most": most[index], "safety": safety[index],
                "ad_spectral_similarity": max_similarity(smiles, spectral_refs, self.bits),
                "ad_most_similarity": max_similarity(smiles, self.references["most_smiles_by_family"].get(family, []), self.bits),
                "family_reference_medians": self.references["family_medians"][family],
                "reviewer_purpose": self.purpose,
            })
        return results

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("wb") as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str | Path) -> "ReviewerBundle":
        with Path(path).open("rb") as handle:
            bundle = pickle.load(handle)
        if not isinstance(bundle, cls) and not (
            hasattr(bundle, "predict_many") and hasattr(bundle, "purpose")
        ):
            raise TypeError("Reviewer artifact has an unexpected type")
        return bundle


def _matrix(rows: list[dict[str, str]], bits: int) -> np.ndarray:
    return np.vstack([feature_vector(row["smiles"], row["family"], bits) for row in rows])


def _targets(rows: list[dict[str, str]], names: tuple[str, ...]) -> np.ndarray:
    return np.asarray([[float(row[name]) for name in names] for row in rows], dtype=np.float64)


def _new_model(kind: str, seed: int):
    common = dict(n_estimators=28, random_state=seed, n_jobs=1, min_samples_leaf=2, max_features=0.65)
    if kind == "reward":
        return RandomForestRegressor(bootstrap=True, **common)
    return ExtraTreesRegressor(bootstrap=False, **common)


def _fit_members(kind: str, count: int, seed: int, x: np.ndarray, y: np.ndarray) -> list[Any]:
    models = []
    for index in range(count):
        model = _new_model(kind, seed + 1009 * index)
        model.fit(x, y)
        models.append(model)
    return models


def _predict_mean(models: list[Any], x: np.ndarray) -> np.ndarray:
    return np.mean([np.asarray(model.predict(x), dtype=float) for model in models], axis=0)


def _split_metrics(
    rows: list[dict[str, str]],
    bits: int,
    spectrum_models: list[Any],
    safety_models: list[Any],
    most_models: dict[str, list[Any]],
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for split in ("validation", "test"):
        subset = [row for row in rows if row["split"] == split]
        if not subset:
            results[split] = {"n": 0, "status": "not_available"}
            continue
        x = _matrix(subset, bits)
        result: dict[str, Any] = {"n": len(subset)}
        for group, targets, models in (
            ("spectrum", SPECTRAL_TARGETS, spectrum_models),
            ("safety", SAFETY_TARGETS, safety_models),
        ):
            true = _targets(subset, targets)
            pred = _predict_mean(models, x)
            result[group] = {
                "mae": float(mean_absolute_error(true, pred)),
                "rmse": float(root_mean_squared_error(true, pred)),
            }
        most_errors = []
        for family in FAMILIES:
            members = [row for row in subset if row["family"] == family]
            if not members:
                continue
            family_x = _matrix(members, bits)
            true = _targets(members, MOST_TARGETS)
            pred = _predict_mean(most_models[family], family_x)
            most_errors.extend((true - pred).reshape(-1).tolist())
        result["most"] = {
            "mae": float(np.mean(np.abs(most_errors))) if most_errors else math.nan,
            "rmse": float(np.sqrt(np.mean(np.square(most_errors)))) if most_errors else math.nan,
        }
        results[split] = result
    return results


def _check_scaffold_leakage(rows: list[dict[str, str]]) -> None:
    split_scaffolds: dict[str, set[str]] = {}
    for row in rows:
        split_scaffolds.setdefault(row["split"], set()).add(row["scaffold"])
    names = sorted(split_scaffolds)
    for index, first in enumerate(names):
        for second in names[index + 1:]:
            overlap = split_scaffolds[first] & split_scaffolds[second]
            if overlap:
                raise ValueError(f"Scaffold leakage between {first} and {second}: {sorted(overlap)[:3]}")


def _family_medians(rows: Iterable[dict[str, str]]) -> dict[str, dict[str, float]]:
    rows = list(rows)
    medians: dict[str, dict[str, float]] = {}
    for family in FAMILIES:
        members = [row for row in rows if row["family"] == family]
        if not members:
            raise ValueError(f"No training references for family {family}")
        medians[family] = {
            key: float(median(float(row[key]) for row in members))
            for key in ("uvb_auc", "uva_auc", "energy_kj_mol", "specific_energy_wh_kg", "log_half_life_h")
        }
    return medians


def _train_bundle(rows: list[dict[str, str]], config: dict[str, Any], purpose: str) -> ReviewerBundle:
    _check_scaffold_leakage(rows)
    train = [row for row in rows if row["split"] == "train"]
    if not train:
        raise ValueError("Training split is empty")
    bits = int(config["reviewers"]["fingerprint_bits"])
    members = int(config["reviewers"]["ensemble_size"])
    base_seed = int(config["project"]["default_seed"]) + (0 if purpose == "reward" else 500_009)
    x = _matrix(train, bits)
    spectrum_models = _fit_members(purpose, members, base_seed + 11, x, _targets(train, SPECTRAL_TARGETS))
    safety_models = _fit_members(purpose, members, base_seed + 23, x, _targets(train, SAFETY_TARGETS))
    most_models: dict[str, list[Any]] = {}
    for offset, family in enumerate(FAMILIES):
        family_rows = [row for row in train if row["family"] == family]
        if len(family_rows) < 5:
            raise ValueError(f"Insufficient family-aware training data for {family}")
        most_models[family] = _fit_members(
            purpose, members, base_seed + 101 + offset,
            _matrix(family_rows, bits), _targets(family_rows, MOST_TARGETS),
        )
    metrics = _split_metrics(rows, bits, spectrum_models, safety_models, most_models)
    references = {
        "spectral_smiles": [row["smiles"] for row in train],
        "most_smiles_by_family": {family: [row["smiles"] for row in train if row["family"] == family] for family in FAMILIES},
        "family_medians": _family_medians(train),
        "training_data_hash": stable_hash("\n".join(sorted(row["candidate_id"] for row in train)), 32),
        "training_scaffolds": sorted({row["scaffold"] for row in train}),
    }
    return ReviewerBundle(purpose, bits, spectrum_models, safety_models, most_models, references, metrics)


def train_reviewers(config: dict[str, Any], data_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
    rows = read_csv(data_path)
    if rows and "endpoint" in rows[0]:
        from .real_reviewers import train_real_reviewers

        return train_real_reviewers(config, data_path, output_dir)
    if len(rows) > int(config["project"]["max_training_structures"]):
        raise ValueError("Reviewer training rows exceed configured cap")
    tiers = {row.get("evidence_tier") for row in rows}
    if config["execution"]["mode"] == "production" and tiers == {"synthetic_smoke_only"}:
        raise RuntimeError("Production mode refuses synthetic-only reviewer data")
    destination = Path(output_dir)
    reward = _train_bundle(rows, config, "reward")
    evaluator = _train_bundle(rows, config, "evaluator")
    reward_path = destination / "reward" / "reviewers.pkl"
    evaluator_path = destination / "evaluator" / "reviewers.pkl"
    reward.save(reward_path)
    evaluator.save(evaluator_path)
    metadata = {
        "schema_version": "1.0",
        "rows": len(rows),
        "evidence_tiers": sorted(str(item) for item in tiers),
        "feature_schema": [f"morgan_{i}" for i in range(reward.bits)] + list(DESCRIPTOR_NAMES) + [f"family_{item}" for item in FAMILIES],
        "reward": {"algorithm": "RandomForestRegressor ensemble", "path": str(reward_path.resolve()), "metrics": reward.metrics},
        "evaluator": {"algorithm": "ExtraTreesRegressor ensemble", "path": str(evaluator_path.resolve()), "metrics": evaluator.metrics},
        "independent_instances": reward_path.resolve() != evaluator_path.resolve(),
        "limitations": [
            "Synthetic smoke labels are not experimental observations.",
            "Applicability domains are fingerprint-neighbourhood proxies.",
            "Phototoxicity output is conservative triage, never proof of safety.",
        ],
    }
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "model_cards.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metadata
