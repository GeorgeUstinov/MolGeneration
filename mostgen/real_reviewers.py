from __future__ import annotations

"""Endpoint-specific reviewers trained only on parsed experimental records."""

import json
import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, balanced_accuracy_score, mean_absolute_error, root_mean_squared_error, roc_auc_score

from .chemistry import max_similarity
from .data import WAVELENGTHS, read_csv
from .numerics import critical_wavelength, trapezoid_auc
from .provenance import stable_hash
from .reviewers import FAMILIES, Prediction, feature_vector


REGRESSION_ENDPOINTS = ("lambda_max_nm", "log_half_life_h", "kp_log_cm_s")
CLASSIFICATION_ENDPOINTS = ("phototoxicity", "skin_sensitization", "skin_irritation")
ALL_ENDPOINTS = REGRESSION_ENDPOINTS + CLASSIFICATION_ENDPOINTS


@dataclass
class EndpointEnsemble:
    endpoint: str
    task: str
    models: list[Any]
    conformal_q90: float
    reference_smiles: list[str]
    metrics: dict[str, Any]

    def members(self, x: np.ndarray) -> np.ndarray:
        if self.task == "classification":
            outputs = []
            for model in self.models:
                classes = list(model.classes_)
                probabilities = model.predict_proba(x)
                outputs.append(probabilities[:, classes.index(1.0)] if 1.0 in classes else np.zeros(len(x)))
            return np.asarray(outputs, dtype=float)
        return np.asarray([model.predict(x) for model in self.models], dtype=float)

    def predict(self, x: np.ndarray, z: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        values = self.members(x)
        mean = values.mean(axis=0)
        ensemble_std = values.std(axis=0, ddof=0)
        calibrated = np.maximum(ensemble_std, self.conformal_q90 / max(z, 1e-8))
        return mean, calibrated, values


class RealReviewerBundle:
    """Serializable bundle with independent endpoint models and provenance."""

    data_mode = "real_endpoint_specific"

    def __init__(self, purpose: str, bits: int, z: float, endpoints: dict[str, EndpointEnsemble],
                 references: dict[str, Any], metrics: dict[str, Any]) -> None:
        self.purpose = purpose
        self.bits = bits
        self.z = z
        self.endpoints = endpoints
        self.references = references
        self.metrics = metrics

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("wb") as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str | Path) -> "RealReviewerBundle":
        with Path(path).open("rb") as handle:
            model = pickle.load(handle)
        if not isinstance(model, cls):
            raise TypeError("Not a real endpoint-specific reviewer bundle")
        return model

    def predict(self, smiles: str, family: str) -> dict[str, Any]:
        return self.predict_many([(smiles, family)])[0]

    def predict_many(self, molecules: list[tuple[str, str]]) -> list[dict[str, Any]]:
        if not molecules:
            return []
        x = np.vstack([feature_vector(smiles, family, self.bits) for smiles, family in molecules])
        predictions = {name: model.predict(x, self.z) for name, model in self.endpoints.items()}
        output = []
        for index, (smiles, family) in enumerate(molecules):
            lambda_mean, lambda_std, lambda_members = predictions["lambda_max_nm"]
            member_curves = []
            for peak in lambda_members[:, index]:
                # This deliberately remains a declared band-shape proxy.  A
                # lambda-max label cannot identify an experimental spectrum.
                member_curves.append(np.asarray([
                    math.exp(-0.5 * ((wavelength - float(peak)) / 34.0) ** 2)
                    for wavelength in WAVELENGTHS
                ]))
            curves = np.vstack(member_curves)
            spectrum = Prediction(
                mean={f"abs_{w}": float(v) for w, v in zip(WAVELENGTHS, curves.mean(axis=0))},
                std={f"abs_{w}": float(v) for w, v in zip(WAVELENGTHS, curves.std(axis=0))},
            )

            def pred(endpoint: str, output_name: str) -> Prediction:
                mean, spread, _ = predictions[endpoint]
                return Prediction({output_name: float(mean[index])}, {output_name: float(spread[index])})

            half = pred("log_half_life_h", "log_half_life_h")
            kp = pred("kp_log_cm_s", "kp_log_cm_s")
            photo = pred("phototoxicity", "phototoxicity_probability")
            sensitization = pred("skin_sensitization", "sensitization_probability")
            irritation = pred("skin_irritation", "irritation_probability")
            safety = Prediction(
                mean={**kp.mean, **photo.mean, **sensitization.mean, **irritation.mean},
                std={**kp.std, **photo.std, **sensitization.std, **irritation.std},
            )
            most = Prediction(
                mean={"energy_kj_mol": math.nan, "specific_energy_wh_kg": math.nan, **half.mean},
                std={"energy_kj_mol": math.nan, "specific_energy_wh_kg": math.nan, **half.std},
            )
            endpoint_ad = {
                name: max_similarity(smiles, model.reference_smiles, self.bits)
                for name, model in self.endpoints.items()
            }
            output.append({
                "spectrum": spectrum, "most": most, "safety": safety,
                "ad_spectral_similarity": endpoint_ad["lambda_max_nm"],
                # Energy is the defining MOST endpoint; without molecular
                # energy labels it is out of domain until the physical oracle.
                "ad_most_similarity": 0.0,
                "endpoint_ad": endpoint_ad,
                "family_reference_medians": self.references["family_medians"][family],
                "reviewer_purpose": self.purpose,
                "data_mode": self.data_mode,
                "evidence": {
                    "spectrum": "lambda_max_gaussian_band_proxy",
                    "half_life": "M01_class_shift_model",
                    "energy": "missing_requires_gfn2_xtb_or_experiment",
                    "kp": "U12_SkinPiX_plus_U13_human_epidermis",
                    "phototoxicity": "U16_3T3_NRU",
                    "sensitization": "U09_human_patch_test",
                    "irritation": "U07_NICE_exact_DTXSID_join",
                },
                "lambda_max_nm": float(lambda_mean[index]),
                "lambda_max_uncertainty": float(lambda_std[index]),
            })
        return output


def _matrix(rows: list[dict[str, str]], bits: int) -> np.ndarray:
    return np.vstack([feature_vector(row["smiles"], row.get("family", "unassigned"), bits) for row in rows])


def _deterministic_refs(rows: list[dict[str, str]], limit: int = 768) -> list[str]:
    unique = sorted({row["smiles"] for row in rows}, key=lambda smiles: stable_hash(smiles, 32))
    return unique[:limit]


def _new_model(purpose: str, task: str, seed: int, trees: int):
    common = dict(n_estimators=trees, random_state=seed, n_jobs=1, min_samples_leaf=2, max_features=0.65)
    if task == "classification":
        cls = RandomForestClassifier if purpose == "reward" else ExtraTreesClassifier
        return cls(class_weight="balanced", bootstrap=purpose == "reward", **common)
    cls = RandomForestRegressor if purpose == "reward" else ExtraTreesRegressor
    return cls(bootstrap=purpose == "reward", **common)


def _fit_endpoint(rows: list[dict[str, str]], endpoint: str, purpose: str, bits: int,
                  members: int, trees: int, seed: int, z: float) -> EndpointEnsemble:
    endpoint_rows = [row for row in rows if row["endpoint"] == endpoint]
    task = "classification" if endpoint in CLASSIFICATION_ENDPOINTS else "regression"
    train = [row for row in endpoint_rows if row["split"] == "train"]
    calibration = [row for row in endpoint_rows if row["split"] == "validation"]
    test = [row for row in endpoint_rows if row["split"] == "test"]
    if task == "classification" and len({float(row["target"]) for row in train}) < 2:
        # Deterministic group split can be class-imbalanced for a 53-molecule
        # source. Move complete scaffolds, never individual duplicate rows.
        train = [row for row in endpoint_rows if row["split"] != "test"]
        calibration = test
    if len(train) < 8:
        raise ValueError(f"Insufficient real training records for {endpoint}: {len(train)}")
    x_train = _matrix(train, bits)
    y_train = np.asarray([float(row["target"]) for row in train])
    models = []
    for index in range(members):
        model = _new_model(purpose, task, seed + index * 1009, trees)
        model.fit(x_train, y_train)
        models.append(model)
    provisional = EndpointEnsemble(endpoint, task, models, 0.0, _deterministic_refs(train), {})
    calibration_residuals = []
    if calibration:
        y_cal = np.asarray([float(row["target"]) for row in calibration])
        member_values = provisional.members(_matrix(calibration, bits))
        calibration_residuals = np.abs(y_cal - member_values.mean(axis=0)).tolist()
    q90 = float(np.quantile(calibration_residuals, 0.90, method="higher")) if calibration_residuals else (0.25 if task == "classification" else 1.0)
    metrics: dict[str, Any] = {
        "n_total": len(endpoint_rows), "n_train": len(train), "n_calibration": len(calibration), "n_test": len(test),
        "unique_structures": len({row["smiles"] for row in endpoint_rows}), "conformal_q90": q90,
    }
    if test:
        y_test = np.asarray([float(row["target"]) for row in test])
        pred = provisional.members(_matrix(test, bits)).mean(axis=0)
        if task == "classification":
            labels = (pred >= 0.5).astype(float)
            metrics.update({"accuracy": float(accuracy_score(y_test, labels)),
                            "balanced_accuracy": float(balanced_accuracy_score(y_test, labels))})
            if len(set(y_test)) == 2:
                metrics["roc_auc"] = float(roc_auc_score(y_test, pred))
        else:
            metrics.update({"mae": float(mean_absolute_error(y_test, pred)),
                            "rmse": float(root_mean_squared_error(y_test, pred)),
                            "interval_90_coverage": float(np.mean(np.abs(y_test - pred) <= q90))})
    provisional.conformal_q90 = q90
    provisional.metrics = metrics
    return provisional


def _family_medians(rows: list[dict[str, str]]) -> dict[str, dict[str, float]]:
    peaks = [float(row["target"]) for row in rows if row["endpoint"] == "lambda_max_nm"]
    peak = float(np.median(peaks))
    curve = [math.exp(-0.5 * ((wavelength - peak) / 34.0) ** 2) for wavelength in WAVELENGTHS]
    reference = {
        "uvb_auc": trapezoid_auc(WAVELENGTHS, curve, 290.0, 320.0),
        "uva_auc": trapezoid_auc(WAVELENGTHS, curve, 320.0, 400.0),
        "lambda_c_nm": critical_wavelength(WAVELENGTHS, curve),
        "energy_kj_mol": math.nan, "specific_energy_wh_kg": math.nan,
        "log_half_life_h": math.log10(8.0),
    }
    return {family: dict(reference) for family in FAMILIES}


def train_real_reviewers(config: dict[str, Any], data_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
    rows = read_csv(data_path)
    endpoints_present = {row.get("endpoint") for row in rows}
    missing = set(ALL_ENDPOINTS) - endpoints_present
    if missing:
        raise ValueError(f"Missing required real endpoints: {sorted(missing)}")
    unique = {row["smiles"] for row in rows}
    if len(unique) > int(config["project"]["max_training_structures"]):
        raise ValueError("Reviewer training structures exceed configured cap")
    bits = int(config["reviewers"]["fingerprint_bits"])
    members = int(config["reviewers"]["ensemble_size"])
    trees = int(config["reviewers"].get("trees_per_member", 28))
    z = float(config["reviewers"]["confidence_z"])
    destination = Path(output_dir)
    cards = {}
    for purpose, shift in (("reward", 0), ("evaluator", 500_009)):
        endpoint_models = {
            endpoint: _fit_endpoint(rows, endpoint, purpose, bits, members, trees,
                                    int(config["project"]["default_seed"]) + shift + 97 * idx, z)
            for idx, endpoint in enumerate(ALL_ENDPOINTS)
        }
        references = {
            "family_medians": _family_medians(rows),
            "training_data_hash": stable_hash("\n".join(sorted(row["record_id"] for row in rows)), 32),
            "energy_model_status": "unavailable_no_open_molecular_labels",
        }
        metrics = {name: model.metrics for name, model in endpoint_models.items()}
        bundle = RealReviewerBundle(purpose, bits, z, endpoint_models, references, metrics)
        path = destination / purpose / "reviewers.pkl"
        bundle.save(path)
        cards[purpose] = {"path": str(path.resolve()), "algorithm": ("RandomForest" if purpose == "reward" else "ExtraTrees") + " endpoint ensembles",
                          "metrics": metrics}
    metadata = {
        "schema_version": "2.0", "rows": len(rows), "unique_structures": len(unique),
        "evidence_tiers": ["experimental"], "data_mode": "real_endpoint_specific",
        **cards, "independent_instances": True,
        "uncertainty": "90% split-conformal absolute-residual radius, lower-bounded by ensemble spread",
        "energy_model_status": "not_trained; physical GFN2-xTB oracle required",
        "limitations": [
            "The spectral reward is a declared Gaussian band proxy around predicted lambda-max, not a measured spectrum.",
            "M01 kinetics are class-shift evidence and are outside-domain for most generated MOST scaffolds.",
            "U16 is small; uncertain phototoxicity predictions fail closed.",
        ],
    }
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "model_cards.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metadata
