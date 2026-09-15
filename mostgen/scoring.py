from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Sequence

from .chemistry import ChemistryError, descriptors, safety_veto, synthetic_accessibility
from .data import WAVELENGTHS
from .numerics import (
    beer_lambert_transmittance,
    critical_wavelength,
    interval_score,
    lower_confidence_bound,
    sigmoid,
    trapezoid_auc,
    upper_confidence_bound,
    weighted_geometric_mean,
)
from .reviewers import ReviewerBundle


def _auc_uncertainty(wavelengths: Sequence[float], std: Sequence[float], low: float, high: float) -> float:
    # Independent-bin propagation is a conservative-enough transparent proxy
    # for the smoke ensemble. Production models should retain member curves.
    selected = [(w, s) for w, s in zip(wavelengths, std) if low <= w <= high]
    if len(selected) < 2:
        return 0.0
    variance = 0.0
    for (x1, s1), (x2, s2) in zip(selected, selected[1:]):
        dx = x2 - x1
        variance += (0.5 * dx) ** 2 * (s1**2 + s2**2)
    return math.sqrt(variance)


def _lambda_uncertainty(wavelengths: Sequence[float], mean: Sequence[float], std: Sequence[float]) -> float:
    centre = critical_wavelength(wavelengths, mean)
    lower_curve = [max(0.0, value - spread) for value, spread in zip(mean, std)]
    upper_curve = [max(0.0, value + spread) for value, spread in zip(mean, std)]
    return max(abs(centre - critical_wavelength(wavelengths, lower_curve)), abs(centre - critical_wavelength(wavelengths, upper_curve)))


@dataclass
class ScoringContext:
    config: dict[str, Any]
    reviewers: ReviewerBundle

    def review_candidate(
        self, candidate: dict[str, Any], method_id: str, seed: int,
        raw_prediction: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        smiles = candidate["smiles"]
        charged = candidate["charged_smiles"]
        family = candidate["family"]
        veto = safety_veto(smiles, self.config)
        base: dict[str, Any] = {
            **candidate,
            "method_id": method_id,
            "seed": seed,
            "valid": not veto.veto,
            "psoralen_alert": veto.psoralen_alert,
            "known_phototoxic_match": veto.known_phototoxic_match,
            "psoralen_similarity": veto.psoralen_similarity,
            "psoralen_similarity_warning": veto.psoralen_similarity_warning,
            "reactive_alerts": ";".join(veto.reactive_alerts),
            "unsupported_elements": ";".join(veto.unsupported_elements),
            "safety_veto": veto.veto,
            "not_iso_certified": True,
            "selected": False,
            "reviewer_purpose": self.reviewers.purpose,
        }
        if veto.veto:
            base.update({
                "reward": 0.0,
                "reward_pre_diversity": 0.0,
                "joint_uv_pass": False,
                "most_pass": False,
                "safety_pass": False,
                "joint_pass": False,
                "phototoxicity_uncertain": True,
                "ad_spectral": False,
                "ad_most": False,
                "failure_reasons": ";".join(veto.reasons),
                "spectrum_json": "[]",
            })
            return base
        try:
            raw = raw_prediction if raw_prediction is not None else self.reviewers.predict(smiles, family)
        except (ChemistryError, ValueError) as exc:
            base.update({
                "valid": False, "reward": 0.0, "joint_uv_pass": False,
                "most_pass": False, "safety_pass": False, "joint_pass": False,
                "phototoxicity_uncertain": True, "ad_spectral": False,
                "ad_most": False, "failure_reasons": f"reviewer_error:{exc}",
                "spectrum_json": "[]",
            })
            return base

        spectrum_mean = [max(0.0, raw["spectrum"].mean[f"abs_{w}"]) for w in WAVELENGTHS]
        spectrum_std = [max(0.0, raw["spectrum"].std[f"abs_{w}"]) for w in WAVELENGTHS]
        uvb = trapezoid_auc(WAVELENGTHS, spectrum_mean, 290.0, 320.0)
        uva = trapezoid_auc(WAVELENGTHS, spectrum_mean, 320.0, 400.0)
        uvb_std = _auc_uncertainty(WAVELENGTHS, spectrum_std, 290.0, 320.0)
        uva_std = _auc_uncertainty(WAVELENGTHS, spectrum_std, 320.0, 400.0)
        lambda_c = critical_wavelength(WAVELENGTHS, spectrum_mean)
        lambda_std = _lambda_uncertainty(WAVELENGTHS, spectrum_mean, spectrum_std)
        z = float(self.config["reviewers"]["confidence_z"])
        uvb_lcb = lower_confidence_bound(uvb, uvb_std, z)
        uva_lcb = lower_confidence_bound(uva, uva_std, z)
        lambda_lcb = lower_confidence_bound(lambda_c, lambda_std, z)
        medians = raw["family_reference_medians"]
        joint_uv = (
            uvb_lcb >= medians["uvb_auc"]
            and uva_lcb >= medians["uva_auc"]
            and lambda_lcb >= float(self.config["reviewers"]["lambda_c_min_nm"])
        )

        most_mean, most_std = raw["most"].mean, raw["most"].std
        energy = most_mean["energy_kj_mol"]
        energy_std = most_std["energy_kj_mol"]
        energy_lcb = lower_confidence_bound(energy, energy_std, z)
        specific = most_mean["specific_energy_wh_kg"]
        specific_std = most_std["specific_energy_wh_kg"]
        half_log = most_mean["log_half_life_h"]
        half_log_std = most_std["log_half_life_h"]
        half_hours = 10.0 ** half_log
        half_low = 10.0 ** lower_confidence_bound(half_log, half_log_std, z)
        half_high = 10.0 ** upper_confidence_bound(half_log, half_log_std, z)
        half_window = self.config["reviewers"]["half_life_window_hours"]
        most_pass = (
            energy_lcb > 0.0
            and energy_lcb >= medians["energy_kj_mol"]
            and half_window[0] <= half_hours <= half_window[1]
        )

        safety_mean, safety_std = raw["safety"].mean, raw["safety"].std
        photo = min(1.0, max(0.0, safety_mean["phototoxicity_probability"]))
        photo_std = safety_std["phototoxicity_probability"]
        photo_upper = min(1.0, upper_confidence_bound(photo, photo_std, z))
        uncertain_band = self.config["reviewers"]["uncertain_probability_band"]
        photo_uncertain = (
            uncertain_band[0] <= photo <= uncertain_band[1]
            or (photo - z * photo_std) <= uncertain_band[1] <= photo_upper
        )
        kp = safety_mean["kp_log_cm_s"]
        kp_std = safety_std["kp_log_cm_s"]
        kp_upper = upper_confidence_bound(kp, kp_std, z)
        sa = synthetic_accessibility(smiles)
        ad_threshold = float(self.config["reviewers"]["ad_similarity_threshold"])
        ad_spectral = raw["ad_spectral_similarity"] >= ad_threshold
        ad_most = raw["ad_most_similarity"] >= ad_threshold
        safety_pass = (
            not photo_uncertain
            and photo_upper <= float(self.config["reviewers"]["phototoxicity_max_probability"])
            and kp_upper <= float(self.config["reviewers"]["kp_max_log_cm_s"])
            and sa <= 5.0
        )

        floor = float(self.config["reward"]["floor"])
        components = {
            "uvb": sigmoid(uvb_lcb, medians["uvb_auc"], max(0.5, medians["uvb_auc"] * 0.12)),
            "uva": sigmoid(uva_lcb, medians["uva_auc"], max(0.5, medians["uva_auc"] * 0.12)),
            "lambda_c": sigmoid(lambda_lcb, float(self.config["reviewers"]["lambda_c_min_nm"]), 5.0),
            "energy": sigmoid(energy_lcb, medians["energy_kj_mol"], max(2.0, medians["energy_kj_mol"] * 0.10)),
            "half_life": interval_score(half_log, math.log10(half_window[0]), math.log10(half_window[1]), 0.13),
            "phototoxicity": sigmoid(float(self.config["reviewers"]["phototoxicity_max_probability"]) - photo_upper, 0.0, 0.07),
            "permeation": sigmoid(float(self.config["reviewers"]["kp_max_log_cm_s"]) - kp_upper, 0.0, 0.25),
            "sa": sigmoid(5.0 - sa, 0.0, 0.75),
            "ad": min(1.0, min(raw["ad_spectral_similarity"], raw["ad_most_similarity"]) / ad_threshold),
        }
        weights = self.config["reward"]["weights"]
        reward = weighted_geometric_mean(components, weights, floor)
        stage_rewards = {
            "chemistry": 1.0,
            "spectrum": weighted_geometric_mean({k: components[k] for k in ("uvb", "uva", "lambda_c")}, weights, floor),
            "most": weighted_geometric_mean({k: components[k] for k in ("energy", "half_life")}, weights, floor),
            "safety": weighted_geometric_mean({k: components[k] for k in ("phototoxicity", "permeation", "sa", "ad")}, weights, floor),
        }
        failures = []
        if not joint_uv:
            failures.append("uv_joint_lcb_or_lambda")
        if not most_pass:
            failures.append("most_energy_or_half_life")
        if not ad_spectral:
            failures.append("spectral_out_of_domain")
        if not ad_most:
            failures.append("most_out_of_domain")
        if photo_uncertain:
            failures.append("phototoxicity_uncertain")
        elif photo_upper > float(self.config["reviewers"]["phototoxicity_max_probability"]):
            failures.append("phototoxicity_risk")
        if kp_upper > float(self.config["reviewers"]["kp_max_log_cm_s"]):
            failures.append("permeation_risk")
        if sa > 5.0:
            failures.append("sa_proxy")
        joint_pass = bool(joint_uv and most_pass and safety_pass and ad_spectral and ad_most)
        d = descriptors(smiles)
        base.update({
            "reference_uvb_median": medians["uvb_auc"], "reference_uva_median": medians["uva_auc"],
            "reference_energy_median": medians["energy_kj_mol"],
            "uvb_auc": uvb, "uvb_auc_uncertainty": uvb_std, "uvb_auc_lcb": uvb_lcb,
            "uva_auc": uva, "uva_auc_uncertainty": uva_std, "uva_auc_lcb": uva_lcb,
            "lambda_c_nm": lambda_c, "lambda_c_uncertainty": lambda_std, "lambda_c_lcb": lambda_lcb,
            "uvb_transmittance": beer_lambert_transmittance(spectrum_mean[:7], 1.0),
            "uva_transmittance": beer_lambert_transmittance(spectrum_mean[6:], 1.0),
            "energy_kj_mol": energy, "energy_uncertainty": energy_std, "energy_lcb": energy_lcb,
            "specific_energy_wh_kg": specific, "specific_energy_uncertainty": specific_std,
            "log_half_life_h": half_log, "half_life_uncertainty_log": half_log_std,
            "half_life_h": half_hours, "half_life_lcb_h": half_low, "half_life_ucb_h": half_high,
            "kp_log_cm_s": kp, "kp_uncertainty": kp_std, "kp_ucb": kp_upper,
            "phototoxicity_probability": photo, "phototoxicity_uncertainty": photo_std,
            "phototoxicity_ucb": photo_upper, "phototoxicity_uncertain": photo_uncertain,
            "similarity_D_A": raw["ad_spectral_similarity"], "similarity_D_B": raw["ad_most_similarity"],
            "ad_spectral": ad_spectral, "ad_most": ad_most,
            "sa_score": sa, "mol_wt": d["mol_wt"], "logp": d["logp"], "tpsa": d["tpsa"],
            "joint_uv_pass": joint_uv, "most_pass": most_pass, "safety_pass": safety_pass,
            "joint_pass": joint_pass, "reward": reward, "reward_pre_diversity": reward,
            "reward_components_json": json.dumps(components, sort_keys=True),
            "stage_rewards_json": json.dumps(stage_rewards, sort_keys=True),
            "failure_reasons": ";".join(failures),
            "spectrum_json": json.dumps([{"wavelength_nm": w, "absorbance": round(a, 6), "uncertainty": round(s, 6)} for w, a, s in zip(WAVELENGTHS, spectrum_mean, spectrum_std)]),
        })
        return base

    def review_candidates(self, candidates: list[dict[str, Any]], method_id: str, seed: int) -> list[dict[str, Any]]:
        safe_indices = []
        safe_molecules = []
        for index, candidate in enumerate(candidates):
            if not safety_veto(candidate["smiles"], self.config).veto:
                safe_indices.append(index)
                safe_molecules.append((candidate["smiles"], candidate["family"]))
        predictions = self.reviewers.predict_many(safe_molecules)
        prediction_by_index = dict(zip(safe_indices, predictions))
        return [
            self.review_candidate(candidate, method_id, seed, prediction_by_index.get(index))
            for index, candidate in enumerate(candidates)
        ]
