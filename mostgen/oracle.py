from __future__ import annotations

import importlib.util
import json
import math
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors

from .data import WAVELENGTHS, write_csv
from .numerics import beer_lambert_transmittance, critical_wavelength, trapezoid_auc
from .provenance import stable_hash


ROOT = Path(__file__).resolve().parents[1]


def _resolved_executable(value: str) -> str | None:
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / path
    if path.is_file() and os.access(path, os.X_OK):
        return str(path.resolve())
    return shutil.which(value)


def availability(config: dict[str, Any]) -> dict[str, Any]:
    production = config["production"]
    tools = {
        "xtb_python": importlib.util.find_spec("xtb") is not None,
        "ase": importlib.util.find_spec("ase") is not None,
        "stda": _resolved_executable(production["stda_executable"]),
        "xtb4stda": _resolved_executable(production.get("xtb4stda_executable", "xtb4stda")),
    }
    return {"tools": tools, "energy_ready": bool(tools["xtb_python"] and tools["ase"]),
            "spectrum_ready": bool(tools["stda"] and tools["xtb4stda"]),
            "ready": all(bool(value) for value in tools.values())}


def _conformers(smiles: str, seed: int, count: int) -> tuple[Chem.Mol, list[tuple[float, int, str]]]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("invalid_smiles")
    mol = Chem.AddHs(mol)
    conformer_ids: list[int] = []
    for retry, random_coordinates in enumerate((False, True, True)):
        params = AllChem.ETKDGv3()
        params.randomSeed = int((seed + 7919 * retry) & 0x7FFFFFFF)
        params.useRandomCoords = random_coordinates
        params.enforceChirality = True
        params.pruneRmsThresh = 0.35
        conformer_ids = list(AllChem.EmbedMultipleConfs(mol, numConfs=count, params=params))
        if conformer_ids:
            break
    if not conformer_ids:
        raise RuntimeError("rdkit_conformer_embedding_failed_after_retries")
    energies = []
    for conf_id in conformer_ids:
        try:
            if AllChem.MMFFHasAllMoleculeParams(mol):
                properties = AllChem.MMFFGetMoleculeProperties(mol)
                forcefield = AllChem.MMFFGetMoleculeForceField(mol, properties, confId=conf_id)
                method = "MMFF94"
            else:
                forcefield = AllChem.UFFGetMoleculeForceField(mol, confId=conf_id)
                method = "UFF"
            forcefield.Minimize(maxIts=800)
            energies.append((float(forcefield.CalcEnergy()), int(conf_id), method))
        except Exception:
            continue
    if not energies:
        raise RuntimeError("forcefield_conformer_minimization_failed")
    return mol, sorted(energies)


def _xtb_optimize(smiles: str, workdir: Path, stem: str, seed: int, config: dict[str, Any]) -> dict[str, Any]:
    from ase import Atoms
    from ase.io import write as ase_write
    from ase.optimize import BFGS
    from xtb.ase.calculator import XTB

    mol, ranked = _conformers(smiles, seed, int(config["oracle"]["conformers"]))
    candidates = ranked[: min(3, len(ranked))]
    results = []
    net_charge = int(Chem.GetFormalCharge(mol))
    electrons = sum(atom.GetAtomicNum() for atom in mol.GetAtoms()) - net_charge
    uhf = electrons % 2
    for ff_energy, conf_id, ff_method in candidates:
        conformer = mol.GetConformer(conf_id)
        atoms = Atoms(numbers=[atom.GetAtomicNum() for atom in mol.GetAtoms()],
                      positions=np.asarray(conformer.GetPositions(), dtype=float))
        charges = np.zeros(len(atoms)); charges[0] = net_charge
        moments = np.zeros(len(atoms)); moments[0] = uhf
        atoms.set_initial_charges(charges)
        atoms.set_initial_magnetic_moments(moments)
        atoms.calc = XTB(method="GFN2-xTB", accuracy=1.0, max_iterations=250)
        optimizer = BFGS(atoms, logfile=None)
        optimizer.run(fmax=float(config["oracle"]["xtb_fmax_ev_a"]), steps=int(config["oracle"]["xtb_max_steps"]))
        results.append((float(atoms.get_potential_energy()), atoms, ff_energy, ff_method,
                        optimizer.converged(), optimizer.get_number_of_steps()))
    energy_ev, atoms, ff_energy, ff_method, converged, steps = min(results, key=lambda item: item[0])
    xyz = workdir / f"{stem}.xyz"
    ase_write(xyz, atoms, format="xyz")
    return {"energy_ev": energy_ev, "xyz": str(xyz), "conformers_embedded": len(ranked),
            "conformers_xtb_optimized": len(candidates), "forcefield": ff_method,
            "selected_forcefield_energy": ff_energy, "xtb_converged": bool(converged),
            "xtb_steps": int(steps), "formal_charge": net_charge, "uhf": int(uhf)}


def _stda_spectrum(xyz: Path, workdir: Path, stem: str, config: dict[str, Any]) -> dict[str, Any]:
    production = config["production"]
    run_dir = workdir / f"stda_{stem}"
    run_dir.mkdir(parents=True, exist_ok=True)
    local_xyz = run_dir / "structure.xyz"
    shutil.copy2(xyz, local_xyz)
    env = dict(os.environ)
    home = Path(production["xtb4stda_home"])
    if not home.is_absolute():
        home = ROOT / home
    env.update({"XTB4STDAHOME": str(home.resolve()), "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"})
    xtb4stda = _resolved_executable(production["xtb4stda_executable"])
    stda = _resolved_executable(production["stda_executable"])
    first = subprocess.run([str(xtb4stda), local_xyz.name], cwd=run_dir, env=env, text=True,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=300, check=False)
    if first.returncode != 0 or not (run_dir / "wfn.xtb").exists():
        raise RuntimeError(f"xtb4stda_failed:{first.returncode}")
    second = subprocess.run([str(stda), "-xtb"], cwd=run_dir, env=env, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=300, check=False)
    if second.returncode != 0 or not (run_dir / "tda.dat").exists():
        raise RuntimeError(f"stda_failed:{second.returncode}")
    transitions = []
    in_table = False
    for line in second.stdout.splitlines():
        if "state    eV      nm" in line:
            in_table = True
            continue
        if in_table:
            fields = line.split()
            if len(fields) < 4 or not fields[0].isdigit():
                if transitions:
                    break
                continue
            try:
                transitions.append((float(fields[1]), float(fields[2]), max(0.0, float(fields[3]))))
            except ValueError:
                continue
    width_ev = float(config["oracle"]["spectral_broadening_ev"])
    curve = []
    for wavelength in WAVELENGTHS:
        energy_ev = 1239.841984 / wavelength
        curve.append(sum(oscillator * math.exp(-0.5 * ((energy_ev - transition_ev) / width_ev) ** 2)
                         for transition_ev, _, oscillator in transitions))
    loading = float(config["reviewers"].get("beer_lambert_loading_scale", 1.0))
    uvb_curve = [value for wavelength, value in zip(WAVELENGTHS, curve) if 290 <= wavelength <= 320]
    uva_curve = [value for wavelength, value in zip(WAVELENGTHS, curve) if 320 <= wavelength <= 400]
    return {"transitions": transitions, "curve": curve,
            "uvb_auc": trapezoid_auc(WAVELENGTHS, curve, 290.0, 320.0),
            "uva_auc": trapezoid_auc(WAVELENGTHS, curve, 320.0, 400.0),
            "lambda_c_nm": critical_wavelength(WAVELENGTHS, curve),
            "uvb_transmittance": beer_lambert_transmittance(uvb_curve, loading),
            "uva_transmittance": beer_lambert_transmittance(uva_curve, loading),
            "beer_lambert_loading_scale": loading,
            "output": str((run_dir / "tda.dat").resolve())}


def evaluate_pair(row: dict[str, Any], destination: Path, config: dict[str, Any], rank: int) -> dict[str, Any]:
    candidate_id = row.get("candidate_id") or stable_hash(row["smiles"], 20)
    workdir = destination / candidate_id
    workdir.mkdir(parents=True, exist_ok=True)
    seed = int(config["project"]["default_seed"]) + rank * 1009
    ground = _xtb_optimize(row["smiles"], workdir, "ground", seed, config)
    charged = _xtb_optimize(row["charged_smiles"], workdir, "charged", seed + 500_003, config)
    delta_kj_mol = (charged["energy_ev"] - ground["energy_ev"]) * 96.4853321233
    molecular_weight = float(Descriptors.MolWt(Chem.MolFromSmiles(row["smiles"])))
    ground_spectrum = _stda_spectrum(Path(ground["xyz"]), workdir, "ground", config)
    charged_spectrum = _stda_spectrum(Path(charged["xyz"]), workdir, "charged", config)
    return {"candidate_id": candidate_id, "oracle_rank": rank, "family": row["family"],
            "gfn2_delta_e_kj_mol": delta_kj_mol,
            "gfn2_specific_energy_wh_kg": delta_kj_mol * 277.7777778 / molecular_weight,
            "gfn2_ground_energy_ev": ground["energy_ev"], "gfn2_charged_energy_ev": charged["energy_ev"],
            "ground_xtb_converged": ground["xtb_converged"], "charged_xtb_converged": charged["xtb_converged"],
            "stda_uvb_auc": ground_spectrum["uvb_auc"], "stda_uva_auc": ground_spectrum["uva_auc"],
            "stda_lambda_c_nm": ground_spectrum["lambda_c_nm"],
            "stda_uvb_transmittance": ground_spectrum["uvb_transmittance"],
            "stda_uva_transmittance": ground_spectrum["uva_transmittance"],
            "stda_beer_lambert_loading_scale": ground_spectrum["beer_lambert_loading_scale"],
            "stda_ground_spectrum_json": json.dumps(list(zip(WAVELENGTHS, ground_spectrum["curve"]))),
            "stda_charged_spectrum_json": json.dumps(list(zip(WAVELENGTHS, charged_spectrum["curve"]))),
            "stda_ground_transitions_json": json.dumps(ground_spectrum["transitions"]),
            "stda_charged_transitions_json": json.dumps(charged_spectrum["transitions"]),
            "oracle_status": "complete"}


def prepare_oracle_queue(rows: Iterable[dict[str, Any]], output_dir: str | Path,
                         config: dict[str, Any]) -> dict[str, Any]:
    destination = Path(output_dir)
    calculations = destination / "calculations"
    calculations.mkdir(parents=True, exist_ok=True)
    status = availability(config)
    rows = list(rows)
    queue, results, errors = [], [], []
    run = bool(config.get("oracle", {}).get("run_automatically")) and status["ready"]
    limit = min(len(rows), int(config.get("oracle", {}).get("max_candidates", len(rows))))
    for rank, row in enumerate(rows, start=1):
        candidate_id = row.get("candidate_id") or stable_hash(row["smiles"], 20)
        queue.append({"oracle_rank": rank, "candidate_id": candidate_id, "family": row["family"],
                      "smiles": row["smiles"], "charged_smiles": row["charged_smiles"],
                      "oracle_status": "scheduled" if run and rank <= limit else "not_run_by_configuration"})
        if run and rank <= limit:
            try:
                results.append(evaluate_pair(row, calculations, config, rank))
            except Exception as exc:
                errors.append({"candidate_id": candidate_id, "error": f"{type(exc).__name__}:{exc}"})
    write_csv(destination / "oracle_queue.csv", queue)
    write_csv(destination / "oracle_results.csv", results)
    complete = run and limit > 0 and len(results) == limit and not errors
    manifest = {"schema_version": "2.0", "availability": status, "queued": len(queue),
                "requested": limit if run else 0, "completed": len(results), "errors": errors,
                "calculation_contract": {
                    "energy": "multi-conformer RDKit -> GFN2-xTB/ASE geometry optimization; charged-ground DeltaE proxy",
                    "spectrum": "official xtb4stda 1.0 + sTDA 1.6.1 transitions, Gaussian broadening over 290-400 nm",
                    "inside_rl_loop": False, "automatic_proxy_substitution": False},
                "status": "complete" if complete else ("partially_complete" if results else "not_run")}
    (destination / "oracle_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest
