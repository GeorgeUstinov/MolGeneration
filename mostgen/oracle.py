from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Any, Iterable

from rdkit import Chem
from rdkit.Chem import AllChem

from .data import write_csv
from .provenance import stable_hash


def availability(config: dict[str, Any]) -> dict[str, Any]:
    tools = {
        "xtb": shutil.which(config["production"]["xtb_executable"]),
        "stda": shutil.which(config["production"]["stda_executable"]),
        "xtb4stda": shutil.which(config["production"].get("xtb4stda_executable", "xtb4stda")),
    }
    return {"tools": tools, "ready": all(tools.values())}


def _write_low_energy_xyz(smiles: str, path: Path, seed: int) -> dict[str, Any]:
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    params = AllChem.ETKDGv3()
    params.randomSeed = int(seed & 0x7FFFFFFF)
    conformers = list(AllChem.EmbedMultipleConfs(mol, numConfs=8, params=params))
    if not conformers:
        raise RuntimeError("RDKit conformer embedding failed")
    energies = []
    for conf_id in conformers:
        try:
            if AllChem.MMFFHasAllMoleculeParams(mol):
                props = AllChem.MMFFGetMoleculeProperties(mol)
                forcefield = AllChem.MMFFGetMoleculeForceField(mol, props, confId=conf_id)
            else:
                forcefield = AllChem.UFFGetMoleculeForceField(mol, confId=conf_id)
            forcefield.Minimize(maxIts=500)
            energies.append((float(forcefield.CalcEnergy()), conf_id))
        except Exception:
            continue
    if not energies:
        raise RuntimeError("No conformer could be force-field minimized")
    energy, best = min(energies)
    Chem.MolToXYZFile(mol, str(path), confId=int(best))
    return {"conformers": len(conformers), "selected_forcefield_energy": energy}


def prepare_oracle_queue(
    rows: Iterable[dict[str, Any]],
    output_dir: str | Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    destination = Path(output_dir)
    structures = destination / "structures"
    structures.mkdir(parents=True, exist_ok=True)
    status = availability(config)
    queue = []
    errors = []
    for rank, row in enumerate(rows, start=1):
        candidate_id = row.get("candidate_id") or stable_hash(row["smiles"], 20)
        try:
            ground_path = structures / f"{candidate_id}_ground.xyz"
            charged_path = structures / f"{candidate_id}_charged.xyz"
            ground_info = _write_low_energy_xyz(row["smiles"], ground_path, int(config["project"]["default_seed"]) + rank)
            charged_info = _write_low_energy_xyz(row["charged_smiles"], charged_path, int(config["project"]["default_seed"]) + 100_000 + rank)
            queue.append({
                "oracle_rank": rank, "candidate_id": candidate_id, "family": row["family"],
                "smiles": row["smiles"], "charged_smiles": row["charged_smiles"],
                "ground_xyz": str(ground_path.resolve()), "charged_xyz": str(charged_path.resolve()),
                "ground_conformers": ground_info["conformers"], "charged_conformers": charged_info["conformers"],
                "xtb_energy_command_ground": f"{config['production']['xtb_executable']} {ground_path.name} --gfn 2 --opt tight --json",
                "xtb_energy_command_charged": f"{config['production']['xtb_executable']} {charged_path.name} --gfn 2 --opt tight --json",
                "spectral_commands": "xtb4stda <xyz> then stda -xtb; broaden transitions over 290-400 nm",
                "oracle_status": "queued_external" if status["ready"] else "not_run_tools_unavailable",
            })
        except Exception as exc:
            errors.append({"candidate_id": candidate_id, "error": str(exc)})
    write_csv(destination / "oracle_queue.csv", queue)
    manifest = {
        "schema_version": "1.0", "availability": status, "queued": len(queue), "errors": errors,
        "calculation_contract": {
            "energy": "independent GFN2-xTB optimization and ground/charged energy difference",
            "spectrum": "sTDA-xTB transitions broadened over 290-400 nm",
            "inside_rl_loop": False,
            "automatic_proxy_substitution": False,
        },
        "status": "queued_external" if status["ready"] else "not_run_external_tools_unavailable",
    }
    (destination / "oracle_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest
