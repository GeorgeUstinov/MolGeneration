from pathlib import Path
import subprocess

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
BASE_BUILDER = ROOT / "scripts" / "build_uv_most_joint_notebook.py"
NOTEBOOK = ROOT / "UV_Photoswitch_Application_RL.ipynb"

# Start from the reproducible joint notebook source, then append the applied
# pair-aware photoswitch stage. The original joint notebook remains untouched.
subprocess.run(["python", str(BASE_BUILDER)], cwd=ROOT, check=True)
base_path = ROOT / "UV_MOST_Joint_RL.ipynb"
nb = nbf.read(base_path, as_version=4)

def md(text):
    return nbf.v4.new_markdown_cell(text.strip())

def code(text):
    return nbf.v4.new_code_cell(text.strip())

nb.cells.extend([
    md(
        """
## Applied photoswitch stage

The previous joint RL model is now fine-tuned with an explicit photoswitch
constraint. A proposal must contain an azo `N=N` motif and produce two distinct
RDKit stereochemical states, E and Z. M01 E/Z transition labels train new
transition-wavelength experts. Half-life and UV absorption remain the existing
joint experts.

The output is a candidate photoswitch pair, not proof of PSS, quantum yield,
stored energy, or reversible cycling. Those endpoints are exported as missing
and require physical validation.
        """
    ),
    code(
        r"""
PHOTOSWITCH_OUT = ROOT / "outputs" / "uv_most_photoswitch"
PHOTOSWITCH_MODEL_DIR = PHOTOSWITCH_OUT / "models"
PHOTOSWITCH_OUT.mkdir(parents=True, exist_ok=True)
PHOTOSWITCH_MODEL_DIR.mkdir(parents=True, exist_ok=True)
M01_PHOTO = pd.read_csv(m01_path, low_memory=False)
E_COL = "E isomer pi-pi* wavelength in nm"
Z_COL = "Z isomer pi-pi* wavelength in nm"
M01_PHOTO["canonical_smiles"] = M01_PHOTO["SMILES"].map(standardize_smiles)
M01_PHOTO["e_lambda_nm"] = pd.to_numeric(M01_PHOTO[E_COL], errors="coerce")
M01_PHOTO["z_lambda_nm"] = pd.to_numeric(M01_PHOTO[Z_COL], errors="coerce")
M01_PHOTO = M01_PHOTO.dropna(subset=["canonical_smiles"]).drop_duplicates("canonical_smiles")

from sklearn.ensemble import RandomForestRegressor

def fit_transition_models(target, seed):
    frame = M01_PHOTO.dropna(subset=[target])
    x = fingerprint_matrix(frame["canonical_smiles"].tolist())
    y = frame[target].to_numpy(float)
    models = []
    for member in range(3):
        model = RandomForestRegressor(
            n_estimators=100, min_samples_leaf=2, max_features="sqrt",
            bootstrap=True, n_jobs=-1, random_state=seed + 1009 * member,
        )
        model.fit(x, y)
        models.append(model)
    return models

photoswitch_E_models = fit_transition_models("e_lambda_nm", BASE_SEED + 501)
photoswitch_Z_models = fit_transition_models("z_lambda_nm", BASE_SEED + 601)
print(
    "transition labels:",
    int(M01_PHOTO.e_lambda_nm.notna().sum()),
    int(M01_PHOTO.z_lambda_nm.notna().sum()),
)

def pair_states(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    directional_pairs = (
        ("/N=N/", "/N=N\\"),
        ("/N=N\\", "/N=N/"),
        ("\\N=N\\", "\\N=N/"),
        ("\\N=N/", "\\N=N\\"),
    )
    for source, replacement in directional_pairs:
        if source in canonical:
            alternate = canonical.replace(source, replacement, 1)
            state_1 = Chem.MolFromSmiles(canonical, sanitize=True)
            state_2 = Chem.MolFromSmiles(alternate, sanitize=True)
            if state_1 is not None and state_2 is not None:
                first = Chem.MolToSmiles(state_1, canonical=True, isomericSmiles=True)
                second = Chem.MolToSmiles(state_2, canonical=True, isomericSmiles=True)
                if first != second:
                    return first, second
    for bond in mol.GetBonds():
        left, right = bond.GetBeginAtom(), bond.GetEndAtom()
        if bond.GetBondType() != Chem.BondType.DOUBLE or left.GetSymbol() != "N" or right.GetSymbol() != "N":
            continue
        left_neighbors = [a.GetIdx() for a in left.GetNeighbors() if a.GetIdx() != right.GetIdx()]
        right_neighbors = [a.GetIdx() for a in right.GetNeighbors() if a.GetIdx() != left.GetIdx()]
        if not left_neighbors or not right_neighbors:
            continue
        def state(stereo):
            copy_mol = Chem.RWMol(mol)
            copy_bond = copy_mol.GetBondWithIdx(bond.GetIdx())
            copy_bond.SetStereoAtoms(left_neighbors[0], right_neighbors[0])
            copy_bond.SetStereo(stereo)
            Chem.AssignStereochemistry(copy_mol, cleanIt=True, force=True)
            state_smiles = Chem.MolToSmiles(copy_mol, isomericSmiles=True)
            state_mol = Chem.MolFromSmiles(state_smiles, sanitize=True)
            return Chem.MolToSmiles(state_mol, canonical=True, isomericSmiles=True) if state_mol else None
        e_state = state(Chem.BondStereo.STEREOE)
        z_state = state(Chem.BondStereo.STEREOZ)
        if e_state and z_state and e_state != z_state:
            return e_state, z_state
    return None

def photoswitch_filter(smiles):
    row = {
        "input_smiles": smiles, "canonical_smiles": None, "base_smiles": None,
        "state_A_smiles": None, "state_B_smiles": None, "pair_valid": False,
        "rdkit_valid": False, "passes_hard_filters": False,
        "filter_reason": "invalid_smiles", "molecular_weight": np.nan,
        "sa_score": np.nan, "qed": np.nan, "rotatable_bonds": np.nan,
        "psoralen_similarity": np.nan,
    }
    canonical = standardize_smiles(smiles)
    if not canonical:
        return row
    mol = Chem.MolFromSmiles(canonical)
    row.update({"canonical_smiles": canonical, "base_smiles": canonical, "rdkit_valid": True})
    if len(Chem.GetMolFrags(mol)) != 1:
        row["filter_reason"] = "multiple_fragments"
        return row
    if not mol.HasSubstructMatch(Chem.MolFromSmarts("[N]=[N]")):
        row["filter_reason"] = "no_azo_photoswitch_motif"
        return row
    row["molecular_weight"] = float(Descriptors.MolWt(mol))
    row["sa_score"] = float(sascorer.calculateScore(mol))
    row["qed"] = float(QED.qed(mol))
    row["rotatable_bonds"] = int(Lipinski.NumRotatableBonds(mol))
    row["psoralen_similarity"] = float(max(DataStructs.BulkTanimotoSimilarity(bit_fingerprint(canonical), KNOWN_PSORALEN_FPS), default=0.0))
    checks = [
        (mol.GetNumHeavyAtoms() < 8, "too_few_heavy_atoms"),
        (mol.GetNumHeavyAtoms() > 55, "too_many_heavy_atoms"),
        (not 120.0 <= row["molecular_weight"] <= 650.0, "molecular_weight"),
        (row["rotatable_bonds"] > 14, "too_many_rotatable_bonds"),
        (row["sa_score"] > 6.0, "synthetic_accessibility"),
        (row["qed"] < 0.10, "very_low_qed"),
        (abs(Chem.GetFormalCharge(mol)) > 1, "formal_charge"),
        (any(atom.GetNumRadicalElectrons() for atom in mol.GetAtoms()), "radical"),
        (any(atom.GetAtomicNum() not in ALLOWED_ATOMIC_NUMBERS for atom in mol.GetAtoms()), "disallowed_element"),
    ]
    for failed, reason in checks:
        if failed:
            row["filter_reason"] = reason
            return row
    if any(mol.HasSubstructMatch(query) for query in FUROCOUMARIN_QUERIES.values()):
        row["filter_reason"] = "psoralen_or_angelicin"
        return row
    if row["psoralen_similarity"] >= 0.45:
        row["filter_reason"] = "psoralen_similarity"
        return row
    # PAINS azo_A(324) is the intended photoswitch motif; other PAINS remain hard vetoes.
    matches = PAINS_CATALOG.GetMatches(mol)
    if any(not match.GetDescription().startswith("azo_A") for match in matches):
        row["filter_reason"] = "PAINS_alert"
        return row
    for name, query in REACTIVE_QUERIES.items():
        if mol.HasSubstructMatch(query):
            row["filter_reason"] = f"reactive:{name}"
            return row
    pair = pair_states(canonical)
    if pair is None:
        row.update({"passes_hard_filters": False, "filter_reason": "no_distinct_EZ_pair"})
        return row
    row.update({
        "state_A_smiles": pair[0], "state_B_smiles": pair[1],
        "pair_valid": True, "passes_hard_filters": True, "filter_reason": "ok",
    })
    return row

def model_predict(models, smiles):
    values = np.stack([m.predict(fingerprint_matrix(smiles)) for m in models], axis=0)
    return values.mean(axis=0), values.std(axis=0, ddof=0)

def photoswitch_reward_batch(smiles_values):
    rows = [photoswitch_filter(smiles) for smiles in smiles_values]
    eligible = [i for i, row in enumerate(rows) if row.get("passes_hard_filters") and row.get("canonical_smiles") not in KNOWN_UNION]
    if eligible:
        bases = [rows[i]["canonical_smiles"] for i in eligible]
        e_mean, e_unc = model_predict(photoswitch_E_models, bases)
        z_mean, z_unc = model_predict(photoswitch_Z_models, bases)
        h_mean, h_unc = bundle_predict(reward_A, fingerprint_matrix(bases))
        uv_mean, uv_unc = bundle_predict(reward_B, fingerprint_matrix(bases))
        for i, em, eu, zm, zu, hm, hu, um, uu in zip(eligible, e_mean, e_unc, z_mean, z_unc, h_mean, h_unc, uv_mean, uv_unc):
            row = rows[i]
            delta = abs(float(em - zm))
            switch_score = sigmoid((delta - 20.0) / 8.0)
            e_score = interval_score(float(em), float(eu), 300.0, 400.0, 14.0)
            h_score = interval_score(float(hm), float(hu), A_LOW, A_HIGH, 0.18)
            uv_score = interval_score(float(um), float(uu), 300.0, 400.0, 14.0)
            uncertainty_score = math.exp(-float(hu) / 1.2) * math.exp(-float(uu) / 70.0)
            sa_score = math.exp(-0.30 * max(0.0, float(row["sa_score"]) - 3.0))
            components = [switch_score, e_score, h_score, uv_score, uncertainty_score, sa_score]
            row.update({
                "pred_E_lambda_nm": float(em), "pred_Z_lambda_nm": float(zm),
                "pred_E_uncertainty_nm": float(eu), "pred_Z_uncertainty_nm": float(zu),
                "pred_delta_lambda_nm": delta,
                "pred_half_life_log10_h": float(hm), "pred_half_life_h": float(10.0 ** hm),
                "pred_half_life_uncertainty_log10_h": float(hu),
                "pred_uv_lambda_nm": float(um), "pred_uv_uncertainty_nm": float(uu),
                "switch_proxy_score": float(switch_score),
                "energy_endpoint_available": False, "PSS_available": False,
                "quantum_yield_available": False,
                "reward": float(np.prod(np.clip(components, 1.0e-12, 1.0)) ** (1.0 / len(components))),
            })
    for row in rows:
        row.setdefault("reward", 0.0)
        row.setdefault("energy_endpoint_available", False)
        row.setdefault("PSS_available", False)
        row.setdefault("quantum_yield_available", False)
    return pd.DataFrame(rows)
"""
    ),
    md("## Fine-tune the existing RL agents with the pair-aware reward"),
    code(
        r"""
def photoswitch_finetune(agent, seed):
    seed_everything(seed + 300000)
    agent = copy.deepcopy(agent).to(DEVICE)
    for parameter in agent.parameters():
        parameter.requires_grad_(True)
    optimizer = torch.optim.Adam(agent.parameters(), lr=2.0e-4)
    history = []
    archive = set()
    for step in range(1, 41):
        smiles, sequences = sample_token_sequences(agent, 64, 0.90)
        scored = photoswitch_reward_batch(smiles)
        rewards_np = scored["reward"].to_numpy(np.float32)
        for i, canonical in enumerate(scored["canonical_smiles"]):
            if canonical and canonical in archive:
                rewards_np[i] *= 0.20
            if canonical and rewards_np[i] > 0:
                archive.add(canonical)
        rewards = torch.tensor(rewards_np, dtype=torch.float32, device=DEVICE)
        mean_logp, mean_kl, entropy = sequence_policy_terms(agent, prior, sequences)
        advantage = (rewards - rewards.mean()) / (rewards.std(unbiased=False) + 1.0e-6)
        loss = -(advantage.detach() * mean_logp).mean() + 0.035 * mean_kl.mean() - 0.002 * entropy.mean()
        optimizer.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(agent.parameters(), 1.0); optimizer.step()
        history.append({"seed": seed, "step": step, "mean_reward": float(rewards.mean()), "photoswitch_fraction": float(scored["pair_valid"].mean()), "archive": len(archive)})
    torch.save({"state_dict": {k: v.detach().cpu() for k, v in agent.state_dict().items()}, "vocabulary": vocabulary, "seed": seed}, PHOTOSWITCH_MODEL_DIR / f"agent_seed{seed}.pt")
    return agent, pd.DataFrame(history)

photoswitch_agents = {}
photoswitch_histories = []
for seed in RUN_SEEDS:
    photoswitch_agents[seed], history = photoswitch_finetune(agents[("M1_joint_rl", seed)], seed)
    photoswitch_histories.append(history)
pd.concat(photoswitch_histories, ignore_index=True).to_csv(PHOTOSWITCH_OUT / "photoswitch_rl_history.csv", index=False)
"""
    ),
    md("## Final E/Z photoswitch generation"),
    code(
        r"""
def sample_photoswitch_run(model, seed, count=12000):
    seed_everything(seed + 500000)
    proposals = []
    remaining = count
    while remaining:
        batch = min(256, remaining)
        smiles, _ = sample_token_sequences(model, batch, 0.90)
        proposals.extend(smiles)
        remaining -= batch
    scored = photoswitch_reward_batch(proposals)
    scored["method_id"] = "photoswitch_pair_rl"
    scored["seed"] = seed
    return scored

photo_parts = [sample_photoswitch_run(photoswitch_agents[seed], seed) for seed in RUN_SEEDS]
photo_audit = pd.concat(photo_parts, ignore_index=True)
photo_audit.to_csv(PHOTOSWITCH_OUT / "photoswitch_proposal_audit.csv", index=False)
generated_photoswitch = photo_audit[
    photo_audit["passes_hard_filters"] & photo_audit["pair_valid"] & ~photo_audit["canonical_smiles"].isin(KNOWN_UNION)
].drop_duplicates(["seed", "canonical_smiles"]).copy()
generated_photoswitch["deltaH_kJ_mol"] = np.nan
generated_photoswitch["stored_energy_MJ_kg"] = np.nan
generated_photoswitch["PSS_pred"] = np.nan
generated_photoswitch["quantum_yield_pred"] = np.nan
generated_photoswitch["final_status"] = np.where(
    (generated_photoswitch["pred_delta_lambda_nm"] >= 20.0)
    & generated_photoswitch["pred_E_lambda_nm"].between(300.0, 400.0)
    & generated_photoswitch["pred_half_life_h"].between(4.0, 24.0),
    "PASS_PHOTOSWITCH_PROXY", "FAIL_PHOTOSWITCH_PROXY",
)
photo_columns = [
    "canonical_smiles", "state_A_smiles", "state_B_smiles", "method_id", "seed",
    "pair_valid", "pred_E_lambda_nm", "pred_Z_lambda_nm", "pred_E_uncertainty_nm", "pred_Z_uncertainty_nm",
    "pred_delta_lambda_nm", "pred_uv_lambda_nm", "pred_uv_uncertainty_nm", "pred_half_life_h",
    "pred_half_life_uncertainty_log10_h", "switch_proxy_score", "reward", "sa_score", "qed",
    "molecular_weight", "rotatable_bonds", "psoralen_similarity", "deltaH_kJ_mol",
    "stored_energy_MJ_kg", "PSS_pred", "quantum_yield_pred", "energy_endpoint_available",
    "PSS_available", "quantum_yield_available", "final_status",
]
generated_photoswitch[photo_columns].to_csv(ROOT / "generated_photoswitch.csv", index=False)
if generated_photoswitch.empty:
    raise RuntimeError("No azo E/Z photoswitch candidates passed hard filters")
print(f"generated photoswitch pairs={len(generated_photoswitch):,}; proxy passes={(generated_photoswitch.final_status == 'PASS_PHOTOSWITCH_PROXY').sum():,}")
"""
    ),
    md("## Pair visualization and physical limitations"),
    code(
        r"""
from rdkit.Chem import Draw
top = generated_photoswitch.sort_values("reward", ascending=False).head(16)
display(top[["canonical_smiles", "state_A_smiles", "state_B_smiles", "pred_delta_lambda_nm", "pred_half_life_h", "final_status"]])
display(Draw.MolsToGridImage([Chem.MolFromSmiles(s) for s in top["state_A_smiles"]], legends=["E " + s[:18] for s in top["canonical_smiles"]], molsPerRow=4, useSVG=True))
display(Draw.MolsToGridImage([Chem.MolFromSmiles(s) for s in top["state_B_smiles"]], legends=["Z " + s[:18] for s in top["canonical_smiles"]], molsPerRow=4, useSVG=True))
manifest = {
    "purpose": "pair-aware de novo azo photoswitch candidate generation",
    "scope_warning": "E/Z and lambda/half-life proxies only; PSS, quantum yield, DeltaH, stored energy and cycling require physical validation",
    "run_seeds": list(RUN_SEEDS), "device": str(DEVICE), "M01_sha256": sha256_file(m01_path),
    "base_joint_manifest": str(ROOT / "outputs" / "uv_most_joint" / "run_manifest.json"),
    "generated_rows": int(len(generated_photoswitch)),
    "proxy_pass_rows": int((generated_photoswitch.final_status == "PASS_PHOTOSWITCH_PROXY").sum()),
    "artifacts": {"generated": str(ROOT / "generated_photoswitch.csv"), "audit": str(PHOTOSWITCH_OUT / "photoswitch_proposal_audit.csv")},
}
(PHOTOSWITCH_OUT / "photoswitch_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
"""
    ),
])

nbf.write(nb, NOTEBOOK)
print(f"Wrote {NOTEBOOK}")
