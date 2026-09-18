from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "Unified_Multiclass_Photoswitch_RL.ipynb"


def md(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str):
    return nbf.v4.new_code_cell(text.strip())


cells = [
    md(
        """
# Conditional generator and safety screen for four photoswitch classes

Одна общая SELFIES-GRU policy обучается с четырьмя управляющими токенами:

- `<AZO>` — `N=N` E/Z photoswitch;
- `<STILBENE>` — diaryl-alkene E/Z candidate.
- `<SPIROPYRAN>` — spiropyran/merocyanine pair;
- `<DIARYLETHENE>` — open/closed dithienylethene pair.

У модели единые embedding, GRU, vocabulary, optimizer и checkpoint. Различаются
только химические правила построения state-pair и доступные property labels.
Для классов с изменением связности используются reaction-constrained templates.

Пространство A теперь означает MOST/photoswitch suitability, а пространство B —
экспериментально обученный 3T3 NRU phototoxicity screen. QDB labels являются
целевой переменной; ADME-like descriptors используются только как признаки.
PhotoChem применяется для независимой проверки согласия labels по CAS.
        """
    ),
    code(
        r"""
from __future__ import annotations

import copy
import hashlib
import importlib.metadata as importlib_metadata
import json
import math
import os
import random
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import selfies as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem, DataStructs, RDConfig, RDLogger
from rdkit.Chem import Descriptors, Lipinski, QED, rdFingerprintGenerator, rdMolDescriptors
from rdkit.Chem.MolStandardize import rdMolStandardize
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from mostgen.metrics import generative_quality_metrics
from mostgen.chemistry import build_pair as build_template_pair
from mostgen.config import load_config
from mostgen.phototoxicity import (
    cross_validate_phototoxicity,
    fit_phototoxicity_ensemble,
    load_experimental_phototoxicity,
)
from mostgen.photoswitch_pairs import build_diarylethene_pair

RDLogger.DisableLog("rdApp.*")
ROOT = Path.cwd().resolve()
OUT = ROOT / "outputs" / "uv_most_multiclass"
MODEL_DIR = OUT / "models"
OUT.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
BASE_SEED = 20260917
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FAMILIES = ("azo", "stilbene", "spiropyran", "diarylethene")
FAMILY_TOKENS = {
    "azo": "<AZO>",
    "stilbene": "<STILBENE>",
    "spiropyran": "<SPIROPYRAN>",
    "diarylethene": "<DIARYLETHENE>",
}
MAX_TOKENS = 100
PRIOR_EPOCHS = 6 if DEVICE.type == "cpu" else 8
RL_STEPS = 120
RL_BATCH_PER_FAMILY = 48
FINAL_PROPOSALS = {"azo": 8000, "stilbene": 8000, "spiropyran": 0, "diarylethene": 0}
TEMPLATE_RL_PROPOSALS = 2000
TOP_CANDIDATES = 1000
TOP_PER_FAMILY = TOP_CANDIDATES // len(FAMILIES)
TOXICITY_SCREEN_THRESHOLD = 0.70
assert TOP_CANDIDATES % len(FAMILIES) == 0
assert max(FINAL_PROPOSALS.values()) <= 10_000

def seed_everything(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

seed_everything(BASE_SEED)
print(f"device={DEVICE}; one shared generator; families={FAMILIES}")
"""
    ),
    md("## Data and shared molecular utilities"),
    code(
        r"""
_uncharger = rdMolStandardize.Uncharger()
ALLOWED_ATOMIC_NUMBERS = {1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 35, 53}
FP_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
AZO_QUERY = Chem.MolFromSmarts("[N]=[N]")
STILBENE_QUERY = Chem.MolFromSmarts("[c:1]-[C;H0,H1:2]=[C;H0,H1:3]-[c:4]")
PSORALEN_QUERIES = [Chem.MolFromSmiles(x) for x in (
    "C1=CC(=O)OC2=CC3=C(C=CO3)C=C21",
    "C1=CC2=C(C=CO2)C3=C1C=CC(=O)O3",
)]
REACTIVE_QUERIES = [Chem.MolFromSmarts(x) for x in (
    "[CX3](=[OX1])[F,Cl,Br,I]", "[SX4](=[OX1])(=[OX1])[F,Cl,Br,I]",
    "[NX2]=[CX2]=[OX1]", "[OX2]-[OX2]", "[N+]#N",
)]
PAIR_LOOKUP = {}
if str(Path(RDConfig.RDContribDir) / "SA_Score") not in sys.path:
    sys.path.insert(0, str(Path(RDConfig.RDContribDir) / "SA_Score"))
import sascorer

def standardize_smiles(value):
    try:
        mol = Chem.MolFromSmiles(str(value), sanitize=True)
        if mol is None: return None
        mol = rdMolStandardize.Cleanup(mol)
        mol = rdMolStandardize.FragmentParent(mol)
        mol = _uncharger.uncharge(mol)
        Chem.SanitizeMol(mol)
        if not mol.GetNumHeavyAtoms(): return None
        if any(a.GetAtomicNum() not in ALLOWED_ATOMIC_NUMBERS for a in mol.GetAtoms()): return None
        first = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        second_mol = Chem.MolFromSmiles(first)
        second = Chem.MolToSmiles(second_mol, canonical=True, isomericSmiles=True)
        return second
    except Exception:
        return None

def bit_fingerprint(smiles):
    return FP_GENERATOR.GetFingerprint(Chem.MolFromSmiles(smiles))

def fp_matrix(smiles_values):
    matrix = np.zeros((len(smiles_values), 2048), dtype=np.uint8)
    for i, smiles in enumerate(smiles_values):
        DataStructs.ConvertToNumpyArray(bit_fingerprint(smiles), matrix[i])
    return matrix

def build_ez_pair(smiles, family):
    if family not in {"azo", "stilbene"}: return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return None
    for bond in mol.GetBonds():
        left, right = bond.GetBeginAtom(), bond.GetEndAtom()
        if bond.GetBondType() != Chem.BondType.DOUBLE or bond.IsInRing(): continue
        if family == "azo":
            matched = left.GetSymbol() == right.GetSymbol() == "N"
        else:
            matched = (
                left.GetSymbol() == right.GetSymbol() == "C"
                and any(n.GetIsAromatic() for n in left.GetNeighbors() if n.GetIdx() != right.GetIdx())
                and any(n.GetIsAromatic() for n in right.GetNeighbors() if n.GetIdx() != left.GetIdx())
            )
        if not matched: continue
        left_neighbors = [n.GetIdx() for n in left.GetNeighbors() if n.GetIdx() != right.GetIdx()]
        right_neighbors = [n.GetIdx() for n in right.GetNeighbors() if n.GetIdx() != left.GetIdx()]
        if not left_neighbors or not right_neighbors: continue
        states = []
        for stereo in (Chem.BondStereo.STEREOE, Chem.BondStereo.STEREOZ):
            state = Chem.Mol(mol)
            for other in state.GetBonds(): other.SetBondDir(Chem.BondDir.NONE)
            target = state.GetBondWithIdx(bond.GetIdx())
            target.SetStereo(Chem.BondStereo.STEREONONE)
            target.SetStereoAtoms(left_neighbors[0], right_neighbors[0])
            target.SetStereo(stereo)
            Chem.AssignStereochemistry(state, cleanIt=False, force=True)
            state_smiles = Chem.MolToSmiles(state, canonical=True, isomericSmiles=True)
            reparsed = Chem.MolFromSmiles(state_smiles)
            if reparsed is None: return None
            states.append(Chem.MolToSmiles(reparsed, canonical=True, isomericSmiles=True))
        if states[0] == states[1]: continue
        state_a, state_b = (Chem.MolFromSmiles(value) for value in states)
        connectivity_a = Chem.MolToSmiles(state_a, canonical=True, isomericSmiles=False)
        connectivity_b = Chem.MolToSmiles(state_b, canonical=True, isomericSmiles=False)
        if connectivity_a != connectivity_b: continue
        if rdMolDescriptors.CalcMolFormula(state_a) != rdMolDescriptors.CalcMolFormula(state_b): continue
        return tuple(states)
    return None

def common_filter(smiles, family):
    row = {"input_smiles": smiles, "family": family, "canonical_smiles": None,
           "state_A_smiles": None, "state_B_smiles": None, "pair_valid": False,
           "valid": False, "passes_common_filter": False, "filter_reason": "invalid"}
    canonical = standardize_smiles(smiles)
    if not canonical: return row
    mol = Chem.MolFromSmiles(canonical)
    row.update({"canonical_smiles": canonical, "valid": True})
    if len(Chem.GetMolFrags(mol)) != 1:
        row["filter_reason"] = "multiple_fragments"; return row
    if any(mol.HasSubstructMatch(q) for q in PSORALEN_QUERIES):
        row["filter_reason"] = "psoralen_core"; return row
    if any(mol.HasSubstructMatch(q) for q in REACTIVE_QUERIES):
        row["filter_reason"] = "reactive_group"; return row
    mw = float(Descriptors.MolWt(mol)); sa = float(sascorer.calculateScore(mol)); qed = float(QED.qed(mol))
    rotors = int(Lipinski.NumRotatableBonds(mol))
    row.update({"molecular_weight": mw, "sa_score": sa, "qed": qed, "rotatable_bonds": rotors})
    maximum_mw = 725 if family == "diarylethene" else 650
    if not 120 <= mw <= maximum_mw or sa > 6 or qed < 0.10 or rotors > 14 or abs(Chem.GetFormalCharge(mol)) > 1:
        row["filter_reason"] = "size_sa_qed_charge"; return row
    pair = PAIR_LOOKUP.get((family, canonical)) if family in {"spiropyran", "diarylethene"} else build_ez_pair(canonical, family)
    if pair is None:
        row["filter_reason"] = "family_motif_or_pair"; return row
    row.update({"state_A_smiles": pair[0], "state_B_smiles": pair[1], "pair_valid": True,
                "passes_common_filter": True, "filter_reason": "ok"})
    return row
"""
    ),
    md("## Build a balanced four-family corpus"),
    code(
        r"""
dataset_a = pd.read_csv(ROOT / "outputs" / "uv_most_joint" / "dataset_A_curated.csv")
dataset_b = pd.read_csv(ROOT / "outputs" / "uv_most_joint" / "dataset_B_curated.csv")
azo_smiles = dataset_a.loc[dataset_a.split.eq("train"), "canonical_smiles"].drop_duplicates().tolist()
stilbene_candidates = dataset_b.loc[dataset_b.split.eq("train"), "canonical_smiles"].drop_duplicates().tolist()
stilbene_smiles = []
for smiles in stilbene_candidates:
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None and mol.HasSubstructMatch(STILBENE_QUERY) and build_ez_pair(smiles, "stilbene"):
        stilbene_smiles.append(smiles)
stilbene_smiles = sorted(stilbene_smiles, key=lambda s: hashlib.sha256(s.encode()).hexdigest())[:1800]

# Reaction-constrained libraries guarantee chemically paired states for the
# connectivity-changing classes.  Only a deterministic subset enters prior
# training; the held-out combinations form novel enumerated proposals.
template_config = load_config(ROOT / "config" / "default.json", mode="smoke")
template_rows = []
for first in template_config["synthons"]:
    for second in template_config["synthons"]:
        try:
            pair = build_template_pair(template_config, "spiropyran", first["id"], second["id"])
            template_rows.append({"family":"spiropyran", "smiles":pair.smiles, "state_B":pair.charged_smiles})
        except Exception:
            pass
        try:
            pair = build_diarylethene_pair(first["smiles"], second["smiles"])
            template_rows.append({"family":"diarylethene", "smiles":pair.state_a_smiles, "state_B":pair.state_b_smiles})
        except Exception:
            pass
template_unique = {}
for row in template_rows:
    canonical = standardize_smiles(row["smiles"])
    state_b = standardize_smiles(row["state_B"])
    if canonical and state_b:
        template_unique.setdefault((row["family"], canonical), (canonical, state_b))
PAIR_LOOKUP.update({key:value for key,value in template_unique.items()})

template_training = {}; TEMPLATE_HOLDOUT = {}
for family in ("spiropyran", "diarylethene"):
    members = [value[0] for key,value in template_unique.items() if key[0] == family]
    members = [smiles for smiles in members if common_filter(smiles, family)["passes_common_filter"]]
    members = sorted(set(members), key=lambda s: hashlib.sha256((family+"|"+s).encode()).hexdigest())
    template_training[family] = members[:180]
    TEMPLATE_HOLDOUT[family] = members[180:]
    if len(TEMPLATE_HOLDOUT[family]) < TOP_PER_FAMILY:
        raise RuntimeError(f"Insufficient held-out {family} templates: {len(TEMPLATE_HOLDOUT[family])}")

target_corpus_size = max(len(stilbene_smiles), 1)
def balance(values):
    return (values * math.ceil(target_corpus_size / max(len(values), 1)))[:target_corpus_size]

family_corpora = {
    "azo": balance(azo_smiles),
    "stilbene": balance(stilbene_smiles),
    "spiropyran": balance(template_training["spiropyran"]),
    "diarylethene": balance(template_training["diarylethene"]),
}

def encoded(smiles, family):
    try: tokens = list(sf.split_selfies(sf.encoder(smiles)))
    except Exception: return None
    return (family, smiles, tokens) if 1 <= len(tokens) <= MAX_TOKENS else None

records = []
for family, corpus in family_corpora.items():
    records += [record for smiles in corpus if (record := encoded(smiles, family))]
records.sort(key=lambda r: hashlib.sha256((r[0] + "|" + r[1]).encode()).hexdigest())
training_smiles = {smiles for _, smiles, _ in records}
print("conditional corpus:", {family:len(corpus) for family,corpus in family_corpora.items()}, "total=",len(records))
print("held-out template proposals:", {family:len(values) for family,values in TEMPLATE_HOLDOUT.items()})
"""
    ),
    md("## One conditional SELFIES-GRU"),
    code(
        r"""
SPECIAL = ["<pad>", "<bos>", "<eos>", *FAMILY_TOKENS.values()]
vocabulary = SPECIAL + sorted({token for _, _, tokens in records for token in tokens})
token_to_id = {token: i for i, token in enumerate(vocabulary)}
id_to_token = {i: token for token, i in token_to_id.items()}
PAD_ID, BOS_ID, EOS_ID = (token_to_id[x] for x in SPECIAL[:3])
FAMILY_IDS = {family: token_to_id[token] for family, token in FAMILY_TOKENS.items()}

class ConditionalDataset(Dataset):
    def __init__(self, source):
        self.sequences = [torch.tensor([BOS_ID, FAMILY_IDS[f]] + [token_to_id[t] for t in tokens] + [EOS_ID]) for f, _, tokens in source]
    def __len__(self): return len(self.sequences)
    def __getitem__(self, i): return self.sequences[i]

def collate(items): return pad_sequence(items, batch_first=True, padding_value=PAD_ID)

class ConditionalGRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(len(vocabulary), 160, padding_idx=PAD_ID)
        self.gru = nn.GRU(160, 224, num_layers=2, batch_first=True, dropout=0.1)
        self.output = nn.Linear(224, len(vocabulary))
    def forward(self, ids, hidden=None):
        states, hidden = self.gru(self.embedding(ids), hidden)
        return self.output(states), hidden

loader = DataLoader(ConditionalDataset(records), batch_size=128, shuffle=True,
                    generator=torch.Generator().manual_seed(BASE_SEED), collate_fn=collate)
prior = ConditionalGRU().to(DEVICE)
optimizer = torch.optim.AdamW(prior.parameters(), lr=1e-3, weight_decay=1e-5)
prior_history = []
for epoch in range(1, PRIOR_EPOCHS + 1):
    total, count = 0.0, 0
    prior.train()
    for seq in loader:
        seq = seq.to(DEVICE); logits, _ = prior(seq[:, :-1]); targets = seq[:, 1:]
        loss_sum = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), ignore_index=PAD_ID, reduction="sum")
        n = int(targets.ne(PAD_ID).sum()); loss = loss_sum / max(n, 1)
        optimizer.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(prior.parameters(), 2.0); optimizer.step()
        total += float(loss_sum.detach()); count += n
    prior_history.append({"epoch": epoch, "cross_entropy": total/max(count,1)})
    print(epoch, prior_history[-1])
pd.DataFrame(prior_history).to_csv(OUT / "prior_history.csv", index=False)
"""
    ),
    md("## Shared conditioning, reward models and one RL optimizer"),
    code(
        r"""
reward_uv = joblib.load(ROOT / "outputs" / "uv_most_joint" / "models" / "reward_B_rf.joblib")
reward_half = joblib.load(ROOT / "outputs" / "uv_most_joint" / "models" / "reward_A_rf.joblib")
evaluator_uv = joblib.load(ROOT / "outputs" / "uv_most_joint" / "models" / "evaluator_B_extra_trees.joblib")
evaluator_half = joblib.load(ROOT / "outputs" / "uv_most_joint" / "models" / "evaluator_A_extra_trees.joblib")

toxicity_data = load_experimental_phototoxicity(
    ROOT / "datasets" / "QDB_2011TIV324_phototoxicity_normalized.xlsx",
    ROOT / "datasets" / "PhotoChem_phototoxicity_251_verified.xlsx",
)
toxicity_cv = cross_validate_phototoxicity(toxicity_data, seed=BASE_SEED)
reward_toxicity = fit_phototoxicity_ensemble(
    toxicity_data, model_kind="random_forest", seed=BASE_SEED+701,
    threshold=TOXICITY_SCREEN_THRESHOLD,
)
evaluator_toxicity = fit_phototoxicity_ensemble(
    toxicity_data, model_kind="extra_trees", seed=BASE_SEED+801,
    threshold=TOXICITY_SCREEN_THRESHOLD,
)
toxicity_data.to_csv(OUT / "phototoxicity_training.csv", index=False)
pd.DataFrame([toxicity_cv]).to_csv(OUT / "phototoxicity_model_metrics.csv", index=False)
joblib.dump(reward_toxicity, MODEL_DIR / "reward_phototoxicity_rf.joblib")
joblib.dump(evaluator_toxicity, MODEL_DIR / "evaluator_phototoxicity_extra_trees.joblib")
print("phototoxicity diagnostics:", toxicity_cv)
print("PhotoChem CAS-linked calls:", int(toxicity_data.photochem_label.notna().sum()),
      "agreement:", float(toxicity_data.photochem_agrees.dropna().mean()))

m01 = pd.read_csv(ROOT / "data" / "raw" / "M01_photoswitch" / "dataset" / "photoswitches.csv")
m01["canonical_smiles"] = m01.SMILES.map(standardize_smiles)
m01["E"] = pd.to_numeric(m01["E isomer pi-pi* wavelength in nm"], errors="coerce")
m01["Z"] = pd.to_numeric(m01["Z isomer pi-pi* wavelength in nm"], errors="coerce")
from sklearn.ensemble import RandomForestRegressor
def fit_transition(target, seed):
    frame = m01.dropna(subset=["canonical_smiles", target]).drop_duplicates("canonical_smiles")
    x = fp_matrix(frame.canonical_smiles.tolist()); y = frame[target].to_numpy(float); models=[]
    for member in range(3):
        model=RandomForestRegressor(n_estimators=100,min_samples_leaf=2,max_features="sqrt",bootstrap=True,n_jobs=-1,random_state=seed+1009*member)
        model.fit(x,y);models.append(model)
    return models
azo_E_models=fit_transition("E",BASE_SEED+501);azo_Z_models=fit_transition("Z",BASE_SEED+601)

def predict_models(bundle, smiles_values):
    x=fp_matrix(smiles_values);values=np.stack([m.predict(x) for m in bundle["models"]],axis=0)
    mean=values.mean(0);std=np.maximum(values.std(0),float(bundle["conformal_q90"])/1.645)
    return mean,std

def predict_list(models, smiles_values):
    x=fp_matrix(smiles_values);values=np.stack([m.predict(x) for m in models],axis=0)
    return values.mean(0),values.std(0)

def sigmoid(x): return 1/(1+np.exp(-np.clip(x,-50,50)))
def interval(mean,unc,low,high,soft): return sigmoid((mean-unc-low)/soft)*sigmoid((high-mean-unc)/soft)

@torch.no_grad()
def sample_conditioned(model, family, count, temperature=0.92):
    model.eval(); fam=FAMILY_IDS[family]
    bos=torch.full((count,1),BOS_ID,dtype=torch.long,device=DEVICE);_,hidden=model(bos)
    current=torch.full((count,1),fam,dtype=torch.long,device=DEVICE);finished=torch.zeros(count,dtype=torch.bool,device=DEVICE)
    sequences=[[BOS_ID,fam] for _ in range(count)]
    blocked=[PAD_ID,BOS_ID,*FAMILY_IDS.values()]
    for _ in range(MAX_TOKENS+1):
        logits,hidden=model(current,hidden);next_logits=logits[:,-1,:]/temperature
        for token_id in blocked: next_logits[:,token_id]=-torch.inf
        nxt=torch.multinomial(torch.softmax(next_logits,-1),1).squeeze(1);nxt=torch.where(finished,torch.full_like(nxt,EOS_ID),nxt)
        for i,t in enumerate(nxt.tolist()):
            if not finished[i]:sequences[i].append(t)
        finished|=nxt.eq(EOS_ID);current=nxt[:,None]
        if bool(finished.all()):break
    smiles=[]
    for seq in sequences:
        content=[]
        for token in seq[2:]:
            if token==EOS_ID:break
            if token not in blocked:content.append(id_to_token[token])
        try:smiles.append(sf.decoder("".join(content)) if content else "")
        except Exception:smiles.append("")
    return smiles,sequences

def policy_terms(agent,frozen,sequences):
    tensors=[torch.tensor(s) for s in sequences];padded=pad_sequence(tensors,batch_first=True,padding_value=PAD_ID).to(DEVICE)
    inputs,targets=padded[:,:-1],padded[:,1:];mask=targets.ne(PAD_ID);mask[:,0]=False;lengths=mask.sum(1).clamp_min(1)
    al,_=agent(inputs)
    with torch.no_grad():pl,_=frozen(inputs)
    for t in [PAD_ID,BOS_ID,*FAMILY_IDS.values()]:al[...,t]=-1e9;pl[...,t]=-1e9
    alp=F.log_softmax(al,-1);plp=F.log_softmax(pl,-1);ap=alp.exp();selected=alp.gather(-1,targets[...,None]).squeeze(-1)
    return (selected*mask).sum(1)/lengths,((ap*(alp-plp)).sum(-1)*mask).sum(1)/lengths,(-(ap*alp).sum(-1)*mask).sum(1)/lengths

def score_batch(smiles_values,family):
    rows=[common_filter(s,family) for s in smiles_values];eligible=[i for i,r in enumerate(rows) if r["passes_common_filter"] and r["canonical_smiles"] not in training_smiles]
    if eligible:
        smiles=[rows[i]["canonical_smiles"] for i in eligible];uv,uv_unc=predict_models(reward_uv,smiles)
        toxicity=reward_toxicity.predict(smiles).reset_index(drop=True)
        if family=="azo":
            half,half_unc=predict_models(reward_half,smiles);em,_=predict_list(azo_E_models,smiles);zm,_=predict_list(azo_Z_models,smiles)
        for pos,i in enumerate(eligible):
            row=rows[i];uv_score=float(interval(uv[pos],uv_unc[pos],300,400,14));sa_score=math.exp(-.3*max(0,row["sa_score"]-3))
            tox=toxicity.iloc[pos]
            domain_score=.35+.65*float(np.clip((tox.toxicity_similarity-.08)/.35,0,1))
            components=[uv_score,math.exp(-float(uv_unc[pos])/70),sa_score,
                        max(.02,1-float(tox.phototoxic_probability)),
                        math.exp(-3*float(tox.phototoxic_uncertainty)),domain_score]
            if not bool(tox.toxicity_pass): components.append(.05)
            row.update({"pred_uv_lambda_nm":float(uv[pos]),"pred_uv_uncertainty_nm":float(uv_unc[pos]),
                        "reward_phototoxic_probability":float(tox.phototoxic_probability),
                        "reward_phototoxic_uncertainty":float(tox.phototoxic_uncertainty),
                        "reward_toxicity_similarity":float(tox.toxicity_similarity),
                        "reward_toxicity_pass":bool(tox.toxicity_pass)})
            if family=="azo":
                delta=abs(float(em[pos]-zm[pos]));half_score=float(interval(half[pos],half_unc[pos],math.log10(4),math.log10(24),.18))
                components += [float(sigmoid((delta-20)/8)),half_score,math.exp(-float(half_unc[pos])/1.2)]
                row.update({"pred_half_life_h":float(10**half[pos]),"pred_half_uncertainty_log10_h":float(half_unc[pos]),"pred_delta_lambda_nm":delta})
            row["reward"]=float(np.prod(np.clip(components,1e-12,1))**(1/len(components)))
    for row in rows:row.setdefault("reward",0.0)
    return pd.DataFrame(rows)
"""
    ),
    code(
        r"""
for p in prior.parameters():p.requires_grad_(False)
prior.eval();agent=copy.deepcopy(prior).to(DEVICE)
for p in agent.parameters():p.requires_grad_(True)
optimizer=torch.optim.Adam(agent.parameters(),lr=2e-4);history=[];archive=set()
for step in range(1,RL_STEPS+1):
    all_rewards=[];all_advantages=[];all_logp=[];all_kl=[];all_entropy=[];family_stats={}
    for family in FAMILIES:
        smiles,sequences=sample_conditioned(agent,family,RL_BATCH_PER_FAMILY);scored=score_batch(smiles,family);rewards_np=scored.reward.to_numpy(np.float32)
        for i,canonical in enumerate(scored.canonical_smiles):
            key=(family,canonical)
            if canonical and key in archive:rewards_np[i]*=.2
            if canonical and rewards_np[i]>0:archive.add(key)
        logp,kl,entropy=policy_terms(agent,prior,sequences);reward_tensor=torch.tensor(rewards_np,device=DEVICE)
        advantage=(reward_tensor-reward_tensor.mean())/(reward_tensor.std(unbiased=False)+1e-6)
        all_rewards.append(reward_tensor);all_advantages.append(advantage);all_logp.append(logp);all_kl.append(kl);all_entropy.append(entropy)
        family_stats[family]=float(np.mean(rewards_np))
    rewards=torch.cat(all_rewards);advantage=torch.cat(all_advantages);logp=torch.cat(all_logp);kl=torch.cat(all_kl);entropy=torch.cat(all_entropy)
    loss=-(advantage.detach()*logp).mean()+.035*kl.mean()-.002*entropy.mean()
    optimizer.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(agent.parameters(),1);optimizer.step()
    history.append({"step":step,**{f"{family}_reward":family_stats[family] for family in FAMILIES},
                    "mean_reward":float(rewards.mean()),"archive":len(archive)})
    if step==1 or step%20==0:print(history[-1])
pd.DataFrame(history).to_csv(OUT/"rl_history.csv",index=False)
torch.save({"state_dict":{k:v.detach().cpu() for k,v in agent.state_dict().items()},"vocabulary":vocabulary,"family_tokens":FAMILY_TOKENS,"seed":BASE_SEED},MODEL_DIR/"unified_conditional_agent.pt")
"""
    ),
    md("## Generate all four classes (maximum 10,000 proposals per class)"),
    code(
        r"""
parts=[]
for family_index,family in enumerate(FAMILIES):
    seed_everything(BASE_SEED+1000+family_index);proposals=[];sources=[]
    if family in TEMPLATE_HOLDOUT:
        proposals.extend(TEMPLATE_HOLDOUT[family]);sources.extend(["heldout_reaction_template"]*len(TEMPLATE_HOLDOUT[family]))
        count=min(TEMPLATE_RL_PROPOSALS,10_000-len(proposals))
        smiles,_=sample_conditioned(agent,family,count);proposals.extend(smiles);sources.extend(["unified_conditional_rl"]*count)
    else:
        remaining=FINAL_PROPOSALS[family]
        while remaining:
            n=min(256,remaining);smiles,_=sample_conditioned(agent,family,n);proposals.extend(smiles);remaining-=n
        sources=["unified_conditional_rl"]*len(proposals)
    assert len(proposals)<=10_000
    scored=score_batch(proposals,family);scored["method_id"]=sources;parts.append(scored)
audit=pd.concat(parts,ignore_index=True)

# Evaluate every unique valid proposal so JSR has exactly N_generated as its
# denominator. Invalid/unevaluated proposals are mapped to target failures.
evaluated=(audit[audit.valid & audit.canonical_smiles.ne("")]
           .drop_duplicates(["family","canonical_smiles"]).copy())
for family in FAMILIES:
    idx=evaluated.family.eq(family);smiles=evaluated.loc[idx,"canonical_smiles"].tolist()
    if not smiles:continue
    uv,uv_unc=predict_models(evaluator_uv,smiles)
    evaluated.loc[idx,"eval_uv_lambda_nm"]=uv;evaluated.loc[idx,"eval_uv_uncertainty_nm"]=uv_unc
    toxicity=evaluator_toxicity.predict(smiles);toxicity.index=evaluated.index[idx]
    for column in toxicity.columns:evaluated.loc[idx,column]=toxicity[column]
    if family=="azo":
        half,half_unc=predict_models(evaluator_half,smiles)
        em,_=predict_list(azo_E_models,smiles);zm,_=predict_list(azo_Z_models,smiles)
        evaluated.loc[idx,"eval_half_life_h"]=10**half
        evaluated.loc[idx,"eval_half_uncertainty_log10_h"]=half_unc
        evaluated.loc[idx,"pred_delta_lambda_nm"]=np.abs(em-zm)

evaluated["photoswitch_space_pass"]=evaluated.eval_uv_lambda_nm.between(300,400) & evaluated.pair_valid
azo_idx=evaluated.family.eq("azo")
stilbene_idx=evaluated.family.eq("stilbene")
spiropyran_idx=evaluated.family.eq("spiropyran")
diarylethene_idx=evaluated.family.eq("diarylethene")
evaluated.loc[azo_idx,"photoswitch_space_pass"]=(evaluated.loc[azo_idx,"photoswitch_space_pass"] &
    evaluated.loc[azo_idx,"eval_half_life_h"].between(4,24) & evaluated.loc[azo_idx,"pred_delta_lambda_nm"].ge(20))
evaluated["toxicity_space_pass"]=evaluated.toxicity_pass.fillna(False).astype(bool)
evaluated["target_A_pass"]=evaluated.photoswitch_space_pass
evaluated["target_B_pass"]=evaluated.toxicity_space_pass
evaluated["joint_success"]=evaluated.target_A_pass & evaluated.target_B_pass

def evaluator_selection_score(row):
    components=[
        interval(row.eval_uv_lambda_nm,row.eval_uv_uncertainty_nm,300,400,14),
        math.exp(-float(row.eval_uv_uncertainty_nm)/70),
        math.exp(-.3*max(0,float(row.sa_score)-3)),
        max(.02,1-float(row.phototoxic_probability)),
        math.exp(-3*float(row.phototoxic_uncertainty)),
        .35+.65*float(np.clip((row.toxicity_similarity-.08)/.35,0,1)),
    ]
    if row.family=="azo":
        components += [
            interval(math.log10(row.eval_half_life_h),row.eval_half_uncertainty_log10_h,math.log10(4),math.log10(24),.18),
            math.exp(-float(row.eval_half_uncertainty_log10_h)/1.2),
            float(sigmoid((row.pred_delta_lambda_nm-20)/8)),
        ]
    return float(np.prod(np.clip(components,1e-12,1))**(1/len(components)))

evaluated["selection_score"]=evaluated.apply(evaluator_selection_score,axis=1)
evaluated["final_status"]="FAIL_PHOTOSWITCH_OR_TOXICITY_SCREEN"
evaluated.loc[azo_idx & evaluated.joint_success,"final_status"]="PASS_AZO_AND_TOXICITY_SCREEN"
evaluated.loc[stilbene_idx & evaluated.joint_success,"final_status"]="PASS_STILBENE_AND_TOXICITY_SCREEN"
evaluated.loc[spiropyran_idx & evaluated.joint_success,"final_status"]="PASS_SPIROPYRAN_AND_TOXICITY_SCREEN"
evaluated.loc[diarylethene_idx & evaluated.joint_success,"final_status"]="PASS_DIARYLETHENE_AND_TOXICITY_SCREEN"
evaluated["PSS_pred"]=np.nan;evaluated["quantum_yield_pred"]=np.nan;evaluated["deltaH_kJ_mol"]=np.nan;evaluated["stored_energy_MJ_kg"]=np.nan

metric_flags=evaluated[["family","canonical_smiles","target_A_pass","target_B_pass","joint_success",
                        "photoswitch_space_pass","toxicity_space_pass","phototoxic_probability",
                        "phototoxic_uncertainty","phototoxic_upper_confidence","toxicity_similarity",
                        "toxicity_in_domain","known_phototoxic_match","toxicity_pass"]]
audit=audit.drop(columns=["target_A_pass","target_B_pass","joint_success"],errors="ignore").merge(
    metric_flags,on=["family","canonical_smiles"],how="left")
for column in ["target_A_pass","target_B_pass","joint_success"]:
    audit[column]=audit[column].fillna(False).astype(bool)
audit.to_csv(OUT/"proposal_audit.csv",index=False)

candidate_pool=evaluated[evaluated.passes_common_filter & evaluated.toxicity_space_pass &
                         ~evaluated.canonical_smiles.isin(training_smiles)].copy()
candidate_pool=candidate_pool.sort_values(
    ["family","joint_success","selection_score","reward","canonical_smiles"],
    ascending=[True,False,False,False,True],kind="mergesort")
candidate_pool["rank_within_family"]=candidate_pool.groupby("family").cumcount()+1

# A balanced shortlist avoids comparing scores backed by different endpoint sets.
selected_parts=[];selected_smiles=set()
for family in FAMILIES:
    family_pool=candidate_pool[candidate_pool.family.eq(family) & ~candidate_pool.canonical_smiles.isin(selected_smiles)]
    chosen=family_pool.head(TOP_PER_FAMILY).copy()
    assert len(chosen)==TOP_PER_FAMILY,f"not enough {family} candidates"
    selected_smiles.update(chosen.canonical_smiles)
    selected_parts.append(chosen)
generated=pd.concat(selected_parts,ignore_index=True)
generated["candidate_id"]=[f"MC-{row.family.upper()}-{rank:04d}" for rank,row in enumerate(generated.itertuples(),1)]
generated.insert(0,"selection_rank",np.arange(1,len(generated)+1))
generated.to_csv(ROOT/"generated_multiclass_photoswitch.csv",index=False)
candidate_pool.to_csv(OUT/"candidate_pool.csv",index=False)

metric_rows=[]
for scope,frame in [("all_proposals",audit),("top_1000",generated)]:
    for family in ("all",*FAMILIES):
        subset=frame if family=="all" else frame[frame.family.eq(family)]
        values=generative_quality_metrics(
            subset.to_dict(orient="records"),training_smiles,2048,
            smiles_key="canonical_smiles",joint_a_key="target_A_pass",joint_b_key="target_B_pass")
        metric_rows.append({"scope":scope,"family":family,**values})
metrics=pd.DataFrame(metric_rows)
metrics.to_csv(OUT/"generation_metrics.csv",index=False)

summary=(candidate_pool.groupby(["family","final_status"])
         .agg(rows=("canonical_smiles","size"),selected_top_1000=("canonical_smiles",lambda values: values.isin(selected_smiles).sum()))
         .reset_index())
summary.to_csv(OUT/"summary.csv",index=False);display(summary);display(metrics)
assert len(generated)==TOP_CANDIDATES
assert generated.canonical_smiles.nunique()==TOP_CANDIDATES
assert generated.groupby("family").size().eq(TOP_PER_FAMILY).all()
assert generated.toxicity_space_pass.all()
for row in generated.itertuples():
    state_a, state_b = Chem.MolFromSmiles(row.state_A_smiles), Chem.MolFromSmiles(row.state_B_smiles)
    assert state_a is not None and state_b is not None
    assert Chem.MolToSmiles(state_a, canonical=True, isomericSmiles=True) != Chem.MolToSmiles(state_b, canonical=True, isomericSmiles=True)
    assert rdMolDescriptors.CalcMolFormula(state_a) == rdMolDescriptors.CalcMolFormula(state_b)
    if row.family in {"azo","stilbene"}:
        assert Chem.MolToSmiles(state_a, canonical=True, isomericSmiles=False) == Chem.MolToSmiles(state_b, canonical=True, isomericSmiles=False)
manifest = {
    "purpose": "four-class photoswitch generation with an experimental phototoxicity screen",
    "generator": {"architecture": "ConditionalGRU plus reaction-constrained state-pair templates", "single_checkpoint": str(MODEL_DIR/"unified_conditional_agent.pt"), "family_tokens": FAMILY_TOKENS},
    "sources": {"M01": "The Photoswitch Dataset", "UVVisML": "pinned cached split files",
                "QDB_2011TIV324": "53 structured OECD 432 / 3T3 NRU labels",
                "PhotoChem_251": "heterogeneous external label agreement audit joined by CAS",
                "generator_training_structures": len(training_smiles)},
    "families": summary.to_dict(orient="records"),
    "proposal_cap_per_family": 10000,
    "selection": {"total": TOP_CANDIDATES, "per_family": TOP_PER_FAMILY, "primary": "photoswitch AND toxicity-space success", "secondary": "evaluator/proxy selection_score"},
    "phototoxicity": {"training_rows": len(toxicity_data), "screening_threshold_upper_confidence": TOXICITY_SCREEN_THRESHOLD,
                       "model_metrics": toxicity_cv, "applicability_is_reported_not_assumed": True},
    "metrics": {
        "path": str(OUT/"generation_metrics.csv"),
        "validity": "N_valid / N_generated",
        "uniqueness": "N_unique_valid / N_valid",
        "novelty": "N_unique_valid_not_in_training / N_unique_valid",
        "joint_success_rate": "mean(target_A_pass AND target_B_pass) over N_generated",
        "diversity": "1 - mean pairwise Morgan(radius=2,2048-bit) Tanimoto over unique valid structures",
        "target_A": "MOST/photoswitch space: valid paired states and UV proxy in [300,400] nm; azo additionally requires half-life [4,24] h and delta-lambda >=20 nm",
        "target_B": "phototoxicity space: independent ExtraTrees ensemble upper-confidence probability < 0.70 and no exact known-positive match",
    },
    "scope_warning": "toxicity_pass is an early in-silico screen, not proof of non-phototoxicity; the 53-compound model is often out-of-domain for new photoswitch classes",
    "missing_physical_endpoints": ["PSS", "quantum_yield", "deltaH_kJ_mol", "stored_energy_MJ_kg"],
}
(OUT/"run_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
print(f"selected={len(generated):,}; pool={len(candidate_pool):,}; one checkpoint={MODEL_DIR/'unified_conditional_agent.pt'}")
"""
    ),
    md(
        """
## Interpretation

Все четыре класса входят в одну conditional policy. Для spiropyran и
diarylethene state pairs дополнительно ограничены реакционными шаблонами, потому
что их нельзя корректно получить простой сменой E/Z stereo. `target_A_pass`
описывает photoswitch/MOST-пространство, `target_B_pass` — экспериментально
обученный phototoxicity screen. Последний является приоритизацией для дальнейшей
OECD 432 проверки, а не утверждением о безопасности.
        """
    ),
]

nb = nbf.v4.new_notebook(cells=cells)
nb.metadata.kernelspec = {"display_name":"Python 3 (uvgen)","language":"python","name":"python3"}
nb.metadata.language_info = {"name":"python","version":"3.12"}
nbf.write(nb, NOTEBOOK)
print(f"Wrote {NOTEBOOK}")
