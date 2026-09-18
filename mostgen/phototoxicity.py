from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, Lipinski, rdFingerprintGenerator, rdMolDescriptors
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from .chemistry import ChemistryError, standardize_smiles


FP_BITS = 2048
_FP_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=FP_BITS)
_DESCRIPTOR_NAMES = (
    "mol_wt",
    "logp",
    "tpsa",
    "hbd",
    "hba",
    "rotatable_bonds",
    "aromatic_fraction",
    "fraction_csp3",
    "formal_charge",
)


def _clean_cas(value: object) -> str | None:
    if pd.isna(value):
        return None
    cas = str(value).strip()
    return cas if cas and cas.lower() != "nan" else None


def _photochem_call(values: pd.Series) -> float:
    """Conservative call from heterogeneous PhotoChem endpoint text.

    Mixed calls such as ``PT/NPT`` are deliberately treated as unresolved.
    The standardized OECD 432 QDB label remains the training target.
    """
    positive = negative = False
    for raw in values.dropna().astype(str):
        token = raw.upper().replace(" ", "")
        if not token or token == "-" or "/" in token:
            continue
        if "NPT" in token:
            negative = True
        elif "PT" in token:
            positive = True
    if positive:
        return 1.0
    if negative:
        return 0.0
    return np.nan


def load_experimental_phototoxicity(
    qdb_excel: str | Path,
    photochem_excel: str | Path | None = None,
) -> pd.DataFrame:
    """Load molecular OECD 432 labels and attach PhotoChem agreement metadata.

    QDB supplies a structure and a single standardized 3T3 NRU label for all
    training rows.  PhotoChem contains useful human, animal and in-vitro calls
    but no structures, so it is joined by CAS only and used as an external
    evidence audit rather than silently mixing heterogeneous endpoints.
    """
    qdb = pd.read_excel(qdb_excel, sheet_name="Dataset")
    required = {"Compound ID", "CAS", "InChI", "Phototoxicity (binary)"}
    missing = required - set(qdb.columns)
    if missing:
        raise ValueError(f"QDB workbook is missing columns: {sorted(missing)}")

    rows: list[dict[str, object]] = []
    for _, row in qdb.iterrows():
        record = row.to_dict()
        inchi = record.get("InChI")
        mol = Chem.MolFromInchi(str(inchi)) if pd.notna(inchi) else None
        if mol is None:
            continue
        try:
            smiles = standardize_smiles(Chem.MolToSmiles(mol, isomericSmiles=True))
        except ChemistryError:
            continue
        raw_label = str(record["Phototoxicity (binary)"]).strip().lower()
        if raw_label in {"yes", "p", "positive", "1", "true"}:
            label = 1
        elif raw_label in {"no", "n", "negative", "0", "false"}:
            label = 0
        else:
            continue
        rows.append(
            {
                "compound_id": int(record["Compound ID"]),
                "name": record.get("Name"),
                "cas": _clean_cas(record.get("CAS")),
                "smiles": smiles,
                "phototoxic": label,
                "endpoint": record.get("Endpoint"),
            }
        )
    frame = pd.DataFrame(rows).drop_duplicates("smiles").reset_index(drop=True)

    frame["photochem_label"] = np.nan
    frame["photochem_evidence_count"] = 0
    if photochem_excel is not None and Path(photochem_excel).exists():
        photo = pd.read_excel(photochem_excel, sheet_name="PhotoChem 251")
        endpoint_columns = [
            "Human result",
            "Animal result",
            "TG 432 (3T3)",
            "TG 495 (ROS)",
            "TG 498 (RhE)",
        ]
        available = [column for column in endpoint_columns if column in photo]
        calls: dict[str, tuple[float, int]] = {}
        for cas, group in photo.assign(_cas=photo["CAS No."].map(_clean_cas)).dropna(subset=["_cas"]).groupby("_cas"):
            values = group[available].stack()
            calls[str(cas)] = (_photochem_call(values), int(values.notna().sum()))
        frame["photochem_label"] = frame.cas.map(lambda value: calls.get(str(value), (np.nan, 0))[0])
        frame["photochem_evidence_count"] = frame.cas.map(lambda value: calls.get(str(value), (np.nan, 0))[1])
    frame["photochem_agrees"] = np.where(
        frame.photochem_label.notna(),
        frame.photochem_label.astype("Int64") == frame.phototoxic,
        pd.NA,
    )
    return frame


def molecular_features(smiles_values: list[str]) -> np.ndarray:
    matrix = np.zeros((len(smiles_values), FP_BITS + len(_DESCRIPTOR_NAMES)), dtype=np.float32)
    for index, smiles in enumerate(smiles_values):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES at index {index}")
        DataStructs.ConvertToNumpyArray(_FP_GENERATOR.GetFingerprint(mol), matrix[index, :FP_BITS])
        heavy = max(1, mol.GetNumHeavyAtoms())
        matrix[index, FP_BITS:] = (
            Descriptors.MolWt(mol),
            Descriptors.MolLogP(mol),
            rdMolDescriptors.CalcTPSA(mol),
            Lipinski.NumHDonors(mol),
            Lipinski.NumHAcceptors(mol),
            Lipinski.NumRotatableBonds(mol),
            sum(atom.GetIsAromatic() for atom in mol.GetAtoms()) / heavy,
            rdMolDescriptors.CalcFractionCSP3(mol),
            Chem.GetFormalCharge(mol),
        )
    return matrix


@dataclass
class PhototoxicityEnsemble:
    models: list[RandomForestClassifier | ExtraTreesClassifier]
    reference_smiles: tuple[str, ...]
    positive_smiles: frozenset[str]
    model_kind: str
    threshold: float = 0.50
    applicability_similarity: float = 0.15

    def predict(self, smiles_values: list[str]) -> pd.DataFrame:
        if not smiles_values:
            return pd.DataFrame(
                columns=[
                    "phototoxic_probability",
                    "phototoxic_uncertainty",
                    "phototoxic_upper_confidence",
                    "toxicity_similarity",
                    "toxicity_in_domain",
                    "known_phototoxic_match",
                    "toxicity_pass",
                ]
            )
        features = molecular_features(smiles_values)
        probabilities = np.stack([model.predict_proba(features)[:, 1] for model in self.models])
        mean = probabilities.mean(axis=0)
        uncertainty = probabilities.std(axis=0)
        upper = np.clip(mean + 1.645 * uncertainty, 0.0, 1.0)
        reference_fps = [_FP_GENERATOR.GetFingerprint(Chem.MolFromSmiles(value)) for value in self.reference_smiles]
        similarities = []
        exact = []
        for smiles in smiles_values:
            fp = _FP_GENERATOR.GetFingerprint(Chem.MolFromSmiles(smiles))
            similarities.append(max(DataStructs.BulkTanimotoSimilarity(fp, reference_fps)))
            exact.append(smiles in self.positive_smiles)
        similarities_array = np.asarray(similarities)
        exact_array = np.asarray(exact, dtype=bool)
        # Applicability is reported separately.  It is not a declaration of
        # safety: candidates outside the domain retain an explicit warning and
        # receive a ranking penalty, while exact positives always fail closed.
        passes = (upper < self.threshold) & ~exact_array
        return pd.DataFrame(
            {
                "phototoxic_probability": mean,
                "phototoxic_uncertainty": uncertainty,
                "phototoxic_upper_confidence": upper,
                "toxicity_similarity": similarities_array,
                "toxicity_in_domain": similarities_array >= self.applicability_similarity,
                "known_phototoxic_match": exact_array,
                "toxicity_pass": passes,
            }
        )


def fit_phototoxicity_ensemble(
    frame: pd.DataFrame,
    *,
    model_kind: Literal["random_forest", "extra_trees"] = "random_forest",
    seed: int = 1701,
    members: int = 5,
    trees_per_member: int = 160,
    threshold: float = 0.70,
) -> PhototoxicityEnsemble:
    smiles = frame.smiles.astype(str).tolist()
    labels = frame.phototoxic.to_numpy(dtype=int)
    if len(np.unique(labels)) != 2:
        raise ValueError("Phototoxicity training data must contain both classes")
    features = molecular_features(smiles)
    models: list[RandomForestClassifier | ExtraTreesClassifier] = []
    for member in range(members):
        common = dict(
            n_estimators=trees_per_member,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced",
            n_jobs=-1,
            random_state=seed + 1009 * member,
        )
        if model_kind == "random_forest":
            model = RandomForestClassifier(bootstrap=True, **common)
        elif model_kind == "extra_trees":
            model = ExtraTreesClassifier(bootstrap=False, **common)
        else:
            raise ValueError(f"Unknown model_kind: {model_kind}")
        model.fit(features, labels)
        models.append(model)
    positives = frozenset(frame.loc[frame.phototoxic.eq(1), "smiles"].astype(str))
    return PhototoxicityEnsemble(models, tuple(smiles), positives, model_kind, threshold)


def cross_validate_phototoxicity(frame: pd.DataFrame, *, seed: int = 1701) -> dict[str, float | int | str]:
    """Deterministic stratified out-of-fold diagnostic for the small U16 set."""
    smiles = frame.smiles.astype(str).tolist()
    labels = frame.phototoxic.to_numpy(dtype=int)
    features = molecular_features(smiles)
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    predictions = np.zeros(len(frame), dtype=float)
    for fold, (train, test) in enumerate(splitter.split(features, labels)):
        model = RandomForestClassifier(
            n_estimators=240,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced",
            bootstrap=True,
            n_jobs=-1,
            random_state=seed + 1009 * fold,
        )
        model.fit(features[train], labels[train])
        predictions[test] = model.predict_proba(features[test])[:, 1]
    return {
        "validation": "5-fold stratified out-of-fold",
        "n_compounds": int(len(frame)),
        "n_phototoxic": int(labels.sum()),
        "n_non_phototoxic": int((labels == 0).sum()),
        "roc_auc": float(roc_auc_score(labels, predictions)),
        "balanced_accuracy_at_0_5": float(balanced_accuracy_score(labels, predictions >= 0.5)),
        "brier_score": float(brier_score_loss(labels, predictions)),
    }
