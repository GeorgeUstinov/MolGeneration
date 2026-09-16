from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable
import xml.etree.ElementTree as ET

from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator


class ChemistryError(ValueError):
    """A molecule cannot be standardized or does not satisfy the chemistry contract."""


@dataclass(frozen=True)
class MoleculePair:
    smiles: str
    charged_smiles: str
    family: str
    synthon_a: str
    synthon_b: str
    formula: str


@dataclass(frozen=True)
class SafetyVeto:
    veto: bool
    psoralen_alert: bool
    known_phototoxic_match: bool
    psoralen_similarity: float
    psoralen_similarity_warning: bool
    reactive_alerts: tuple[str, ...]
    unsupported_elements: tuple[str, ...]
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["reactive_alerts"] = ";".join(self.reactive_alerts)
        result["unsupported_elements"] = ";".join(self.unsupported_elements)
        result["reasons"] = ";".join(self.reasons)
        return result


def parse_smiles(smiles: str) -> Chem.Mol:
    if not isinstance(smiles, str) or not smiles.strip():
        raise ChemistryError("empty_smiles")
    mol = Chem.MolFromSmiles(smiles.strip(), sanitize=True)
    if mol is None:
        raise ChemistryError("invalid_smiles")
    return mol


@lru_cache(maxsize=100_000)
def standardize_smiles(smiles: str) -> str:
    mol = parse_smiles(smiles)
    try:
        mol = rdMolStandardize.Cleanup(mol)
        mol = rdMolStandardize.FragmentParent(mol)
        uncharger = rdMolStandardize.Uncharger(canonicalOrder=True)
        # Preserve genuine zwitterions: only accept uncharging when net formal
        # charge is non-zero.  Merocyanines contain balanced explicit charges.
        if Chem.GetFormalCharge(mol) != 0:
            mol = uncharger.uncharge(mol)
        Chem.SanitizeMol(mol)
    except Exception as exc:  # RDKit exposes several sanitizer exception types.
        raise ChemistryError(f"standardization_failed:{exc}") from exc
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def molecular_formula(smiles: str) -> str:
    return rdMolDescriptors.CalcMolFormula(parse_smiles(smiles))


def build_pair(config: dict[str, Any], family: str, synthon_a: str, synthon_b: str) -> MoleculePair:
    if family not in config["families"]:
        raise ChemistryError(f"unsupported_family:{family}")
    library = {entry["id"]: entry for entry in config["synthons"]}
    try:
        first, second = library[synthon_a], library[synthon_b]
    except KeyError as exc:
        raise ChemistryError(f"unknown_synthon:{exc.args[0]}") from exc
    if not first.get("commercial") or not second.get("commercial"):
        raise ChemistryError("noncommercial_synthon")
    family_config = config["families"][family]
    raw_ground = family_config["ground_template"].format(r1=first["smiles"], r2=second["smiles"])
    raw_charged = family_config["charged_template"].format(r1=first["smiles"], r2=second["smiles"])
    ground = standardize_smiles(raw_ground)
    charged = standardize_smiles(raw_charged)
    ground_formula = molecular_formula(ground)
    charged_formula = molecular_formula(charged)
    if ground_formula != charged_formula:
        raise ChemistryError(f"isomer_formula_mismatch:{ground_formula}!={charged_formula}")
    if ground == charged:
        raise ChemistryError("isomer_pair_identical")
    return MoleculePair(ground, charged, family, synthon_a, synthon_b, ground_formula)


def murcko_scaffold(smiles: str) -> str:
    mol = parse_smiles(smiles)
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    if scaffold.GetNumAtoms() == 0:
        return Chem.MolToSmiles(mol, canonical=True)
    generic = MurckoScaffold.MakeScaffoldGeneric(scaffold)
    return Chem.MolToSmiles(generic, canonical=True)


@lru_cache(maxsize=8)
def _morgan_generator(bits: int):
    return GetMorganGenerator(radius=2, fpSize=bits, includeChirality=True)


@lru_cache(maxsize=100_000)
def fingerprint(smiles: str, bits: int = 256):
    return _morgan_generator(bits).GetFingerprint(parse_smiles(smiles))


def fingerprint_indices(smiles: str, bits: int = 256) -> list[int]:
    return list(fingerprint(smiles, bits).GetOnBits())


def tanimoto(smiles_a: str, smiles_b: str, bits: int = 256) -> float:
    return float(DataStructs.TanimotoSimilarity(fingerprint(smiles_a, bits), fingerprint(smiles_b, bits)))


def max_similarity(smiles: str, references: Iterable[str], bits: int = 256) -> float:
    query = fingerprint(smiles, bits)
    ref_fps = [fingerprint(reference, bits) for reference in references]
    if not ref_fps:
        return 0.0
    return float(max(DataStructs.BulkTanimotoSimilarity(query, ref_fps)))


def descriptors(smiles: str) -> dict[str, float]:
    mol = parse_smiles(smiles)
    atoms = max(1, mol.GetNumHeavyAtoms())
    rings = float(rdMolDescriptors.CalcNumRings(mol))
    aromatic = float(rdMolDescriptors.CalcNumAromaticRings(mol))
    rotors = float(Lipinski.NumRotatableBonds(mol))
    hetero = float(rdMolDescriptors.CalcNumHeteroatoms(mol))
    charge_separation = float(sum(abs(atom.GetFormalCharge()) for atom in mol.GetAtoms()))
    return {
        "mol_wt": float(Descriptors.MolWt(mol)),
        "logp": float(Descriptors.MolLogP(mol)),
        "tpsa": float(rdMolDescriptors.CalcTPSA(mol)),
        "hbd": float(Lipinski.NumHDonors(mol)),
        "hba": float(Lipinski.NumHAcceptors(mol)),
        "rings": rings,
        "aromatic_rings": aromatic,
        "rotatable_bonds": rotors,
        "hetero_fraction": hetero / atoms,
        "fraction_csp3": float(rdMolDescriptors.CalcFractionCSP3(mol)),
        "formal_charge": float(Chem.GetFormalCharge(mol)),
        "charge_separation": charge_separation,
        "heavy_atoms": float(atoms),
    }


def synthetic_accessibility(smiles: str) -> float:
    """Transparent Ertl-inspired proxy on a 1 (easy) to 10 (hard) scale.

    This is deliberately named and reported as a proxy; production review may
    replace it with the RDKit contrib SA implementation without changing the
    CSV contract.
    """
    d = descriptors(smiles)
    raw = (
        1.0
        + 0.20 * d["rings"]
        + 0.13 * d["rotatable_bonds"]
        + 0.035 * max(0.0, d["heavy_atoms"] - 12.0)
        + 0.18 * abs(d["logp"] - 2.0)
        + 0.12 * d["charge_separation"]
    )
    return min(10.0, max(1.0, raw))


@lru_cache(maxsize=256)
def _query(pattern: str) -> Chem.Mol | None:
    query = Chem.MolFromSmarts(pattern)
    return query if query is not None else Chem.MolFromSmiles(pattern)


@lru_cache(maxsize=64)
def _core_query(pattern: str) -> Chem.Mol | None:
    """A sanitized aromatic topology query for fused-core matching."""
    return Chem.MolFromSmiles(pattern)


@lru_cache(maxsize=4)
def _u16_positive_registry(directory_name: str) -> tuple[str, ...]:
    """Read exact experimental positives from the pinned U16 QsarDB archive."""
    directory = Path(directory_name)
    if not directory.is_absolute():
        directory = Path(__file__).resolve().parents[1] / directory
    compounds_xml = directory / "compounds" / "compounds.xml"
    values_path = directory / "properties" / "3T3_NRU_Phototoxicity" / "values"
    if not compounds_xml.exists() or not values_path.exists():
        return ()
    namespace = {"q": "http://www.qsardb.org/QDB"}
    compounds: dict[int, str] = {}
    for node in ET.parse(compounds_xml).getroot().findall("q:Compound", namespace):
        cid = int(node.findtext("q:Id", namespaces=namespace))
        inchi = node.findtext("q:InChI", default="", namespaces=namespace)
        mol = Chem.MolFromInchi(inchi) if inchi else None
        if mol is not None:
            compounds[cid] = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    positives = []
    with values_path.open(encoding="utf-8") as handle:
        next(handle, None)
        for line in handle:
            fields = line.rstrip().split("\t")
            if len(fields) == 2 and fields[1] == "P" and int(fields[0]) in compounds:
                try:
                    positives.append(standardize_smiles(compounds[int(fields[0])]))
                except ChemistryError:
                    pass
    return tuple(sorted(set(positives)))


def safety_veto(smiles: str, config: dict[str, Any]) -> SafetyVeto:
    reasons: list[str] = []
    try:
        mol = parse_smiles(smiles)
    except ChemistryError as exc:
        return SafetyVeto(True, False, False, 0.0, False, (), (), (str(exc),))

    allowed = set(config["elements"])
    unsupported = tuple(sorted({atom.GetSymbol() for atom in mol.GetAtoms()} - allowed))
    if unsupported:
        reasons.append("unsupported_elements")

    safety = config["safety"]
    psoralen = False
    for pattern in safety["psoralen_core_smarts"]:
        query = _core_query(pattern)
        if query is not None and mol.HasSubstructMatch(query):
            psoralen = True
            reasons.append("psoralen_or_furocoumarin_core")
            break

    canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    known = False
    known_canonical: list[str] = []
    registry = _u16_positive_registry(str(safety.get("u16_qsardb_directory", ""))) if safety.get("u16_qsardb_directory") else ()
    for known_smiles in [*safety["known_phototoxic_smiles"], *registry]:
        try:
            normalized_known = standardize_smiles(known_smiles)
            known_canonical.append(normalized_known)
            if canonical == normalized_known:
                known = True
                reasons.append("known_phototoxic_structure")
        except ChemistryError:
            continue
    psoralen_similarity = max_similarity(canonical, known_canonical) if known_canonical else 0.0
    psoralen_similarity_warning = (
        not psoralen
        and not known
        and psoralen_similarity >= float(safety.get("psoralen_similarity_warning_threshold", 0.45))
    )

    reactive: list[str] = []
    for pattern in safety["reactive_fragments"]:
        query = _query(pattern)
        if query is not None and mol.HasSubstructMatch(query):
            reactive.append(pattern)
    if reactive:
        reasons.append("reactive_or_unstable_alert")

    return SafetyVeto(
        veto=bool(psoralen or known or reactive or unsupported),
        psoralen_alert=psoralen,
        known_phototoxic_match=known,
        psoralen_similarity=psoralen_similarity,
        psoralen_similarity_warning=psoralen_similarity_warning,
        reactive_alerts=tuple(reactive),
        unsupported_elements=unsupported,
        reasons=tuple(reasons),
    )


def validate_pair(pair: MoleculePair, config: dict[str, Any]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    try:
        ground_formula = molecular_formula(pair.smiles)
        charged_formula = molecular_formula(pair.charged_smiles)
        if ground_formula != charged_formula:
            reasons.append("isomer_formula_mismatch")
        if pair.smiles == pair.charged_smiles:
            reasons.append("isomer_pair_identical")
    except ChemistryError as exc:
        reasons.append(str(exc))
    veto = safety_veto(pair.smiles, config)
    reasons.extend(veto.reasons)
    return not reasons, sorted(set(reasons))
