from __future__ import annotations

from dataclasses import dataclass

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

from .chemistry import ChemistryError, standardize_smiles


DIARYLETHENE_OPEN_TEMPLATE = (
    "Cc1sc(-c2ccc({r1})cc2)c(C)c1C1=C("
    "c2c(C)sc(-c3ccc({r2})cc3)c2C)C(F)(F)C(F)(F)C1(F)F"
)


@dataclass(frozen=True)
class PhotoswitchPair:
    state_a_smiles: str
    state_b_smiles: str
    family: str


def close_diarylethene(open_smiles: str) -> str:
    """Apply the canonical 6-pi photocyclization graph edit to a DTE core."""
    mol = Chem.MolFromSmiles(open_smiles)
    if mol is None:
        raise ChemistryError("invalid_diarylethene")
    ring_bonds = {
        frozenset((ring[index], ring[(index + 1) % len(ring)]))
        for ring in mol.GetRingInfo().AtomRings()
        for index in range(len(ring))
    }
    central = None
    for bond in mol.GetBonds():
        ends = (bond.GetBeginAtom(), bond.GetEndAtom())
        if (
            bond.GetBondType() == Chem.BondType.DOUBLE
            and frozenset((ends[0].GetIdx(), ends[1].GetIdx())) in ring_bonds
            and all(any(n.GetIsAromatic() and n.GetSymbol() == "C" for n in atom.GetNeighbors()) for atom in ends)
        ):
            central = bond
            break
    if central is None:
        raise ChemistryError("diarylethene_central_bond_not_found")

    central_ids = [central.GetBeginAtomIdx(), central.GetEndAtomIdx()]
    anchors: list[int] = []
    active_atoms: list[int] = []
    for central_id in central_ids:
        central_atom = mol.GetAtomWithIdx(central_id)
        aromatic_neighbors = [
            atom for atom in central_atom.GetNeighbors() if atom.GetIsAromatic() and atom.GetSymbol() == "C"
        ]
        if len(aromatic_neighbors) != 1:
            raise ChemistryError("diarylethene_anchor_not_unique")
        anchor = aromatic_neighbors[0]
        candidates = []
        for atom in anchor.GetNeighbors():
            if not atom.GetIsAromatic() or atom.GetSymbol() != "C":
                continue
            has_sulfur = any(n.GetIsAromatic() and n.GetSymbol() == "S" for n in atom.GetNeighbors())
            has_methyl = any(
                n.GetSymbol() == "C" and not n.GetIsAromatic() and n.GetDegree() == 1 for n in atom.GetNeighbors()
            )
            if has_sulfur and has_methyl:
                candidates.append(atom.GetIdx())
        if len(candidates) != 1:
            raise ChemistryError("diarylethene_active_atom_not_unique")
        anchors.append(anchor.GetIdx())
        active_atoms.append(candidates[0])

    editable = Chem.RWMol(mol)
    Chem.Kekulize(editable, clearAromaticFlags=True)

    def set_bond(left: int, right: int, bond_type: Chem.BondType) -> None:
        bond = editable.GetBondBetweenAtoms(left, right)
        if bond is None:
            raise ChemistryError("diarylethene_path_bond_missing")
        bond.SetBondType(bond_type)

    set_bond(active_atoms[0], anchors[0], Chem.BondType.SINGLE)
    set_bond(anchors[0], central_ids[0], Chem.BondType.DOUBLE)
    set_bond(central_ids[0], central_ids[1], Chem.BondType.SINGLE)
    set_bond(central_ids[1], anchors[1], Chem.BondType.DOUBLE)
    set_bond(anchors[1], active_atoms[1], Chem.BondType.SINGLE)
    editable.AddBond(active_atoms[0], active_atoms[1], Chem.BondType.SINGLE)
    closed = editable.GetMol()
    try:
        Chem.SanitizeMol(closed)
    except Exception as exc:
        raise ChemistryError(f"invalid_diarylethene_closed_form:{exc}") from exc
    return standardize_smiles(Chem.MolToSmiles(closed, isomericSmiles=True))


def build_diarylethene_pair(substituent_a: str, substituent_b: str) -> PhotoswitchPair:
    raw_open = DIARYLETHENE_OPEN_TEMPLATE.format(r1=substituent_a, r2=substituent_b)
    state_a = standardize_smiles(raw_open)
    state_b = close_diarylethene(state_a)
    if state_a == state_b:
        raise ChemistryError("diarylethene_pair_identical")
    formula_a = rdMolDescriptors.CalcMolFormula(Chem.MolFromSmiles(state_a))
    formula_b = rdMolDescriptors.CalcMolFormula(Chem.MolFromSmiles(state_b))
    if formula_a != formula_b:
        raise ChemistryError(f"diarylethene_formula_mismatch:{formula_a}!={formula_b}")
    return PhotoswitchPair(state_a, state_b, "diarylethene")
