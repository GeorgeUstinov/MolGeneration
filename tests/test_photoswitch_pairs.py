from __future__ import annotations

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

from mostgen.photoswitch_pairs import build_diarylethene_pair


def test_diarylethene_pair_is_distinct_and_formula_preserving():
    first = build_diarylethene_pair("F", "OC")
    second = build_diarylethene_pair("F", "OC")
    assert first == second
    assert first.state_a_smiles != first.state_b_smiles
    state_a = Chem.MolFromSmiles(first.state_a_smiles)
    state_b = Chem.MolFromSmiles(first.state_b_smiles)
    assert state_a is not None and state_b is not None
    assert rdMolDescriptors.CalcMolFormula(state_a) == rdMolDescriptors.CalcMolFormula(state_b)
    assert state_b.GetRingInfo().NumRings() == state_a.GetRingInfo().NumRings() + 1
