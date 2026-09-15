from __future__ import annotations

import pytest

from mostgen.chemistry import (
    ChemistryError,
    build_pair,
    molecular_formula,
    safety_veto,
    standardize_smiles,
    validate_pair,
)
from mostgen.config import load_config


@pytest.fixture(scope="module")
def config():
    return load_config(mode="smoke")


def test_standardize_smiles_is_canonical_and_deterministic():
    assert standardize_smiles("C(C)O") == "CCO"
    assert standardize_smiles("CCO") == "CCO"


def test_invalid_smiles_fails_closed():
    with pytest.raises(ChemistryError):
        standardize_smiles("C1(not-smiles")


@pytest.mark.parametrize("family", ["nbd_qc", "dewar_pyrimidinone", "spiropyran"])
def test_deterministic_isomer_pair_preserves_formula(config, family):
    first = build_pair(config, family, "S01", "S23")
    second = build_pair(config, family, "S01", "S23")
    assert first == second
    assert first.smiles != first.charged_smiles
    assert molecular_formula(first.smiles) == molecular_formula(first.charged_smiles)
    assert validate_pair(first, config) == (True, [])


@pytest.mark.parametrize("smiles", [
    "C1=CC(=O)OC2=CC3=C(C=CO3)C=C21",
    "C1=CC2=C(C=CO2)C3=C1C=CC(=O)O3",
    "COC1=C2C=CC(=O)OC2=CC3=C1C=CO3",
])
def test_linear_angular_and_substituted_furocoumarins_are_vetoed(config, smiles):
    result = safety_veto(smiles, config)
    assert result.veto
    assert result.psoralen_alert


def test_reactive_and_unsupported_elements_are_vetoed(config):
    peroxide = safety_veto("COOC", config)
    sodium = safety_veto("C[Na]", config)
    assert peroxide.veto and "OO" in peroxide.reactive_alerts
    assert sodium.veto and "Na" in sodium.unsupported_elements


def test_similarity_warning_is_not_itself_a_veto(config):
    result = safety_veto("O=c1ccc2ccccc2o1", config)  # coumarin, no fused furan
    assert not result.psoralen_alert
    assert isinstance(result.psoralen_similarity_warning, bool)


def test_all_configured_synthons_build_at_least_acceptance_quota(config):
    for family in config["families"]:
        valid = 0
        for first in config["synthons"]:
            for second in config["synthons"]:
                try:
                    build_pair(config, family, first["id"], second["id"])
                    valid += 1
                except ChemistryError:
                    pass
        assert valid >= 1_000

