from __future__ import annotations

import copy

import pytest

from mostgen.config import ConfigError, load_config, validate_config


def test_budget_guards():
    config = load_config(mode="smoke")
    bad = copy.deepcopy(config)
    bad["project"]["max_training_structures"] = 30_001
    with pytest.raises(ConfigError):
        validate_config(bad)
    bad = copy.deepcopy(config)
    bad["execution"]["gpu_hours"] = 8.1
    with pytest.raises(ConfigError):
        validate_config(bad)


def test_production_mode_is_explicit():
    config = load_config(mode="production")
    assert config["execution"]["backend"] == "reinvent4"
    assert config["execution"]["mode"] == "production"
