"""Tests for config module."""
import pytest
from rlv_sim import config
from rlv_sim import constants as C


def test_simulation_config_defaults():
    """Test that default config uses constants values."""
    cfg = config.SimulationConfig()
    assert cfg.dt == C.DT
    assert cfg.max_time == C.MAX_TIME
    assert cfg.kp_attitude == C.KP_ATTITUDE
    assert cfg.kd_attitude == C.KD_ATTITUDE
    assert cfg.max_torque == C.MAX_TORQUE


def test_simulation_config_custom_values():
    """Test creating config with custom values."""
    cfg = config.SimulationConfig(dt=0.01, max_time=100.0, verbose=False)
    assert cfg.dt == 0.01
    assert cfg.max_time == 100.0
    assert cfg.verbose is False


def test_simulation_config_frozen():
    """Test that config is immutable (frozen)."""
    cfg = config.SimulationConfig()
    with pytest.raises(Exception):  # FrozenInstanceError
        cfg.dt = 0.5


def test_create_default_config():
    """Test create_default_config factory function."""
    cfg = config.create_default_config()
    assert isinstance(cfg, config.SimulationConfig)
    assert cfg.dt == C.DT


def test_create_test_config():
    """Test create_test_config factory function."""
    cfg = config.create_test_config()
    assert cfg.dt == 0.1
    assert cfg.max_time == 10.0
    assert cfg.verbose is False


def test_create_test_config_custom():
    """Test create_test_config with custom parameters."""
    cfg = config.create_test_config(dt=0.05, max_time=5.0)
    assert cfg.dt == 0.05
    assert cfg.max_time == 5.0
