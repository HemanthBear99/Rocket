import pytest
import numpy as np
from rlv_sim import actuator

def test_actuatorstate_init():
    s = actuator.ActuatorState()
    assert np.allclose(s.thrust_dir, [1.0, 0.0, 0.0])
    s2 = actuator.ActuatorState(thrust_dir=[0.0, 1.0, 0.0])
    assert np.allclose(s2.thrust_dir, [0.0, 1.0, 0.0])

def test_limit_rotation_parallel():
    cur = np.array([1.0, 0.0, 0.0])
    des = np.array([1.0, 0.0, 0.0])
    out = actuator._limit_rotation(cur, des, max_rate=1.0, dt=1.0)
    assert np.allclose(out, des)

def test_update_actuator_moves_toward():
    s = actuator.ActuatorState(thrust_dir=[1.0, 0.0, 0.0])
    des = np.array([0.0, 1.0, 0.0])
    s2 = actuator.update_actuator(s, des, dt=0.1)
    assert np.linalg.norm(s2.thrust_dir) > 0.9
