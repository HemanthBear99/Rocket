import pytest
import numpy as np
from rlv_sim import integrators, state

def test_rk4_step_returns_state():
    s = state.State(
        r=np.array([1.0, 0.0, 0.0]),
        v=np.array([0.0, 1.0, 0.0]),
        q=np.array([1.0, 0.0, 0.0, 0.0]),
        omega=np.zeros(3),
        m=100.0,
        t=0.0
    )
    torque = np.zeros(3)
    s2 = integrators.rk4_step(s, torque, dt=0.1)
    assert isinstance(s2, state.State)
    assert s2.r.shape == (3,)
    assert s2.v.shape == (3,)
    assert s2.q.shape == (4,)
    assert s2.omega.shape == (3,)
    assert isinstance(s2.m, float)
    assert isinstance(s2.t, float)

def test_rk4_step_invalid_dt():
    s = state.State()
    torque = np.zeros(3)
    with pytest.raises(ValueError):
        integrators.rk4_step(s, torque, dt=0.0)

def test_rk4_step_invalid_torque_shape():
    s = state.State()
    torque = np.zeros(2)
    with pytest.raises(ValueError):
        integrators.rk4_step(s, torque, dt=0.1)
