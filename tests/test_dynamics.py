import pytest
import numpy as np
from rlv_sim import dynamics

@pytest.fixture
def dummy_inputs():
    r = np.array([1e7, 0.0, 0.0])
    v = np.array([0.0, 100.0, 0.0])
    q = np.array([1.0, 0.0, 0.0, 0.0])
    omega = np.array([0.0, 0.0, 0.0])
    m = 1e5
    torque = np.array([0.0, 0.0, 0.0])
    return r, v, q, omega, m, torque

def test_compute_angular_acceleration_zero_torque():
    omega = np.zeros(3)
    torque = np.zeros(3)
    I = np.eye(3)
    I_inv = np.eye(3)
    alpha = dynamics.compute_angular_acceleration(omega, torque, I, I_inv)
    assert np.allclose(alpha, 0.0)

def test_compute_linear_acceleration_zero_forces(monkeypatch):
    def zero_force(*a, **k):
        return np.zeros(3)
    # Patch in dynamics module where they are imported, not in forces module
    monkeypatch.setattr('rlv_sim.dynamics.compute_gravity_force', zero_force)
    monkeypatch.setattr('rlv_sim.dynamics.compute_thrust_force', zero_force)
    monkeypatch.setattr('rlv_sim.dynamics.compute_drag_force', zero_force)
    monkeypatch.setattr('rlv_sim.dynamics.compute_lift_force', zero_force)
    r = np.zeros(3)
    v = np.zeros(3)
    q = np.array([1.0, 0.0, 0.0, 0.0])
    m = 1.0
    a = dynamics.compute_linear_acceleration(r, v, q, m)
    assert np.allclose(a, 0.0)

def test_compute_state_derivative_shapes(dummy_inputs):
    r, v, q, omega, m, torque = dummy_inputs
    deriv = dynamics.compute_state_derivative(r, v, q, omega, m, torque)
    assert deriv.r_dot.shape == (3,)
    assert deriv.v_dot.shape == (3,)
    assert deriv.q_dot.shape == (4,)
    assert deriv.omega_dot.shape == (3,)
    assert isinstance(deriv.m_dot, float)

def test_state_derivative_vector_shape(dummy_inputs):
    r, v, q, omega, m, torque = dummy_inputs
    state_vec = np.concatenate([r, v, q, omega, [m]])
    vec = dynamics.state_derivative_vector(state_vec, t=0.0, torque=torque)
    assert vec.shape == (14,)
    assert np.issubdtype(vec.dtype, np.floating)
