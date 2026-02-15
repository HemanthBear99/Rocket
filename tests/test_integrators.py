import pytest
import numpy as np
from rlv_sim import integrators, state
from rlv_sim import constants as C

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

def test_rk4_step_nan_torque():
    s = state.State()
    torque = np.array([np.nan, 0.0, 0.0])
    with pytest.raises(ValueError):
        integrators.rk4_step(s, torque, dt=0.1)

def test_euler_step_returns_state():
    s = state.State(
        r=C.INITIAL_POSITION.copy(),
        v=C.INITIAL_VELOCITY.copy(),
        q=C.INITIAL_QUATERNION.copy(),
        omega=np.zeros(3),
        m=C.INITIAL_MASS,
        t=0.0
    )
    torque = np.zeros(3)
    s2 = integrators.euler_step(s, torque, dt=0.1)
    assert isinstance(s2, state.State)
    assert s2.t == 0.1

def test_euler_step_mass_floor():
    """Test that euler step doesn't go below dry mass."""
    s = state.State(
        r=C.INITIAL_POSITION.copy(),
        v=C.INITIAL_VELOCITY.copy(),
        q=C.INITIAL_QUATERNION.copy(),
        omega=np.zeros(3),
        m=C.DRY_MASS + 100.0,  # Just above dry mass
        t=0.0
    )
    torque = np.zeros(3)
    # Run many steps to consume propellant
    for _ in range(10000):
        s = integrators.euler_step(s, torque, dt=0.1, thrust_on=True)
    assert s.m >= C.DRY_MASS

def test_integrate_rk4():
    s = state.State()
    torque = np.zeros(3)
    s2 = integrators.integrate(s, torque, dt=0.1, method='rk4')
    assert isinstance(s2, state.State)

def test_integrate_euler():
    s = state.State()
    torque = np.zeros(3)
    s2 = integrators.integrate(s, torque, dt=0.1, method='euler')
    assert isinstance(s2, state.State)

def test_integrate_unknown_method():
    s = state.State()
    torque = np.zeros(3)
    with pytest.raises(ValueError):
        integrators.integrate(s, torque, dt=0.1, method='unknown')

def test_rk4_with_throttle():
    s = state.State()
    torque = np.zeros(3)
    s2 = integrators.rk4_step(s, torque, dt=0.1, thrust_on=True, throttle=0.5)
    assert isinstance(s2, state.State)

