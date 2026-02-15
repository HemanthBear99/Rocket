import pytest
import numpy as np
from rlv_sim import state, constants as C

@pytest.fixture
def default_state():
    return state.State()

@pytest.fixture
def custom_state():
    return state.State(
        r=np.array([1.0, 2.0, 3.0]),
        v=np.array([4.0, 5.0, 6.0]),
        q=np.array([0.707, 0.0, 0.707, 0.0]),
        omega=np.array([0.1, 0.2, 0.3]),
        m=100.0,
        t=42.0
    )

def test_state_init_types(default_state):
    assert isinstance(default_state.r, np.ndarray)
    assert isinstance(default_state.v, np.ndarray)
    assert isinstance(default_state.q, np.ndarray)
    assert isinstance(default_state.omega, np.ndarray)
    assert isinstance(default_state.m, float)
    assert isinstance(default_state.t, float)

def test_state_copy(custom_state):
    s2 = custom_state.copy()
    assert np.allclose(s2.r, custom_state.r)
    assert np.allclose(s2.v, custom_state.v)
    assert np.allclose(s2.q, custom_state.q)
    assert np.allclose(s2.omega, custom_state.omega)
    assert s2.m == custom_state.m
    assert s2.t == custom_state.t
    # Ensure deep copy
    s2.r[0] += 1
    assert not np.allclose(s2.r, custom_state.r)

def test_to_vector_and_from_vector(custom_state):
    vec = custom_state.to_vector()
    assert vec.shape == (14,)
    s2 = state.State.from_vector(vec, t=custom_state.t)
    assert np.allclose(s2.r, custom_state.r)
    assert np.allclose(s2.v, custom_state.v)
    assert np.allclose(s2.q, custom_state.q)
    assert np.allclose(s2.omega, custom_state.omega)
    assert s2.m == pytest.approx(custom_state.m)
    assert s2.t == pytest.approx(custom_state.t)

def test_altitude_and_speed(custom_state):
    # Altitude = norm(r) - R_EARTH
    expected_alt = np.linalg.norm(custom_state.r) - C.R_EARTH
    assert custom_state.altitude == pytest.approx(expected_alt)
    # Speed = norm(v)
    expected_speed = np.linalg.norm(custom_state.v)
    assert custom_state.speed == pytest.approx(expected_speed)

def test_propellant_remaining(custom_state):
    # Should be max(0, m - DRY_MASS)
    expected = max(0.0, custom_state.m - C.DRY_MASS)
    assert custom_state.propellant_remaining == pytest.approx(expected)
    # If m < DRY_MASS, should be 0
    s2 = custom_state.copy()
    s2.m = 0.5 * C.DRY_MASS
    assert s2.propellant_remaining == 0.0

def test_str(custom_state):
    s = str(custom_state)
    assert "State(" in s
    assert "alt=" in s
    assert "v=" in s
    assert "kg)" in s

def test_create_initial_state():
    s = state.create_initial_state()
    assert np.allclose(s.r, C.INITIAL_POSITION)
    assert np.allclose(s.v, C.INITIAL_VELOCITY)
    assert np.allclose(s.q, C.INITIAL_QUATERNION)
    assert np.allclose(s.omega, C.INITIAL_OMEGA)
    assert s.m == pytest.approx(C.INITIAL_MASS)
    assert s.t == 0.0
