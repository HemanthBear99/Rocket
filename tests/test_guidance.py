import pytest
import numpy as np
from rlv_sim import guidance, constants as C

def test_compute_local_vertical():
    r = np.array([2.0, 0.0, 0.0])
    v = guidance.compute_local_vertical(r)
    assert np.allclose(v, [1.0, 0.0, 0.0])
    r = np.zeros(3)
    v = guidance.compute_local_vertical(r)
    assert np.allclose(v, [1.0, 0.0, 0.0])

def test_compute_local_frame():
    r = np.array([1.0, 0.0, 0.0])
    vertical, east, north = guidance.compute_local_frame(r)
    assert np.allclose(vertical, [1.0, 0.0, 0.0])
    assert np.allclose(np.dot(east, vertical), 0.0, atol=1e-12)
    assert np.allclose(np.dot(north, vertical), 0.0, atol=1e-12)
    assert np.allclose(np.dot(east, north), 0.0, atol=1e-12)
    assert np.isclose(np.linalg.norm(east), 1.0)
    assert np.isclose(np.linalg.norm(north), 1.0)

def test_compute_local_horizontal():
    r = np.array([1.0, 0.0, 0.0])
    v = np.array([0.0, 1.0, 0.0])
    h = guidance.compute_local_horizontal(r, v)
    assert np.isclose(np.linalg.norm(h), 1.0)
    r = np.zeros(3)
    v = np.zeros(3)
    h = guidance.compute_local_horizontal(r, v)
    assert np.allclose(h, [1.0, 0.0, 0.0])

def test_gamma_profile_from_altitude():
    # Check monotonic decrease and bounds
    last = 100.0
    for alt in np.linspace(0, 100000, 20):
        gamma = guidance.gamma_profile_from_altitude(alt)
        assert 0.0 < gamma <= np.pi/2
        assert gamma <= last + 1e-8
        last = gamma

def test_compute_blend_parameter():
    s = C.GRAVITY_TURN_START_ALTITUDE
    r = C.GRAVITY_TURN_TRANSITION_RANGE
    assert guidance.compute_blend_parameter(s - 1) == 0.0
    assert guidance.compute_blend_parameter(s + r + 1) == 1.0
    mid = s + r/2
    val = guidance.compute_blend_parameter(mid)
    assert 0.0 < val < 1.0

def test_compute_desired_thrust_direction_shape():
    r = np.array([C.R_EARTH + 1000.0, 0.0, 0.0])
    v = np.array([0.0, 100.0, 0.0])
    t = 0.0
    thrust_dir, gamma_cmd, gamma_meas, gs = guidance.compute_desired_thrust_direction(r, v, t)
    assert thrust_dir.shape == (3,)
    assert np.isclose(np.linalg.norm(thrust_dir), 1.0, atol=1e-8)
    assert isinstance(gamma_cmd, float)
    assert isinstance(gamma_meas, float)
    assert isinstance(gs, guidance.GuidanceState)

def test_compute_guidance_output_keys():
    r = np.array([C.R_EARTH + 1000.0, 0.0, 0.0])
    v = np.array([0.0, 100.0, 0.0])
    t = 0.0
    m = C.INITIAL_MASS
    out, gs = guidance.compute_guidance_output(r, v, t, m)
    required_keys = [
        'thrust_direction', 'phase', 'thrust_on', 'pitch_angle', 'gamma_angle',
        'gamma_command_deg', 'gamma_measured_deg', 'velocity_tilt_deg',
        'blend_alpha', 'altitude', 'velocity', 'local_vertical',
        'local_horizontal', 'prograde', 'throttle', 'v_rel', 'v_rel_mag'
    ]
    for k in required_keys:
        assert k in out
    assert np.isclose(np.linalg.norm(out['thrust_direction']), 1.0, atol=1e-8)
    assert out['altitude'] > 0
    assert out['velocity'] >= 0
    assert 0.0 <= out['blend_alpha'] <= 1.0
    assert 0.0 <= out['throttle'] <= 1.0
    assert isinstance(gs, guidance.GuidanceState)
