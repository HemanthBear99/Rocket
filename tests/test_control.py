"""Tests for attitude control system."""


import numpy as np
import pytest
from rlv_sim import control
from rlv_sim import constants as C
from rlv_sim import frames

def test_compute_commanded_quaternion_vertical():
    direction = np.array([1.0, 0.0, 0.0])
    q_cmd = control.compute_commanded_quaternion(direction)
    R = frames.quaternion_to_rotation_matrix(q_cmd)
    body_z = np.array([0.0, 0.0, 1.0])
    result = R @ body_z
    np.testing.assert_array_almost_equal(result, direction, decimal=6)

def test_compute_commanded_quaternion_arbitrary():
    direction = np.array([1.0, 1.0, 0.0])
    direction = direction / np.linalg.norm(direction)
    q_cmd = control.compute_commanded_quaternion(direction)
    R = frames.quaternion_to_rotation_matrix(q_cmd)
    body_z = np.array([0.0, 0.0, 1.0])
    result = R @ body_z
    np.testing.assert_array_almost_equal(result, direction, decimal=6)

def test_attitude_error_zero():
    q = np.array([1.0, 0.0, 0.0, 0.0])
    q_error_vector, angle = control.compute_attitude_error(q, q)
    assert angle == pytest.approx(0.0, abs=1e-6)
    np.testing.assert_array_almost_equal(q_error_vector, [0.0, 0.0, 0.0], decimal=6)

def test_attitude_error_small():
    q_current = np.array([1.0, 0.0, 0.0, 0.0])
    small_angle = np.radians(1.0)
    q_commanded = np.array([
        np.cos(small_angle/2), 0.0, 0.0, np.sin(small_angle/2)
    ])
    q_error_vector, angle = control.compute_attitude_error(q_current, q_commanded)
    assert np.degrees(angle) == pytest.approx(1.0, abs=1e-4)
    # q_ev should be approximately [0, 0, sin(θ/2)] for rotation about Z
    assert q_error_vector[2] == pytest.approx(np.sin(small_angle/2), abs=1e-4)

def test_saturate_torque_below_limit():
    torque = np.array([1.0, 1.0, 1.0])
    result = control.saturate_torque(torque)
    np.testing.assert_array_almost_equal(result, torque)

def test_saturate_torque_above_limit():
    large_torque = np.array([1e8, 1e8, 1e8])
    result = control.saturate_torque(large_torque)
    magnitude = np.linalg.norm(result)
    assert magnitude == pytest.approx(C.MAX_TORQUE, abs=1.0)

def test_saturate_torque_direction_preserved():
    torque = np.array([1e8, 0.0, 0.0])
    result = control.saturate_torque(torque)
    np.testing.assert_array_almost_equal(
        result / np.linalg.norm(result),
        torque / np.linalg.norm(torque)
    )

def test_pd_control_zero_error_zero_omega():
    """Zero error quaternion vector + zero omega → zero torque (Doc §17.6)."""
    q_error_vector = np.array([0.0, 0.0, 0.0])  # No orientation error
    error_angle = 0.0
    omega = np.array([0.0, 0.0, 0.0])
    torque = control.pd_control_law(q_error_vector, error_angle, omega)
    np.testing.assert_array_almost_equal(torque, [0.0, 0.0, 0.0])

def test_pd_control_proportional_response():
    """Positive q_ev → positive torque (body-frame convention, Doc §17.6)."""
    # For a small rotation of 0.1 rad about Z, q_ev ≈ sin(0.05) * [0,0,1]
    small_angle = 0.1
    q_error_vector = np.array([0.0, 0.0, np.sin(small_angle / 2)])
    error_angle = small_angle
    omega = np.array([0.0, 0.0, 0.0])
    torque = control.pd_control_law(q_error_vector, error_angle, omega)
    # Torque should be Kp * q_ev (body-frame adapted from Doc §17.6)
    expected = C.KP_ATTITUDE * q_error_vector
    # Check direction is correct (positive Z error → positive Z torque)
    assert torque[2] > 0
    # If not saturated, check magnitude
    if np.linalg.norm(expected) <= C.MAX_TORQUE:
        np.testing.assert_array_almost_equal(torque, expected)

def test_pd_control_derivative_damping():
    """Positive omega → negative torque (damping, Doc §17.6)."""
    q_error_vector = np.array([0.0, 0.0, 0.0])
    error_angle = 0.0
    omega = np.array([0.0, 0.0, 0.1])
    torque = control.pd_control_law(q_error_vector, error_angle, omega)
    assert torque[2] < 0

def test_compute_control_output_keys():
    q_current = np.array([1.0, 0.0, 0.0, 0.0])
    omega = np.array([0.0, 0.0, 0.0])
    direction = np.array([1.0, 0.0, 0.0])
    output = control.compute_control_output(q_current, omega, direction)
    required_keys = [
        'q_commanded', 'error_axis', 'error_angle', 'error_degrees',
        'torque', 'torque_magnitude', 'saturated'
    ]
    for key in required_keys:
        assert key in output
