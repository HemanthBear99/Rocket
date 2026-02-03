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
    axis, angle = control.compute_attitude_error(q, q)
    assert angle == pytest.approx(0.0, abs=1e-6)

def test_attitude_error_small():
    q_current = np.array([1.0, 0.0, 0.0, 0.0])
    small_angle = np.radians(1.0)
    q_commanded = np.array([
        np.cos(small_angle/2), 0.0, 0.0, np.sin(small_angle/2)
    ])
    axis, angle = control.compute_attitude_error(q_current, q_commanded)
    assert np.degrees(angle) == pytest.approx(1.0, abs=1e-4)

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
    error_axis = np.array([0.0, 0.0, 1.0])
    error_angle = 0.0
    omega = np.array([0.0, 0.0, 0.0])
    torque = control.pd_control_law(error_axis, error_angle, omega)
    np.testing.assert_array_almost_equal(torque, [0.0, 0.0, 0.0])

def test_pd_control_proportional_response():
    error_axis = np.array([0.0, 0.0, 1.0])
    error_angle = 0.1
    omega = np.array([0.0, 0.0, 0.0])
    torque = control.pd_control_law(error_axis, error_angle, omega)
    expected = C.KP_ATTITUDE * error_angle * error_axis
    np.testing.assert_array_almost_equal(torque, expected)

def test_pd_control_derivative_damping():
    error_axis = np.array([0.0, 0.0, 1.0])
    error_angle = 0.0
    omega = np.array([0.0, 0.0, 0.1])
    torque = control.pd_control_law(error_axis, error_angle, omega)
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
