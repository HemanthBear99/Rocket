"""Tests for attitude control system."""

import unittest

import numpy as np

from rlv_sim import control
from rlv_sim import constants as C
from rlv_sim import frames


class TestComputeCommandedQuaternion(unittest.TestCase):
    """Tests for compute_commanded_quaternion function."""
    
    def test_vertical_direction(self):
        """Commanding vertical (radial) direction."""
        direction = np.array([1.0, 0.0, 0.0])  # Radial at launch site
        q_cmd = control.compute_commanded_quaternion(direction)
        
        # Apply rotation to body +Z and verify it points along direction
        R = frames.quaternion_to_rotation_matrix(q_cmd)
        body_z = np.array([0.0, 0.0, 1.0])
        result = R @ body_z
        np.testing.assert_array_almost_equal(result, direction, decimal=6)
    
    def test_arbitrary_direction(self):
        """Commanding arbitrary direction."""
        direction = np.array([1.0, 1.0, 0.0])
        direction = direction / np.linalg.norm(direction)  # Normalize
        
        q_cmd = control.compute_commanded_quaternion(direction)
        
        # Verify body +Z points along direction
        R = frames.quaternion_to_rotation_matrix(q_cmd)
        body_z = np.array([0.0, 0.0, 1.0])
        result = R @ body_z
        np.testing.assert_array_almost_equal(result, direction, decimal=6)


class TestAttitudeError(unittest.TestCase):
    """Tests for attitude error computation."""
    
    def test_zero_error(self):
        """Same current and commanded should give zero error."""
        q = np.array([1.0, 0.0, 0.0, 0.0])
        axis, angle = control.compute_attitude_error(q, q)
        self.assertAlmostEqual(angle, 0.0, places=6)
    
    def test_small_error(self):
        """Small rotation should give small error."""
        q_current = np.array([1.0, 0.0, 0.0, 0.0])
        # Small rotation about Z (1 degree)
        small_angle = np.radians(1.0)
        q_commanded = np.array([
            np.cos(small_angle/2), 0.0, 0.0, np.sin(small_angle/2)
        ])
        
        axis, angle = control.compute_attitude_error(q_current, q_commanded)
        self.assertAlmostEqual(np.degrees(angle), 1.0, places=4)


class TestTorqueSaturation(unittest.TestCase):
    """Tests for torque saturation."""
    
    def test_below_limit(self):
        """Torque below limit should not be saturated."""
        torque = np.array([1.0, 1.0, 1.0])  # Very small
        result = control.saturate_torque(torque)
        np.testing.assert_array_almost_equal(result, torque)
    
    def test_above_limit(self):
        """Torque above limit should be saturated."""
        large_torque = np.array([1e8, 1e8, 1e8])
        result = control.saturate_torque(large_torque)
        magnitude = np.linalg.norm(result)
        self.assertAlmostEqual(magnitude, C.MAX_TORQUE, places=0)
    
    def test_direction_preserved(self):
        """Saturation should preserve direction."""
        torque = np.array([1e8, 0.0, 0.0])
        result = control.saturate_torque(torque)
        # Direction should still be along X
        np.testing.assert_array_almost_equal(
            result / np.linalg.norm(result),
            torque / np.linalg.norm(torque)
        )


class TestPDControl(unittest.TestCase):
    """Tests for PD control law."""
    
    def test_zero_error_zero_omega_gives_zero_torque(self):
        """Zero error and zero angular velocity should give zero torque."""
        error_axis = np.array([0.0, 0.0, 1.0])
        error_angle = 0.0
        omega = np.array([0.0, 0.0, 0.0])
        
        torque = control.pd_control_law(error_axis, error_angle, omega)
        np.testing.assert_array_almost_equal(torque, [0.0, 0.0, 0.0])
    
    def test_proportional_response(self):
        """Error should produce proportional torque."""
        error_axis = np.array([0.0, 0.0, 1.0])
        error_angle = 0.1  # radians
        omega = np.array([0.0, 0.0, 0.0])
        
        torque = control.pd_control_law(error_axis, error_angle, omega)
        
        expected = C.KP_ATTITUDE * error_angle * error_axis
        np.testing.assert_array_almost_equal(torque, expected)
    
    def test_derivative_damping(self):
        """Angular velocity should produce damping torque."""
        error_axis = np.array([0.0, 0.0, 1.0])
        error_angle = 0.0
        omega = np.array([0.0, 0.0, 0.1])  # Spinning about Z
        
        torque = control.pd_control_law(error_axis, error_angle, omega)
        
        # Should oppose the rotation (negative Z)
        self.assertTrue(torque[2] < 0)


class TestControlOutput(unittest.TestCase):
    """Tests for compute_control_output function."""
    
    def test_output_contains_required_keys(self):
        """Control output should contain all required keys."""
        q_current = np.array([1.0, 0.0, 0.0, 0.0])
        omega = np.array([0.0, 0.0, 0.0])
        direction = np.array([1.0, 0.0, 0.0])
        
        output = control.compute_control_output(q_current, omega, direction)
        
        required_keys = [
            'q_commanded', 'error_axis', 'error_angle', 'error_degrees',
            'torque', 'torque_magnitude', 'saturated'
        ]
        for key in required_keys:
            self.assertIn(key, output)


if __name__ == '__main__':
    unittest.main()
