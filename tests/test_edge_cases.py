"""Tests for edge cases and boundary conditions."""

import unittest

import numpy as np

from rlv_sim import forces
from rlv_sim import guidance
from rlv_sim import frames
from rlv_sim import constants as C
from rlv_sim.utils import compute_relative_velocity


class TestAtmosphereEdgeCases(unittest.TestCase):
    """Edge case tests for atmosphere model."""
    
    def test_negative_altitude(self):
        """Negative altitude should be treated as zero."""
        T, P, rho, a = forces.compute_atmosphere_properties(-1000.0)
        T0, P0, rho0, a0 = forces.compute_atmosphere_properties(0.0)
        
        # Should be same as sea level
        self.assertAlmostEqual(T, T0)
        self.assertAlmostEqual(P, P0)
        self.assertAlmostEqual(rho, rho0)
    
    def test_extreme_altitude(self):
        """Very high altitude should have near-zero density."""
        T, P, rho, a = forces.compute_atmosphere_properties(150000.0)  # 150 km
        
        self.assertLessEqual(rho, 1e-10)
    
    def test_karman_line(self):
        """100 km (Karman line) should have very low density."""
        T, P, rho, a = forces.compute_atmosphere_properties(100000.0)
        
        # Density should be < 1e-5 kg/m³ at 100 km
        self.assertLess(rho, 1e-5)


class TestGravityEdgeCases(unittest.TestCase):
    """Edge case tests for gravity model."""
    
    def test_zero_position_vector(self):
        """Zero position vector should return zero force (singularity protection)."""
        r = np.array([0.0, 0.0, 0.0])
        m = 1000.0
        F = forces.compute_gravity_force(r, m)
        np.testing.assert_array_equal(F, [0.0, 0.0, 0.0])
    
    def test_gravity_decreases_with_altitude(self):
        """Gravity should decrease with increasing altitude."""
        m = 1000.0
        r_surface = np.array([C.R_EARTH, 0.0, 0.0])
        r_high = np.array([C.R_EARTH + 100000.0, 0.0, 0.0])  # 100 km
        
        F_surface = forces.compute_gravity_force(r_surface, m)
        F_high = forces.compute_gravity_force(r_high, m)
        
        self.assertGreater(np.linalg.norm(F_surface), np.linalg.norm(F_high))


class TestDragEdgeCases(unittest.TestCase):
    """Edge case tests for drag model."""
    
    def test_zero_velocity(self):
        """Zero air-relative velocity should give zero drag."""
        r = np.array([C.R_EARTH + 1000.0, 0.0, 0.0])
        # Set velocity = co-rotation + wind so air-relative velocity is zero
        omega_earth = np.array([0.0, 0.0, C.EARTH_ROTATION_RATE])
        from rlv_sim.utils import _wind_vector
        v = np.cross(omega_earth, r) + _wind_vector(r)

        F_drag = forces.compute_drag_force(r, v)
        np.testing.assert_array_almost_equal(F_drag, [0.0, 0.0, 0.0], decimal=8)
    
    def test_space_vacuum(self):
        """In space, drag should be effectively zero."""
        r = np.array([C.R_EARTH + 200000.0, 0.0, 0.0])  # 200 km
        v = np.array([7800.0, 0.0, 0.0])  # Typical orbital velocity
        
        F_drag = forces.compute_drag_force(r, v)
        
        # Drag should be negligible at 200 km
        self.assertLess(np.linalg.norm(F_drag), 0.1)


class TestGuidanceEdgeCases(unittest.TestCase):
    """Edge case tests for guidance system."""
    
    def test_zero_position(self):
        """Zero position should return fallback vertical."""
        r = np.array([0.0, 0.0, 0.0])
        vertical = guidance.compute_local_vertical(r)
        np.testing.assert_array_equal(vertical, [1.0, 0.0, 0.0])
    
    def test_zero_velocity(self):
        """Zero relative velocity should use vertical for horizontal."""
        r = np.array([C.R_EARTH, 0.0, 0.0])
        # Set velocity to wind velocity
        omega_earth = np.array([0.0, 0.0, C.EARTH_ROTATION_RATE])
        v = np.cross(omega_earth, r)
        
        horizontal = guidance.compute_local_horizontal(r, v)
        vertical = guidance.compute_local_vertical(r)
        
        # Should fall back to vertical
        np.testing.assert_array_almost_equal(horizontal, vertical)


class TestQuaternionEdgeCases(unittest.TestCase):
    """Edge case tests for quaternion operations."""
    
    def test_parallel_vectors(self):
        """direction_to_quaternion with parallel vectors."""
        z = np.array([0.0, 0.0, 1.0])
        q = frames.direction_to_quaternion(z, z)
        
        # Should be identity
        self.assertAlmostEqual(q[0], 1.0, places=4)
    
    def test_anti_parallel_vectors(self):
        """direction_to_quaternion with anti-parallel vectors."""
        z = np.array([0.0, 0.0, 1.0])
        neg_z = np.array([0.0, 0.0, -1.0])
        q = frames.direction_to_quaternion(neg_z, z)
        
        # Should be valid unit quaternion
        self.assertAlmostEqual(np.linalg.norm(q), 1.0, places=6)
        
        # Should rotate z to -z
        R = frames.quaternion_to_rotation_matrix(q)
        result = R @ z
        np.testing.assert_array_almost_equal(result, neg_z, decimal=6)


class TestRelativeVelocityUtility(unittest.TestCase):
    """Tests for the relative velocity utility function."""
    
    def test_at_rest_on_equator(self):
        """At rest on equator should have zero relative velocity."""
        r = np.array([C.R_EARTH, 0.0, 0.0])
        # Velocity due to Earth rotation
        omega_earth = np.array([0.0, 0.0, C.EARTH_ROTATION_RATE])
        v = np.cross(omega_earth, r)
        
        v_rel = compute_relative_velocity(r, v)
        np.testing.assert_array_almost_equal(v_rel, [0.0, 0.0, 0.0], decimal=6)
    
    def test_added_velocity(self):
        """Added velocity should appear as relative velocity."""
        r = np.array([C.R_EARTH, 0.0, 0.0])
        omega_earth = np.array([0.0, 0.0, C.EARTH_ROTATION_RATE])
        v_earth = np.cross(omega_earth, r)
        
        # Add 1000 m/s in X direction
        extra_v = np.array([1000.0, 0.0, 0.0])
        v = v_earth + extra_v
        
        v_rel = compute_relative_velocity(r, v)
        np.testing.assert_array_almost_equal(v_rel, extra_v, decimal=6)


if __name__ == '__main__':
    unittest.main()
