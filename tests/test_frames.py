"""Tests for reference frame transformations and quaternion operations."""

import unittest

import numpy as np

from rlv_sim import frames


class TestQuaternionNormalize(unittest.TestCase):
    """Tests for quaternion normalization."""
    
    def test_already_normalized(self):
        """Normalized quaternion should stay the same."""
        q = np.array([1.0, 0.0, 0.0, 0.0])
        result = frames.quaternion_normalize(q)
        np.testing.assert_array_almost_equal(result, q)
    
    def test_unnormalized(self):
        """Unnormalized quaternion should be normalized."""
        q = np.array([2.0, 0.0, 0.0, 0.0])
        result = frames.quaternion_normalize(q)
        np.testing.assert_array_almost_equal(result, [1.0, 0.0, 0.0, 0.0])
    
    def test_general_quaternion(self):
        """General quaternion should have unit norm after normalization."""
        q = np.array([1.0, 2.0, 3.0, 4.0])
        result = frames.quaternion_normalize(q)
        self.assertAlmostEqual(np.linalg.norm(result), 1.0, places=10)
    
    def test_near_zero_returns_identity(self):
        """Near-zero quaternion should return identity."""
        q = np.array([1e-15, 1e-15, 1e-15, 1e-15])
        result = frames.quaternion_normalize(q)
        np.testing.assert_array_almost_equal(result, [1.0, 0.0, 0.0, 0.0])


class TestQuaternionMultiply(unittest.TestCase):
    """Tests for quaternion multiplication."""
    
    def test_identity_left(self):
        """Multiplying by identity on left should give same quaternion."""
        identity = np.array([1.0, 0.0, 0.0, 0.0])
        q = np.array([0.707, 0.707, 0.0, 0.0])
        result = frames.quaternion_multiply(identity, q)
        np.testing.assert_array_almost_equal(result, q)
    
    def test_identity_right(self):
        """Multiplying by identity on right should give same quaternion."""
        identity = np.array([1.0, 0.0, 0.0, 0.0])
        q = np.array([0.707, 0.707, 0.0, 0.0])
        result = frames.quaternion_multiply(q, identity)
        np.testing.assert_array_almost_equal(result, q)
    
    def test_double_rotation(self):
        """90° rotation twice should give 180° rotation."""
        # 90° rotation about Z axis
        q_90z = np.array([np.cos(np.pi/4), 0.0, 0.0, np.sin(np.pi/4)])
        result = frames.quaternion_multiply(q_90z, q_90z)
        # Should be 180° about Z
        expected = np.array([0.0, 0.0, 0.0, 1.0])
        np.testing.assert_array_almost_equal(result, expected, decimal=6)


class TestQuaternionConjugate(unittest.TestCase):
    """Tests for quaternion conjugate."""
    
    def test_conjugate(self):
        """Conjugate should negate vector part."""
        q = np.array([1.0, 2.0, 3.0, 4.0])
        result = frames.quaternion_conjugate(q)
        expected = np.array([1.0, -2.0, -3.0, -4.0])
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_identity_conjugate(self):
        """Identity quaternion conjugate should be identity."""
        identity = np.array([1.0, 0.0, 0.0, 0.0])
        result = frames.quaternion_conjugate(identity)
        np.testing.assert_array_almost_equal(result, identity)


class TestQuaternionInverse(unittest.TestCase):
    """Tests for quaternion inverse."""
    
    def test_unit_quaternion_inverse_equals_conjugate(self):
        """For unit quaternion, inverse should equal conjugate."""
        q = frames.quaternion_normalize(np.array([1.0, 2.0, 3.0, 4.0]))
        inverse = frames.quaternion_inverse(q)
        conjugate = frames.quaternion_conjugate(q)
        np.testing.assert_array_almost_equal(inverse, conjugate)
    
    def test_multiply_by_inverse_gives_identity(self):
        """q * q^(-1) should give identity."""
        q = frames.quaternion_normalize(np.array([0.5, 0.5, 0.5, 0.5]))
        q_inv = frames.quaternion_inverse(q)
        result = frames.quaternion_multiply(q, q_inv)
        identity = np.array([1.0, 0.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(result, identity, decimal=6)

    def test_near_zero_quaternion_inverse(self):
        """Near-zero quaternion should return identity."""
        q = np.array([1e-15, 1e-15, 1e-15, 1e-15])
        result = frames.quaternion_inverse(q)
        np.testing.assert_array_almost_equal(result, [1.0, 0.0, 0.0, 0.0])


class TestRotationMatrixConversions(unittest.TestCase):
    """Tests for rotation matrix <-> quaternion conversions."""
    
    def test_identity_quaternion_to_identity_matrix(self):
        """Identity quaternion should give identity rotation matrix."""
        q = np.array([1.0, 0.0, 0.0, 0.0])
        R = frames.quaternion_to_rotation_matrix(q)
        np.testing.assert_array_almost_equal(R, np.eye(3))
    
    def test_roundtrip_quaternion_matrix_quaternion(self):
        """q -> R -> q should give same quaternion (up to sign)."""
        q_original = frames.quaternion_normalize(np.array([0.5, 0.5, 0.5, 0.5]))
        R = frames.quaternion_to_rotation_matrix(q_original)
        q_recovered = frames.rotation_matrix_to_quaternion(R)
        
        # Quaternions q and -q represent the same rotation
        if np.dot(q_original, q_recovered) < 0:
            q_recovered = -q_recovered
        np.testing.assert_array_almost_equal(q_original, q_recovered, decimal=6)
    
    def test_rotation_matrix_is_orthogonal(self):
        """Rotation matrix should be orthogonal (R^T R = I)."""
        q = np.array([0.707, 0.707, 0.0, 0.0])  # 90° about X
        R = frames.quaternion_to_rotation_matrix(q)
        np.testing.assert_array_almost_equal(R.T @ R, np.eye(3), decimal=6)

    def test_rotation_matrix_branch_x_dominant(self):
        """Test rotation matrix conversion when R[0,0] is largest."""
        # Create a rotation that makes R[0,0] > R[1,1] and R[0,0] > R[2,2]
        R = np.array([[0.8, -0.5, 0.3],
                      [0.5, 0.4, -0.7],
                      [0.3, 0.7, 0.5]])
        # Orthogonalize for safety
        U, _, Vt = np.linalg.svd(R)
        R = U @ Vt
        q = frames.rotation_matrix_to_quaternion(R)
        self.assertAlmostEqual(np.linalg.norm(q), 1.0, places=6)

    def test_rotation_matrix_branch_y_dominant(self):
        """Test rotation matrix conversion when R[1,1] is largest."""
        # 180° about Y axis: R[1,1] = 1, others = -1
        R = np.array([[-1, 0, 0],
                      [0, 1, 0],
                      [0, 0, -1]], dtype=float)
        q = frames.rotation_matrix_to_quaternion(R)
        self.assertAlmostEqual(np.linalg.norm(q), 1.0, places=6)

    def test_rotation_matrix_branch_z_dominant(self):
        """Test rotation matrix conversion when R[2,2] is largest."""
        # 180° about Z axis: R[2,2] = 1, others = -1
        R = np.array([[-1, 0, 0],
                      [0, -1, 0],
                      [0, 0, 1]], dtype=float)
        q = frames.rotation_matrix_to_quaternion(R)
        self.assertAlmostEqual(np.linalg.norm(q), 1.0, places=6)


class TestDirectionToQuaternion(unittest.TestCase):
    """Tests for direction_to_quaternion function."""
    
    def test_reference_to_reference(self):
        """Rotating reference to itself should give identity."""
        reference = np.array([0.0, 0.0, 1.0])
        q = frames.direction_to_quaternion(reference, reference)
        # Should be identity or close to it
        identity = np.array([1.0, 0.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(np.abs(q), np.abs(identity), decimal=6)
    
    def test_z_to_x(self):
        """Rotating Z to X should produce correct rotation."""
        z = np.array([0.0, 0.0, 1.0])
        x = np.array([1.0, 0.0, 0.0])
        q = frames.direction_to_quaternion(x, z)
        
        # Apply rotation to verify
        R = frames.quaternion_to_rotation_matrix(q)
        result = R @ z
        np.testing.assert_array_almost_equal(result, x, decimal=6)
    
    def test_opposite_directions(self):
        """Rotating to opposite direction (180°) should work."""
        z = np.array([0.0, 0.0, 1.0])
        neg_z = np.array([0.0, 0.0, -1.0])
        q = frames.direction_to_quaternion(neg_z, z)
        
        # Apply and verify
        R = frames.quaternion_to_rotation_matrix(q)
        result = R @ z
        np.testing.assert_array_almost_equal(result, neg_z, decimal=6)

    def test_default_reference(self):
        """Test with default reference direction."""
        target = np.array([1.0, 0.0, 0.0])
        q = frames.direction_to_quaternion(target)
        self.assertAlmostEqual(np.linalg.norm(q), 1.0, places=6)

    def test_anti_parallel_x_dominant(self):
        """Test anti-parallel case when reference X is large."""
        # Reference along X, target opposite
        ref = np.array([1.0, 0.0, 0.0])
        target = np.array([-1.0, 0.0, 0.0])
        q = frames.direction_to_quaternion(target, ref)
        self.assertAlmostEqual(np.linalg.norm(q), 1.0, places=6)


class TestAxisAngle(unittest.TestCase):
    """Tests for axis-angle conversion."""
    
    def test_identity_quaternion(self):
        """Identity quaternion should give zero angle."""
        identity = np.array([1.0, 0.0, 0.0, 0.0])
        axis, angle = frames.quaternion_to_axis_angle(identity)
        self.assertAlmostEqual(angle, 0.0, places=6)
    
    def test_90_degree_rotation(self):
        """90° rotation about Z should give correct axis and angle."""
        q = np.array([np.cos(np.pi/4), 0.0, 0.0, np.sin(np.pi/4)])
        axis, angle = frames.quaternion_to_axis_angle(q)
        
        np.testing.assert_array_almost_equal(axis, [0.0, 0.0, 1.0], decimal=6)
        self.assertAlmostEqual(angle, np.pi/2, places=6)

    def test_negative_w_quaternion(self):
        """Test that negative w is handled (shortest path)."""
        q = np.array([-np.cos(np.pi/4), 0.0, 0.0, -np.sin(np.pi/4)])
        axis, angle = frames.quaternion_to_axis_angle(q)
        # Should still give valid result
        self.assertGreaterEqual(angle, 0.0)
        self.assertLessEqual(angle, np.pi)


class TestOmegaMatrix(unittest.TestCase):
    """Tests for omega matrix construction."""
    
    def test_zero_omega(self):
        """Zero angular velocity should give zero matrix."""
        omega = np.zeros(3)
        Omega = frames.omega_matrix(omega)
        np.testing.assert_array_almost_equal(Omega, np.zeros((4, 4)))
    
    def test_omega_antisymmetric(self):
        """Omega matrix should be antisymmetric."""
        omega = np.array([1.0, 2.0, 3.0])
        Omega = frames.omega_matrix(omega)
        np.testing.assert_array_almost_equal(Omega, -Omega.T)


class TestQuaternionDerivative(unittest.TestCase):
    """Tests for quaternion derivative."""
    
    def test_zero_omega_zero_derivative(self):
        """Zero angular velocity should give zero derivative."""
        q = np.array([1.0, 0.0, 0.0, 0.0])
        omega = np.zeros(3)
        q_dot = frames.quaternion_derivative(q, omega)
        np.testing.assert_array_almost_equal(q_dot, np.zeros(4))
    
    def test_derivative_preserves_norm(self):
        """For unit quaternion, derivative should be perpendicular."""
        q = np.array([1.0, 0.0, 0.0, 0.0])
        omega = np.array([0.1, 0.0, 0.0])
        q_dot = frames.quaternion_derivative(q, omega)
        # q_dot should be perpendicular to q (dot product ≈ 0)
        self.assertAlmostEqual(np.dot(q, q_dot), 0.0, places=10)


class TestRotateVector(unittest.TestCase):
    """Tests for vector rotation functions."""
    
    def test_identity_rotation(self):
        """Identity quaternion should not change vector."""
        v = np.array([1.0, 2.0, 3.0])
        q = np.array([1.0, 0.0, 0.0, 0.0])
        result = frames.rotate_vector_by_quaternion(v, q)
        np.testing.assert_array_almost_equal(result, v)
    
    def test_90_degree_rotation(self):
        """90° about Z should rotate X to Y."""
        v = np.array([1.0, 0.0, 0.0])
        q = np.array([np.cos(np.pi/4), 0.0, 0.0, np.sin(np.pi/4)])
        result = frames.rotate_vector_by_quaternion(v, q)
        expected = np.array([0.0, 1.0, 0.0])
        np.testing.assert_array_almost_equal(result, expected, decimal=6)
    
    def test_inverse_rotation(self):
        """Inverse rotation should undo forward rotation."""
        v = np.array([1.0, 2.0, 3.0])
        q = frames.quaternion_normalize(np.array([0.5, 0.5, 0.5, 0.5]))
        v_rotated = frames.rotate_vector_by_quaternion(v, q)
        v_back = frames.rotate_vector_inverse(v_rotated, q)
        np.testing.assert_array_almost_equal(v_back, v, decimal=6)


class TestQuaternionError(unittest.TestCase):
    """Tests for quaternion error computation."""
    
    def test_same_quaternion_zero_error(self):
        """Same current and desired should give identity error."""
        q = frames.quaternion_normalize(np.array([0.5, 0.5, 0.5, 0.5]))
        q_err = frames.quaternion_error(q, q)
        identity = np.array([1.0, 0.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(q_err, identity, decimal=6)
    
    def test_error_shortest_path(self):
        """Error quaternion w should be >= 0 (shortest path)."""
        q_current = np.array([1.0, 0.0, 0.0, 0.0])
        q_desired = np.array([-0.707, 0.707, 0.0, 0.0])
        q_err = frames.quaternion_error(q_current, q_desired)
        self.assertGreaterEqual(q_err[0], 0.0)


if __name__ == '__main__':
    unittest.main()

