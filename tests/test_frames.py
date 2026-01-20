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


if __name__ == '__main__':
    unittest.main()
