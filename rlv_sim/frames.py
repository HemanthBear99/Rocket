"""
RLV Phase-I Ascent Simulation - Reference Frame Transformations

This module implements quaternion operations and reference frame transformations.
All orientation representation uses quaternions exclusively (no Euler angles).

Quaternion Convention: [w, x, y, z] where w is the scalar component.
"""

import numpy as np

from . import constants as C


def quaternion_normalize(q: np.ndarray) -> np.ndarray:
    """
    Normalize a quaternion to unit length.
    
    Args:
        q: Quaternion [w, x, y, z]
        
    Returns:
        Normalized quaternion
    """
    norm = np.linalg.norm(q)
    if norm < C.ZERO_TOLERANCE:
        # Return identity quaternion if input is degenerate
        return np.array([1.0, 0.0, 0.0, 0.0])
    return q / norm


def quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """
    Multiply two quaternions: q1 * q2
    
    Args:
        q1: First quaternion [w, x, y, z]
        q2: Second quaternion [w, x, y, z]
        
    Returns:
        Product quaternion [w, x, y, z]
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])


def quaternion_conjugate(q: np.ndarray) -> np.ndarray:
    """
    Compute the conjugate of a quaternion.
    
    Args:
        q: Quaternion [w, x, y, z]
        
    Returns:
        Conjugate quaternion [w, -x, -y, -z]
    """
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quaternion_inverse(q: np.ndarray) -> np.ndarray:
    """
    Compute the inverse of a quaternion.
    For unit quaternions, inverse equals conjugate.
    
    Args:
        q: Quaternion [w, x, y, z]
        
    Returns:
        Inverse quaternion
    """
    conj = quaternion_conjugate(q)
    norm_sq = np.dot(q, q)
    if norm_sq < C.ZERO_TOLERANCE:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return conj / norm_sq


def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """
    Convert a quaternion to a rotation matrix R(q).
    
    The rotation matrix transforms vectors from body frame to inertial frame:
    v_inertial = R(q) @ v_body
    
    Args:
        q: Unit quaternion [w, x, y, z]
        
    Returns:
        3x3 rotation matrix
    """
    w, x, y, z = quaternion_normalize(q)
    
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z
    
    return np.array([
        [1 - 2*(yy + zz),     2*(xy - wz),     2*(xz + wy)],
        [    2*(xy + wz), 1 - 2*(xx + zz),     2*(yz - wx)],
        [    2*(xz - wy),     2*(yz + wx), 1 - 2*(xx + yy)]
    ])


def rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """
    Convert a rotation matrix to a quaternion.
    Uses Shepperd's method for numerical stability.
    
    Args:
        R: 3x3 rotation matrix
        
    Returns:
        Quaternion [w, x, y, z]
    """
    trace = np.trace(R)
    
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    
    q = np.array([w, x, y, z])
    return quaternion_normalize(q)


def omega_matrix(omega: np.ndarray) -> np.ndarray:
    """
    Construct the Omega matrix for quaternion kinematics.
    
    The quaternion derivative is: q_dot = 0.5 * Omega(omega) @ q
    
    Args:
        omega: Angular velocity in body frame [wx, wy, wz] (rad/s)
        
    Returns:
        4x4 Omega matrix
    """
    wx, wy, wz = omega
    
    return np.array([
        [0.0, -wx, -wy, -wz],
        [wx,  0.0,  wz, -wy],
        [wy, -wz,  0.0,  wx],
        [wz,  wy, -wx,  0.0]
    ])


def quaternion_derivative(q: np.ndarray, omega: np.ndarray) -> np.ndarray:
    """
    Compute the quaternion time derivative.
    
    q_dot = 0.5 * Omega(omega) @ q
    
    Args:
        q: Current quaternion [w, x, y, z]
        omega: Angular velocity in body frame (rad/s)
        
    Returns:
        Quaternion derivative [w_dot, x_dot, y_dot, z_dot]
    """
    Omega = omega_matrix(omega)
    return 0.5 * Omega @ q


def rotate_vector_by_quaternion(v: np.ndarray, q: np.ndarray) -> np.ndarray:
    """
    Rotate a vector by a quaternion.
    
    Transforms v from body frame to inertial frame.
    
    Args:
        v: Vector to rotate [3]
        q: Quaternion [w, x, y, z]
        
    Returns:
        Rotated vector [3]
    """
    R = quaternion_to_rotation_matrix(q)
    return R @ v


def rotate_vector_inverse(v: np.ndarray, q: np.ndarray) -> np.ndarray:
    """
    Rotate a vector by the inverse of a quaternion.
    
    Transforms v from inertial frame to body frame.
    
    Args:
        v: Vector to rotate [3]
        q: Quaternion [w, x, y, z]
        
    Returns:
        Rotated vector [3]
    """
    R = quaternion_to_rotation_matrix(q)
    return R.T @ v


def direction_to_quaternion(direction: np.ndarray, 
                            reference: np.ndarray = None) -> np.ndarray:
    """
    Compute a quaternion that rotates the reference direction to the target direction.
    
    Args:
        direction: Target direction in inertial frame (will be normalized)
        reference: Reference direction in body frame (default: +Z = [0, 0, 1])
        
    Returns:
        Quaternion [w, x, y, z] that transforms body reference to inertial target
    """
    # Use default reference if not provided (avoid mutable default argument)
    if reference is None:
        reference = np.array([0.0, 0.0, 1.0])
    
    # Normalize inputs
    d = direction / np.linalg.norm(direction)
    r = reference / np.linalg.norm(reference)
    
    # Compute rotation axis and angle
    dot = np.clip(np.dot(r, d), -1.0, 1.0)
    
    if dot > 0.9999:
        # Vectors are parallel
        return np.array([1.0, 0.0, 0.0, 0.0])
    elif dot < -0.9999:
        # Vectors are anti-parallel, rotate 180 degrees around any perpendicular axis
        perp = np.array([1, 0, 0]) if abs(r[0]) < 0.9 else np.array([0, 1, 0])
        axis = np.cross(r, perp)
        axis = axis / np.linalg.norm(axis)
        return np.array([0.0, axis[0], axis[1], axis[2]])
    
    # General case
    axis = np.cross(r, d)
    axis = axis / np.linalg.norm(axis)
    angle = np.arccos(dot)
    
    # Quaternion from axis-angle
    half_angle = angle / 2.0
    w = np.cos(half_angle)
    xyz = axis * np.sin(half_angle)
    
    return np.array([w, xyz[0], xyz[1], xyz[2]])


def quaternion_error(q_current: np.ndarray, q_desired: np.ndarray) -> np.ndarray:
    """
    Compute the quaternion error between current and desired orientations.
    
    q_error = q_desired * q_current^(-1)
    
    The error quaternion represents the rotation needed to go from
    current orientation to desired orientation.
    
    Args:
        q_current: Current quaternion [w, x, y, z]
        q_desired: Desired quaternion [w, x, y, z]
        
    Returns:
        Error quaternion [w, x, y, z]
    """
    q_inv = quaternion_inverse(q_current)
    # Compute error in BODY frame: q_err = q_inv * q_desired
    q_err = quaternion_multiply(q_inv, q_desired)
    
    # Ensure we take the shortest path (w >= 0)
    if q_err[0] < 0:
        q_err = -q_err
    
    return q_err


def quaternion_to_axis_angle(q: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Convert a quaternion to axis-angle representation.
    
    Args:
        q: Quaternion [w, x, y, z]
        
    Returns:
        (axis, angle) tuple where axis is unit vector and angle is in radians
    """
    q = quaternion_normalize(q)
    
    # Ensure w is positive for shortest path
    if q[0] < 0:
        q = -q
    
    w = np.clip(q[0], -1.0, 1.0)
    angle = 2.0 * np.arccos(w)
    
    sin_half = np.sqrt(1.0 - w*w)
    if sin_half < C.ZERO_TOLERANCE:
        # Small angle, axis is arbitrary
        return np.array([0.0, 0.0, 1.0]), 0.0
    
    axis = q[1:4] / sin_half
    return axis, angle
