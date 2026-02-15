"""
RLV Phase-I Ascent Simulation - Attitude Control

This module implements the attitude control system:
- Direction to quaternion conversion
- Quaternion error computation
- PD control law
- Torque saturation
"""

import numpy as np

from . import constants as C
from .types import ControlOutput
from .frames import (
    direction_to_quaternion,
    quaternion_error,
    quaternion_to_axis_angle,
    quaternion_normalize,
    quaternion_inverse
)


def compute_commanded_quaternion(desired_direction: np.ndarray) -> np.ndarray:
    """
    Convert desired thrust direction to commanded orientation quaternion.
    
    The body +Z axis should align with the desired thrust direction.
    
    Args:
        desired_direction: Desired thrust direction (inertial frame)
        
    Returns:
        Commanded quaternion [w, x, y, z]
    """
    # Body +Z must point along desired direction  
    # Need quaternion that rotates body frame so +Z aligns with inertial target
    return direction_to_quaternion(desired_direction, C.BODY_Z_AXIS)


def compute_attitude_error(q_current: np.ndarray,
                          q_commanded: np.ndarray) -> tuple:
    """
    Compute the attitude error between current and commanded orientation.

    Returns the error quaternion vector part (for PD control per Doc §17.6)
    and the error angle (for logging/diagnostics).

    Per Document Section 17.3:
        q_e = q_cmd ⊗ q⁻¹

    The vector part q_ev is used directly as the proportional error signal.
    The scalar part q_e0 ≈ 1 for small errors, and q_ev ≈ (θ/2)·axis.

    Args:
        q_current: Current orientation quaternion
        q_commanded: Commanded orientation quaternion

    Returns:
        (q_error_vector, error_angle) where q_error_vector is [q_e1, q_e2, q_e3]
    """
    q_current = quaternion_normalize(q_current)
    q_commanded = quaternion_normalize(q_commanded)

    q_err = quaternion_error(q_current, q_commanded)

    # Extract vector part for PD control (Document §17.4, §17.6)
    q_error_vector = q_err[1:4]

    # Also compute axis-angle for logging/diagnostics
    axis, angle = quaternion_to_axis_angle(q_err)

    return q_error_vector, angle


def saturate_torque(torque: np.ndarray) -> np.ndarray:
    """
    Apply torque saturation limits.
    
    Args:
        torque: Commanded torque vector (N*m)
        
    Returns:
        Saturated torque vector (N*m)
    """
    torque_magnitude = np.linalg.norm(torque)
    
    if torque_magnitude > C.MAX_TORQUE:
        return torque * (C.MAX_TORQUE / torque_magnitude)
    
    return torque


def limit_gimbal_rate(omega: np.ndarray) -> np.ndarray:
    """
    Apply gimbal rate limits to angular velocity.
    
    Constrains angular velocity components to MAX_GIMBAL_RATE to respect
    physical engine gimbal actuator capabilities.
    
    Args:
        omega: Current angular velocity in body frame (rad/s)
        
    Returns:
        Rate-limited angular velocity in body frame (rad/s)
    """
    # Limit each component to MAX_GIMBAL_RATE
    # For a rocket, typically limit pitch and roll, while yaw is less critical
    omega_limited = np.array([
        np.clip(omega[0], -C.MAX_GIMBAL_RATE, C.MAX_GIMBAL_RATE),  # Roll
        np.clip(omega[1], -C.MAX_GIMBAL_RATE, C.MAX_GIMBAL_RATE),  # Pitch
        np.clip(omega[2], -C.MAX_GIMBAL_RATE, C.MAX_GIMBAL_RATE)   # Yaw
    ])
    return omega_limited


def pd_control_law(q_error_vector: np.ndarray, error_angle: float,
                   omega: np.ndarray, inertia: float = None,
                   max_torque: float = None) -> np.ndarray:
    """
    Implement PD attitude control law per Document Section 17.6.

    τ_cmd = -Kp · q_ev - Kd · ω_e

    where q_ev is the vector part of the error quaternion and ω_e is
    the angular velocity error (= ω since ω_cmd = 0 in Phase I).

    Gain scheduling: When the vehicle inertia is provided, gains are
    scaled to maintain the designed natural frequency (ωn ≈ 1.06 rad/s)
    and damping ratio (ζ ≈ 0.7) regardless of vehicle mass:
        Kp_eff = Kp_ref * (I / I_ref)
        Kd_eff = Kd_ref * (I / I_ref)

    This ensures consistent control response across S1 (I=5.36e7 kg·m²)
    and S2 (I=0.6e6–4.0e6 kg·m²) without over-torquing or sluggishness.

    Args:
        q_error_vector: Vector part of error quaternion [q_e1, q_e2, q_e3]
        error_angle: Magnitude of rotation error (rad) — for logging only
        omega: Current angular velocity in body frame (rad/s)
        inertia: Representative moment of inertia (kg·m²) for gain scheduling.
                 If None, uses the reference (S1 full) gains directly.
        max_torque: Torque saturation limit (N·m). If None, uses C.MAX_TORQUE.

    Returns:
        Control torque in body frame (N*m)
    """
    # Apply gimbal rate limits to angular velocity
    omega_limited = limit_gimbal_rate(omega)

    # Gain scheduling: scale gains proportionally to inertia
    # Reference design point: I_ref = IXX_FULL = 5.36e7 (S1 at launch)
    # Kp_ref = 1.2e8, Kd_ref = 7.94e7 → ωn = 1.058 rad/s, ζ = 0.7
    if inertia is not None and inertia > 0:
        gain_ratio = inertia / C.IXX_FULL
        kp = C.KP_ATTITUDE * gain_ratio
        kd = C.KD_ATTITUDE * gain_ratio
    else:
        kp = C.KP_ATTITUDE
        kd = C.KD_ATTITUDE

    if max_torque is None:
        max_torque = C.MAX_TORQUE

    # Proportional term: Kp * q_ev  (Document Section 17.6 adapted to body frame)
    tau_p = kp * q_error_vector

    # Derivative term: -Kd * ω_e  (where ω_e = ω since ω_cmd = 0)
    tau_d = -kd * omega_limited

    torque = tau_p + tau_d

    # Saturate torque
    torque_magnitude = np.linalg.norm(torque)
    if torque_magnitude > max_torque:
        torque = torque * (max_torque / torque_magnitude)

    return torque


def compute_control_torque(q_current: np.ndarray, omega: np.ndarray,
                          desired_direction: np.ndarray,
                          inertia: float = None,
                          max_torque: float = None) -> np.ndarray:
    """
    Compute the control torque to track desired thrust direction.

    Implements the Document Section 17.10 closed-loop sequence:
    1. Compute guidance command q_cmd
    2. Compute quaternion error q_e
    3. Compute angular velocity error omega_e
    4. Compute control torque tau_cmd
    5. Apply torque in rotational dynamics

    Args:
        q_current: Current orientation quaternion [w, x, y, z]
        omega: Current angular velocity in body frame (rad/s)
        desired_direction: Desired thrust direction (inertial frame)
        inertia: Representative moment of inertia for gain scheduling (kg*m^2)
        max_torque: Torque saturation limit (N*m)

    Returns:
        Control torque in body frame (N*m)
    """
    q_commanded = compute_commanded_quaternion(desired_direction)
    q_error_vector, error_angle = compute_attitude_error(q_current, q_commanded)
    return pd_control_law(q_error_vector, error_angle, omega,
                         inertia=inertia, max_torque=max_torque)


def compute_control_output(q_current: np.ndarray, omega: np.ndarray,
                          desired_direction: np.ndarray,
                          inertia: float = None,
                          max_torque: float = None) -> ControlOutput:
    """
    Compute full control output for logging and analysis.

    Args:
        q_current: Current orientation quaternion [w, x, y, z]
        omega: Current angular velocity in body frame (rad/s)
        desired_direction: Desired thrust direction (inertial frame)
        inertia: Representative moment of inertia for gain scheduling (kg*m^2)
        max_torque: Torque saturation limit (N*m)

    Returns:
        Dictionary containing control state and commands
    """
    if max_torque is None:
        max_torque = C.MAX_TORQUE

    q_commanded = compute_commanded_quaternion(desired_direction)
    q_error_vector, error_angle = compute_attitude_error(q_current, q_commanded)
    torque = pd_control_law(q_error_vector, error_angle, omega,
                           inertia=inertia, max_torque=max_torque)

    return {
        'q_commanded': q_commanded,
        'error_axis': q_error_vector,
        'error_angle': error_angle,
        'error_degrees': np.degrees(error_angle),
        'torque': torque,
        'torque_magnitude': np.linalg.norm(torque),
        'saturated': np.linalg.norm(torque) >= max_torque * 0.999
    }
