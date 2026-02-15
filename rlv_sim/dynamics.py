"""
RLV Phase-I Ascent Simulation - Dynamics Equations

This module implements the equations of motion:
- Rotational dynamics: ω̇ = I⁻¹ (τ − ω × (Iω))
- Translational dynamics: r̈ = F_total / m
- Quaternion kinematics: q̇ = 0.5 * Ω(ω) * q
"""

from typing import NamedTuple

import numpy as np


from . import constants as C
from .frames import quaternion_derivative, quaternion_normalize
from .forces import compute_aerodynamic_moment, compute_gravity_force, compute_drag_force, compute_thrust_force, compute_lift_force
from .mass import compute_mass_derivative, compute_inertia_tensor, compute_center_of_mass


class StateDerivative(NamedTuple):
    """Container for state derivatives."""
    r_dot: np.ndarray
    v_dot: np.ndarray
    q_dot: np.ndarray
    omega_dot: np.ndarray
    m_dot: float


def compute_angular_acceleration(omega: np.ndarray, torque: np.ndarray, 
                                 I_tensor: np.ndarray, I_inv: np.ndarray) -> np.ndarray:
    """
    Compute angular acceleration from Euler's equation with VARIABLE INERTIA.
    
    ω̇ = I⁻¹ (τ − ω × (Iω))
    
    Args:
        omega: Angular velocity in body frame (rad/s)
        torque: Total torque (Control + Aero) in body frame (N*m)
        I_tensor: Current Inertia Tensor (kg*m^2)
        I_inv: Inverse Inertia Tensor
        
    Returns:
        Angular acceleration in body frame (rad/s²)
    """
    # Euler's equation: I * ω̇ + ω × (I * ω) = τ
    # ω̇ = I⁻¹ (τ − ω × (I * ω))
    I_omega = I_tensor @ omega
    gyroscopic = np.cross(omega, I_omega)
    return I_inv @ (torque - gyroscopic)


def compute_linear_acceleration(r: np.ndarray, v: np.ndarray, q: np.ndarray,
                                m: float, thrust_on: bool = True, throttle: float = 1.0,
                                stage: int = 1, vehicle_model: str = "stacked") -> np.ndarray:
    """
    Compute linear acceleration from Newton's second law (ECI frame).

    r̈ = (F_grav + F_thrust + F_drag + F_lift) / m

    No Coriolis force: ECI is an inertial frame. Earth rotation effects on
    aerodynamic forces are captured via air-relative velocity (v - omega_E x r).
    """
    F_total = (
        compute_gravity_force(r, m) +
        compute_thrust_force(q, r, thrust_on, throttle, stage=stage)
    )

    if stage != 2:
        F_drag = compute_drag_force(r, v)
        F_lift = compute_lift_force(r, v, q)
        if vehicle_model == "booster":
            # Recovery booster deploys drag devices (grid fins/body flap/legs).
            # Model that here as additional effective drag area during descent.
            F_drag = 6.0 * F_drag
        F_total = F_total + F_drag + F_lift

    if m < C.ZERO_TOLERANCE:
        return np.zeros(3)

    return F_total / m


def compute_state_derivative(r: np.ndarray, v: np.ndarray, q: np.ndarray,
                             omega: np.ndarray, m: float, torque: np.ndarray,
                             thrust_on: bool = True, throttle: float = 1.0,
                             dry_mass: float = None, stage: int = 1,
                             vehicle_model: str = "stacked") -> StateDerivative:
    """
    Compute all state derivatives for the full dynamics.

    Includes:
    - Variable Inertia
    - Aerodynamic Moments (Instability)
    - Throttling

    Args:
        dry_mass: Dry mass limit for propellant exhaustion check.
                  Defaults to C.DRY_MASS (stacked vehicle).
                  Use C.STAGE1_DRY_MASS for booster recovery.
        stage: Engine stage (1 = S1 engines, 2 = S2 engine)
    """
    # 1. Update Inertia Properties
    I_tensor = compute_inertia_tensor(m, vehicle_model=vehicle_model)
    # Use proper matrix inverse (handles both diagonal and full tensors)
    try:
        I_inv = np.linalg.inv(I_tensor)
    except np.linalg.LinAlgError:
        # Fallback to diagonal inverse if matrix is singular
        I_inv = np.diag(1.0 / np.diag(I_tensor))

    # 2. Compute Aerodynamic Instability Torque
    cg_pos_z = compute_center_of_mass(m, vehicle_model=vehicle_model)
    tau_aero = compute_aerodynamic_moment(r, v, q, cg_pos_z, cp_pos_z=C.H_CP)

    # 3. Compute Accelerations
    omega_dot = compute_angular_acceleration(omega, torque + tau_aero, I_tensor, I_inv)
    r_dot = v
    v_dot = compute_linear_acceleration(
        r, v, q, m, thrust_on, throttle, stage=stage, vehicle_model=vehicle_model
    )
    q_dot = quaternion_derivative(q, omega)
    m_dot = compute_mass_derivative(m, thrust_on, throttle, dry_mass=dry_mass, stage=stage)

    return StateDerivative(r_dot, v_dot, q_dot, omega_dot, m_dot)


def state_derivative_vector(state_vec: np.ndarray, t: float,
                           torque: np.ndarray, thrust_on: bool = True, throttle: float = 1.0,
                           dry_mass: float = None, stage: int = 1,
                           vehicle_model: str = "stacked") -> np.ndarray:
    """
    Compute state derivative as a flat vector for numerical integration.
    """
    # Unpack state
    r = state_vec[0:3]
    v = state_vec[3:6]
    q = state_vec[6:10]
    omega = state_vec[10:13]
    m = state_vec[13]

    # Normalize quaternion (safety)
    q = quaternion_normalize(q)

    # Compute derivatives
    derivs = compute_state_derivative(r, v, q, omega, m, torque, thrust_on, throttle,
                                      dry_mass=dry_mass, stage=stage,
                                      vehicle_model=vehicle_model)

    # Pack into vector
    return np.concatenate([
        derivs.r_dot,
        derivs.v_dot,
        derivs.q_dot,
        derivs.omega_dot,
        [derivs.m_dot]
    ])




