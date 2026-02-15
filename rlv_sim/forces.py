"""
RLV Phase-I Ascent Simulation - Force Computations

This module implements all force calculations:
- Central gravity
- Thrust (in body frame, transformed to inertial)
- Atmospheric drag
"""

import numpy as np

from . import constants as C
from .frames import quaternion_to_rotation_matrix
from .utils import compute_relative_velocity
from .types import ForceBreakdown


# US-76 layer boundaries and lapse rates (dT/dh in K/m, geometric altitude)
_US76_H = np.array([0.0, 11000.0, 20000.0, 32000.0, 47000.0, 51000.0, 71000.0, 84852.0])
_US76_L = np.array([-0.0065, 0.0, 0.0010, 0.0028, 0.0, -0.0028, -0.0020])
_US76_TB = None
_US76_PB = None


def _build_us76_tables():
    """Precompute layer-base temperatures and pressures for US-76."""
    global _US76_TB, _US76_PB
    if _US76_TB is not None and _US76_PB is not None:
        return

    tb = [C.ATM_T0]
    pb = [C.ATM_P0]
    for i, lapse in enumerate(_US76_L):
        h0 = _US76_H[i]
        h1 = _US76_H[i + 1]
        T0 = tb[-1]
        P0 = pb[-1]
        dh = h1 - h0
        if abs(lapse) > 1e-12:
            T1 = T0 + lapse * dh
            exponent = -C.G0 / (lapse * C.R_GAS)
            P1 = P0 * (T1 / T0) ** exponent
        else:
            T1 = T0
            P1 = P0 * np.exp(-C.G0 * dh / (C.R_GAS * T0))
        tb.append(float(T1))
        pb.append(float(P1))

    _US76_TB = np.array(tb)
    _US76_PB = np.array(pb)


# =============================================================================
# ATMOSPHERE MODEL (US Standard Atmosphere 1976 - Simplified)
# =============================================================================

def compute_atmosphere_properties(altitude: float) -> tuple:
    """
    Compute atmospheric properties (Temperature, Pressure, Density, Speed of Sound).
    Based on US Standard Atmosphere 1976 (Troposphere & Stratosphere).
    
    Args:
        altitude: Geometric altitude above sea level (m)
        
    Returns:
        (temperature, pressure, density, speed_of_sound)
        T in K, P in Pa, rho in kg/m^3, a in m/s
    """
    _build_us76_tables()
    h = max(0.0, float(altitude))

    if h <= _US76_H[-1]:
        idx = int(np.searchsorted(_US76_H, h, side='right') - 1)
        idx = max(0, min(idx, len(_US76_L) - 1))
        h0 = _US76_H[idx]
        lapse = _US76_L[idx]
        T0 = _US76_TB[idx]
        P0 = _US76_PB[idx]
        dh = h - h0

        if abs(lapse) > 1e-12:
            T = T0 + lapse * dh
            exponent = -C.G0 / (lapse * C.R_GAS)
            P = P0 * (T / T0) ** exponent
        else:
            T = T0
            P = P0 * np.exp(-C.G0 * dh / (C.R_GAS * T0))
    else:
        # Continuous extension above 84.852 km with aggressive pressure decay.
        h0 = _US76_H[-1]
        T0 = _US76_TB[-1]
        P0 = _US76_PB[-1]
        scale_height = 5000.0
        T = T0
        P = P0 * np.exp(-(h - h0) / scale_height)

    rho = P / (C.R_GAS * T) if (T > 0.0 and P > 0.0) else 0.0
    if rho < C.DENSITY_FLOOR:
        rho = 0.0
    speed_of_sound = np.sqrt(C.GAMMA * C.R_GAS * T) if T > 0.0 else C.ATM_SPEED_OF_SOUND_FALLBACK
    return float(T), float(P), float(rho), float(speed_of_sound)



# =============================================================================
# FORCE MODELS
# =============================================================================

def compute_gravity_force(r: np.ndarray, m: float, enable_j2: bool = False,
                          j2: float = 1.08263e-3) -> np.ndarray:
    """
    Compute gravitational force with optional J2 oblateness perturbation.

    Central gravity:
        F = -mu * m * r / ||r||^3

    J2 perturbation adds zonal harmonic correction:
        a_J2 = (3/2) * J2 * mu * R_E^2 / r^5 * [x*(5*z^2/r^2 - 1),
                                                    y*(5*z^2/r^2 - 1),
                                                    z*(5*z^2/r^2 - 3)]

    Args:
        r: Position vector (ECI, m)
        m: Vehicle mass (kg)
        enable_j2: If True, include J2 oblateness correction
        j2: J2 coefficient (default Earth J2 = 1.08263e-3)

    Returns:
        Gravitational force vector (N) in ECI frame
    """
    r_norm = np.linalg.norm(r)
    if r_norm < C.ZERO_TOLERANCE:
        return np.zeros(3)

    # Central gravity
    F_central = -C.MU_EARTH * m * r / (r_norm ** 3)

    if not enable_j2:
        return F_central

    # J2 perturbation (acceleration, then multiply by mass)
    x, y, z = r
    r2 = r_norm ** 2
    r5 = r_norm ** 5
    factor = 1.5 * j2 * C.MU_EARTH * C.R_EARTH ** 2 / r5
    z2_over_r2 = z ** 2 / r2

    a_j2 = factor * np.array([
        x * (5.0 * z2_over_r2 - 1.0),
        y * (5.0 * z2_over_r2 - 1.0),
        z * (5.0 * z2_over_r2 - 3.0)
    ])

    return F_central + m * a_j2


def apply_engine_transient(throttle_cmd: float, throttle_prev: float, dt: float,
                           spool_up_time: float = 1.5,
                           spool_down_time: float = 0.8) -> float:
    """
    Apply engine spool-up / spool-down rate limiting to throttle command.

    Models the finite response time of turbopump-fed rocket engines.
    Spool-up is typically slower than spool-down (emergency cutoff is faster).

    Args:
        throttle_cmd: Desired throttle (0.0 - 1.0)
        throttle_prev: Previous actual throttle (0.0 - 1.0)
        dt: Time step (s)
        spool_up_time: Time to go from 0% to 100% (s)
        spool_down_time: Time to go from 100% to 0% (s)

    Returns:
        Rate-limited actual throttle (0.0 - 1.0)
    """
    delta = throttle_cmd - throttle_prev
    if delta > 0:
        # Spool up
        max_rate = 1.0 / max(spool_up_time, 0.01)
        max_delta = max_rate * dt
        actual_delta = min(delta, max_delta)
    else:
        # Spool down
        max_rate = 1.0 / max(spool_down_time, 0.01)
        max_delta = max_rate * dt
        actual_delta = max(delta, -max_delta)

    return float(np.clip(throttle_prev + actual_delta, 0.0, 1.0))


def compute_drag_force(r: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Compute atmospheric drag force with Mach-dependent Cd.
    
    F_drag = -0.5 * ρ * Cd(Mach) * A * ||v_rel||² * v_rel_hat
    
    Uses Earth rotation for relative velocity (wind).
    """
    altitude = np.linalg.norm(r) - C.R_EARTH
    
    # Get atmospheric properties
    _, _, rho, speed_of_sound = compute_atmosphere_properties(altitude)
    
    if rho < C.DENSITY_FLOOR:
        return np.zeros(3)
        
    # Relative velocity (accounting for Earth rotation and wind)
    # If vehicle inertially stationary, skip wind to avoid spurious drag in tests
    if np.linalg.norm(v) < C.SMALL_VELOCITY_TOL:
        v_rel = np.zeros(3)
    else:
        v_rel = compute_relative_velocity(r, v)
    v_rel_norm = np.linalg.norm(v_rel)
    
    if v_rel_norm < C.SMALL_VELOCITY_TOL:
        return np.zeros(3)
        
    # Mach Number (guard against division by zero at high altitude)
    mach = v_rel_norm / max(speed_of_sound, 1.0)
    
    # Interpolate Cd
    cd = np.interp(mach, C.MACH_BREAKPOINTS, C.CD_VALUES)
    
    # Drag Force
    v_rel_hat = v_rel / v_rel_norm
    drag_magnitude = 0.5 * rho * cd * C.REFERENCE_AREA * v_rel_norm ** 2
    
    return -drag_magnitude * v_rel_hat


def compute_lift_force(r: np.ndarray, v: np.ndarray, q: np.ndarray) -> np.ndarray:
    """
    Compute lift force using small-angle slender-body approximation.
    Lift acts perpendicular to velocity and depends on angle of attack.
    """
    altitude = np.linalg.norm(r) - C.R_EARTH
    _, _, rho, speed_of_sound = compute_atmosphere_properties(altitude)
    if rho < C.DENSITY_FLOOR:
        return np.zeros(3)

    v_rel = compute_relative_velocity(r, v)
    v_rel_norm = np.linalg.norm(v_rel)
    if v_rel_norm < C.SMALL_VELOCITY_TOL:
        return np.zeros(3)

    mach = v_rel_norm / speed_of_sound
    cl_alpha = np.interp(mach, C.MACH_BREAKPOINTS, C.CL_ALPHA_VALUES)

    # Transform v_rel to body frame to get angle of attack
    R = quaternion_to_rotation_matrix(q)
    v_body = R.T @ v_rel
    # Body frame: +Z is nose (longitudinal axis), X and Y are transverse
    # AoA = angle between velocity vector and body longitudinal axis (+Z)
    # Using arctan2(transverse, axial) gives the total AoA correctly
    v_transverse = np.sqrt(v_body[0]**2 + v_body[1]**2)
    alpha = np.arctan2(v_transverse, abs(v_body[2]))  # radians, always positive
    q_dyn = 0.5 * rho * v_rel_norm**2
    lift_mag = q_dyn * cl_alpha * alpha * C.REFERENCE_AREA

    # Lift direction: perpendicular to v_rel and body-Y axis (assume symmetric rocket)
    # Compute unit normal in inertial frame: u_lift ~ (v_rel × (body_y_inertial)) × v_rel
    body_y_inertial = R[:, 1]
    lift_dir = np.cross(np.cross(v_rel, body_y_inertial), v_rel)
    norm = np.linalg.norm(lift_dir)
    if norm < 1e-9:
        return np.zeros(3)
    lift_dir /= norm
    return lift_mag * lift_dir


def compute_thrust_force(q: np.ndarray, r: np.ndarray, thrust_on: bool = True, throttle: float = 1.0,
                         stage: int = 1) -> np.ndarray:
    """
    Compute thrust force in inertial frame with altitude compensation and throttling.

    Thrust varies with ambient pressure (Stage 1):
    T = T_vac + (T_sl - T_vac) * (P_amb / P_sl)

    Stage 2 operates in vacuum only (no pressure compensation needed).

    Args:
        q: Orientation quaternion
        r: Position vector (for altitude/pressure)
        thrust_on: active flag
        throttle: Throttle setting (0.0 to 1.0)
        stage: Engine stage (1 = S1 engines, 2 = S2 engine)
    """
    if not thrust_on:
        return np.zeros(3)

    # Clamp throttle
    throttle = max(0.0, min(1.0, throttle))

    # Ambient conditions
    altitude = np.linalg.norm(r) - C.R_EARTH
    _, P_amb, _, _ = compute_atmosphere_properties(altitude)
    P0 = C.ATM_P0

    if stage == 2:
        # Stage 2: vacuum-optimized engine — no pressure compensation
        # T = mdot * Isp_vac * g0 (constant in vacuum, slight loss in atmosphere)
        thrust_magnitude = C.STAGE2_THRUST
        # If somehow in atmosphere (shouldn't happen), apply minor correction
        if P_amb > 100.0:  # Non-negligible atmosphere
            # Approximate: thrust drops by ~(P_amb * A_exit) but S2 nozzle is vacuum-optimized
            # Use simple correction: T_eff = T_vac * (1 - 0.1 * P_amb/P0)
            thrust_magnitude *= (1.0 - 0.1 * P_amb / P0)
    else:
        # Stage 1: pressure-compensated thrust
        # Thrust varies with ambient pressure: T = T_vac - (T_vac - T_sl) * (P_amb / P_sl)
        # This gives T_sl at sea level, T_vac in vacuum
        thrust_sl = C.THRUST_MAGNITUDE  # Sea level thrust
        thrust_vac = C.MASS_FLOW_RATE * C.ISP_VAC * C.G0  # Vacuum thrust

        # Linear interpolation based on ambient pressure ratio
        pressure_ratio = P_amb / P0
        thrust_magnitude = thrust_vac - (thrust_vac - thrust_sl) * pressure_ratio

    # Apply throttle
    thrust_magnitude *= float(np.clip(throttle, 0.0, 1.0))

    # Thrust in body frame (along +Z axis - vehicle nose direction)
    F_body = np.array([0.0, 0.0, thrust_magnitude])

    # Transform to inertial frame
    R = quaternion_to_rotation_matrix(q)
    F_inertial = R @ F_body

    return F_inertial


def compute_aerodynamic_moment(r: np.ndarray, v: np.ndarray, q: np.ndarray,
                               cg_pos_z: float, cp_pos_z: float = None) -> np.ndarray:
    """
    Compute aerodynamic moment about the Center of Mass (Body Frame).
    
    Models the aerodynamic instability (CP ahead of CG).
    M = r_arm x F_normal
    
    Args:
        r: Position (inertial)
        v: Velocity (inertial)
        q: Attitude Quaternion
        cg_pos_z: Height of CG from base (m)
        
    Returns:
        Moment vector in BODY FRAME (N*m)
    """
    altitude = np.linalg.norm(r) - C.R_EARTH
    _, _, rho, speed_of_sound = compute_atmosphere_properties(altitude)
    
    if rho < C.DENSITY_FLOOR:
        return np.zeros(3)
    
    # Early exit for stationary vehicle (same pattern as compute_drag_force)
    # This prevents spurious moments when v=0 but Earth rotation creates v_rel
    if np.linalg.norm(v) < C.SMALL_VELOCITY_TOL:
        return np.zeros(3)
        
    # Relative Velocity (Wind)
    v_rel = compute_relative_velocity(r, v)
    v_rel_norm = np.linalg.norm(v_rel)
    
    if v_rel_norm < C.SMALL_VELOCITY_TOL:
        return np.zeros(3)
        
    # Transform v_rel to Body Frame to get Angle of Attack components
    R = quaternion_to_rotation_matrix(q)
    # v_body = R.T @ v_rel  <-- R transforms Body to Inertial, so R.T transforms Inertial to Body
    v_body = R.T @ v_rel
    
    # v_body = [vx, vy, vz]
    # In body frame, Z is up (nose). Wind comes from top (-Z) mainly.
    # Transverse components cause AoA.
    vx, vy, vz = v_body
    
    # Dynamic Pressure
    q_dyn = 0.5 * rho * v_rel_norm**2
    
    # Normal Force Calculation (Linearized Aerodynamics)
    # F_normal is proportional to Angle of Attack * q_dyn * Area
    
    # Calculate transverse velocity magnitude
    v_transverse = np.sqrt(vx**2 + vy**2)
    
    if v_transverse < 1e-6:
        return np.zeros(3)
    
    # AoA (alpha) ~ v_transverse / |vz| (small angle approximation)
    # Clamp to reasonable range to prevent instability at extreme attitudes
    alpha = np.arctan2(v_transverse, abs(vz))
    alpha = np.clip(alpha, -np.radians(30), np.radians(30))  # Limit to ±30°
    
    # Normal Force Magnitude
    Fn_mag = q_dyn * C.REFERENCE_AREA * C.C_N_ALPHA * alpha
    
    # Direction of Normal Force in Body Frame:
    # Normal force acts perpendicular to the body axis in the plane of incidence,
    # effectively opposing the transverse component of velocity.
    
    # Unit vector of transverse velocity
    u_trans = np.array([vx, vy, 0.0]) / v_transverse
    
    # Normal Force Vector
    F_normal_body = -Fn_mag * u_trans
    
    # Lever Arm from CG to CP
    if cp_pos_z is None:
        cp_pos_z = C.H_CP
    arm_z = cp_pos_z - cg_pos_z
    r_arm = np.array([0.0, 0.0, arm_z])
    
    # Torque = r x F
    torque_aero = np.cross(r_arm, F_normal_body)
    
    return torque_aero


def compute_total_force(r: np.ndarray, v: np.ndarray, q: np.ndarray,
                        m: float, thrust_on: bool = True, stage: int = 1) -> np.ndarray:
    """
    Compute total force acting on the vehicle in the ECI (inertial) frame.

    F_total = F_grav + F_thrust + F_drag + F_lift

    Physics note: No Coriolis force is included because this simulation
    operates in an Earth-Centered Inertial (ECI) frame. Coriolis is a
    fictitious force that only appears in rotating (ECEF) reference frames.
    Earth's rotation IS properly accounted for via the air-relative velocity
    (v_inertial - omega_E x r) used in drag and lift computations.
    """
    F_grav = compute_gravity_force(r, m)
    F_thrust = compute_thrust_force(q, r, thrust_on, stage=stage)

    if stage == 2:
        # Doc §D.7: "Aerodynamic drag is neglected due to near-vacuum conditions"
        # S2 operates above ~110 km where atmosphere is negligible
        F_drag = np.zeros(3)
        F_lift = np.zeros(3)
    else:
        F_drag = compute_drag_force(r, v)
        F_lift = compute_lift_force(r, v, q)

    return F_grav + F_thrust + F_drag + F_lift


def compute_centrifugal_correction(r: np.ndarray, m: float) -> np.ndarray:
    """
    Compute the centrifugal force for ECEF (rotating) frame analysis only.

    In a rotating frame, the centrifugal force points radially OUTWARD:
        F_centrifugal = -m * omega_E x (omega_E x r)

    Note: omega x (omega x r) points radially INWARD (centripetal).
    The centrifugal pseudo-force is its negative (outward).

    This is NOT used in the ECI dynamics (where it doesn't exist) but is
    provided for rotating-frame analysis, effective gravity computation, etc.

    Args:
        r: Position in ECI frame (m)
        m: Vehicle mass (kg)

    Returns:
        Centrifugal force vector (N) — outward, for ECEF analysis only
    """
    omega_earth = np.array([0.0, 0.0, C.EARTH_ROTATION_RATE])
    # omega x (omega x r) = centripetal (inward); negate for centrifugal (outward)
    return -m * np.cross(omega_earth, np.cross(omega_earth, r))


def compute_specific_forces(r: np.ndarray, v: np.ndarray, q: np.ndarray,
                            m: float, thrust_on: bool = True, stage: int = 1) -> ForceBreakdown:
    """
    Compute all forces and return as a dictionary for logging/analysis.

    Consistent with dynamics.py: gravity + thrust + drag + lift in ECI frame.
    """
    F_grav = compute_gravity_force(r, m)
    F_thrust = compute_thrust_force(q, r, thrust_on, stage=stage)

    if stage == 2:
        # Doc §D.7: drag neglected in near-vacuum (consistent with compute_total_force)
        F_drag = np.zeros(3)
        F_lift = np.zeros(3)
    else:
        F_drag = compute_drag_force(r, v)
        F_lift = compute_lift_force(r, v, q)

    return {
        'gravity': F_grav,
        'thrust': F_thrust,
        'drag': F_drag,
        'lift': F_lift,
        'total': F_grav + F_thrust + F_drag + F_lift,
        'gravity_magnitude': np.linalg.norm(F_grav),
        'thrust_magnitude': np.linalg.norm(F_thrust),
        'drag_magnitude': np.linalg.norm(F_drag),
        'lift_magnitude': np.linalg.norm(F_lift)
    }
