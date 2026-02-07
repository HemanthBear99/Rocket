"""
Guidance module implementing altitude-based gamma profile with PID tracking
and thrust-direction generation for gravity turn.
"""

import numpy as np

from . import constants as C
from .forces import compute_atmosphere_properties
from .types import GuidanceOutput
from .utils import compute_relative_velocity


def compute_local_vertical(r: np.ndarray) -> np.ndarray:
    r_norm = np.linalg.norm(r)
    if r_norm < C.ZERO_TOLERANCE:
        return np.array([1.0, 0.0, 0.0])
    return r / r_norm


def compute_local_frame(r: np.ndarray):
    vertical = compute_local_vertical(r)
    k_axis = np.array([0.0, 0.0, 1.0])
    east = np.cross(k_axis, vertical)
    if np.linalg.norm(east) < C.ZERO_TOLERANCE:
        east = np.array([0.0, 1.0, 0.0])
    east = east / np.linalg.norm(east)
    north = np.cross(vertical, east)
    north = north / np.linalg.norm(north)
    return vertical, east, north


def compute_local_horizontal(r: np.ndarray, v: np.ndarray) -> np.ndarray:
    vertical = compute_local_vertical(r)
    v_rel = compute_relative_velocity(r, v)
    v_rel_norm = float(np.linalg.norm(v_rel))
    if v_rel_norm < C.ZERO_TOLERANCE:
        return vertical
    return v_rel / v_rel_norm


def gamma_profile_from_altitude(altitude: float) -> float:
    """
    Continuous, smooth gamma profile (gravity turn) based on altitude.
    
    Uses a tanh-based shaping function to ensure C1 continuity (smooth derivatives)
    which helps the Attitude Control System track the command without saturating.
    
    Gamma starts at 90 deg (Vertical) and smoothly decays to ~18 deg at MECO.
    """
    # Parameters for the smooth turn
    # Start turn around 1 km, main turn fits the 10-60km region.
    
    # Normalized altitude coordinate (0 to 1) for the main atmospheric ascent
    # We want a smooth transition from 90 -> target_gamma
    
    # Target gamma at 110km is ~18-20 deg
    gamma_start = 90.0
    gamma_final = 18.0 
    
    # Smoothstep-like or sigmoid transition
    # Use a modified tanh shape: y = start + (final - start) * tanh(k * alt)^n
    # Or simply: y = 90 - (90 - final) * f(alt)
    # where f(alt) goes 0->1
    
    # Turn parameters
    turn_start_alt = 1000.0   # m
    # Revert to 45000 to ensure the profile actually commands a turn compatible with physics
    turn_scale = 45000.0      

    if altitude < turn_start_alt:
        return np.radians(gamma_start)
        
    # Normalized altitude for turn curve
    h_norm = (altitude - turn_start_alt) / turn_scale
    
    # Tanh shape: goes 0 -> 1 smoothly
    progress = np.tanh(h_norm)
    
    # Gamma in degrees
    gamma_deg = gamma_start + (gamma_final - gamma_start) * progress
    
    return np.radians(gamma_deg)


def compute_blend_parameter(altitude: float) -> float:
    """
    Blend factor (0 → vertical ascent, 1 → prograde) based on altitude.

    A smooth ramp is used so guidance phase labels and thrust blending
    remain continuous. 
    """
    # Smooth sigmoid blend instead of linear ramp
    start = C.GRAVITY_TURN_START_ALTITUDE
    width = C.GRAVITY_TURN_TRANSITION_RANGE
    
    # Sigmoid function centered at (start + width/2)
    # 0.0 at < start, 1.0 at > start + width
    # Simple clamped linear is often robust enough for blending, but let's smooth it
    
    if altitude <= start:
        return 0.0
    if altitude >= start + width:
        return 1.0
        
    # Hermite interpolation (smoothstep): 3x^2 - 2x^3
    x = (altitude - start) / width
    return x * x * (3.0 - 2.0 * x)


# PID state (encapsulated in module-level variables - reset between runs)
_pid_state = {
    'prev_gamma_meas': 90.0,
    'gamma_int': 0.0
}


def reset_guidance():
    """Reset guidance PID state. Call this before starting a new simulation."""
    global _pid_state
    _pid_state['prev_gamma_meas'] = 90.0
    _pid_state['gamma_int'] = 0.0


def compute_desired_thrust_direction(r: np.ndarray, v: np.ndarray, t: float, dt: float = C.DT):
    global _pid_state

    altitude = float(np.linalg.norm(r) - C.R_EARTH)
    vertical, east, north = compute_local_frame(r)

    # -------------------------------------------------------------------------
    # 1. State-Dependent Gamma Command (Smooth)
    # -------------------------------------------------------------------------
    gamma_target = gamma_profile_from_altitude(altitude)
    gamma_target_deg = float(np.degrees(gamma_target))

    # -------------------------------------------------------------------------
    # 2. Measurement & PID correction
    # -------------------------------------------------------------------------
    v_rel = compute_relative_velocity(r, v)
    v_rel_norm = float(np.linalg.norm(v_rel))
    
    v_vert = float(np.dot(v_rel, vertical))
    v_horiz_vec = v_rel - v_vert * vertical
    v_horiz = float(np.linalg.norm(v_horiz_vec))
    
    if v_rel_norm < 1e-3:
        gamma_meas_deg = 90.0
    else:
        gamma_meas_deg = float(np.degrees(np.arctan2(v_vert, v_horiz)))

    error = gamma_target_deg - gamma_meas_deg
    gamma_rate = (gamma_meas_deg - _pid_state['prev_gamma_meas']) / max(dt, 1e-3)

    # PID Gains
    kp, ki, kd = 0.85, 0.05, 0.12
    _pid_state['gamma_int'] += error * dt
    _pid_state['gamma_int'] = float(np.clip(_pid_state['gamma_int'], -20.0, 20.0))
    gamma_raw = gamma_target_deg + kp * error + ki * _pid_state['gamma_int'] - kd * gamma_rate
    
    # -------------------------------------------------------------------------
    # 3. Dynamic Pressure & Zero-Lift Logic (Prograde Locking)
    # -------------------------------------------------------------------------
    _, _, rho, _ = compute_atmosphere_properties(altitude)
    q_dyn = 0.5 * rho * v_rel_norm**2
    
    # Calculate Velocity Vector Direction (Prograde)
    if v_rel_norm > 1.0:
        prograde = v_rel / v_rel_norm
    else:
        prograde = vertical

    # Initial "Guidance" Thrust Vector (from Gamma Profile PID)
    gamma_cmd_clamped = float(np.clip(gamma_raw, 10.0, 90.0))
    pitch_cmd_rad = np.radians(90.0 - gamma_cmd_clamped)
    
    if v_horiz > 1.0:
        horiz_axis = v_horiz_vec / v_horiz
    else:
        horiz_axis = east 
        
    guidance_dir = np.cos(pitch_cmd_rad) * vertical + np.sin(pitch_cmd_rad) * horiz_axis
    guidance_dir /= np.linalg.norm(guidance_dir)

    # -------------------------------------------------------------------------
    # 4. Mode Mixing: Blend from Guidance to Prograde based on ALTITUDE
    # -------------------------------------------------------------------------
    # We blend nicely from 1km to 10km to avoid Q-based jumps.
    # 0km - 1km: Pure Guidance (Launch/Vertical)
    # 1km - 10km: Blend in Prograde (Gravity Turn Start)
    # > 10km: Pure Prograde (Gravity Turn Lock)
    
    mix_start_alt = 1000.0
    mix_end_alt = 10000.0
    
    if altitude < mix_start_alt:
        w_prograde = 0.0
    elif altitude > mix_end_alt:
        w_prograde = 1.0
    else:
        w_prograde = (altitude - mix_start_alt) / (mix_end_alt - mix_start_alt)
        
    # Safety Override: If Q is massive (>20kPa), FORCE prograde regardless of altitude
    # to avoid structural break.
    if q_dyn > 20000.0:
         w_prograde = 1.0
        
    # Apply Blend
    # thrust_dir = (1 - w) * guidance + w * prograde
    thrust_dir_mixed = (1.0 - w_prograde) * guidance_dir + w_prograde * prograde
    
    # Re-normalize
    norm_mixed = np.linalg.norm(thrust_dir_mixed)
    if norm_mixed > 1e-6:
        thrust_dir_mixed /= norm_mixed
    else:
        thrust_dir_mixed = guidance_dir
        
    thrust_dir = thrust_dir_mixed
    
    # Update gamma_cmd simply for logging (what are we actually commanding?)
    # We back-calculate the effective command from our final vector
    cos_p = np.dot(thrust_dir, vertical)
    gamma_cmd_clamped = np.degrees(np.arcsin(np.clip(cos_p, -1.0, 1.0)))

    _pid_state['prev_gamma_meas'] = gamma_meas_deg

    return thrust_dir, gamma_cmd_clamped, gamma_meas_deg


def compute_guidance_output(r: np.ndarray, v: np.ndarray, t: float, m: float,
                            meco_mass: float = None) -> GuidanceOutput:
    """
    Compute ascent guidance output for Stage 1 powered flight.

    Args:
        r: Position vector (ECI, m)
        v: Velocity vector (ECI, m/s)
        t: Current time (s)
        m: Current vehicle mass (kg)
        meco_mass: Mass at which MECO occurs. Defaults to stacked dry mass
                   plus S1 landing fuel reserve.
    """
    if meco_mass is None:
        # S1 MECO occurs when ascent propellant is depleted, reserving fuel for landing
        # MECO mass = S1_dry + S2_wet + landing_reserve
        meco_mass = C.DRY_MASS + C.STAGE1_LANDING_FUEL_RESERVE

    altitude = float(np.linalg.norm(r) - C.R_EARTH)
    thrust_dir, gamma_cmd, gamma_meas = compute_desired_thrust_direction(r, v, t)

    alpha = compute_blend_parameter(altitude)
    if alpha < 0.01:
        if C.PITCHOVER_START_ALTITUDE <= altitude <= C.PITCHOVER_END_ALTITUDE:
            phase = "PITCHOVER"
        else:
            phase = "VERTICAL_ASCENT"
    elif alpha < 0.99:
        phase = "GRAVITY_TURN"
    else:
        phase = "PROGRADE"

    thrust_on = (m > meco_mass)
    v_rel = compute_relative_velocity(r, v)
    v_rel_norm = float(np.linalg.norm(v_rel))

    # Max-Q shaping: reduce throttle smoothly when dynamic pressure exceeds limit
    _, _, rho, _ = compute_atmosphere_properties(altitude)
    q_dyn = 0.5 * rho * (v_rel_norm ** 2)
    q_limit = C.MAX_DYNAMIC_PRESSURE
    if q_dyn > 0.8 * q_limit:
        throttle = float(np.clip(np.sqrt(q_limit / (q_dyn + 1e-6)), 0.55, 1.0))
    else:
        throttle = 1.0

    if v_rel_norm < C.ZERO_TOLERANCE:
        prograde = compute_local_vertical(r)
    else:
        prograde = v_rel / v_rel_norm

    vertical = compute_local_vertical(r)
    cos_pitch = np.clip(np.dot(vertical, thrust_dir), -1.0, 1.0)
    pitch_angle = float(np.arccos(cos_pitch))
    gamma_angle = np.radians(gamma_cmd)

    return {
        'thrust_direction': thrust_dir,
        'phase': phase,
        'thrust_on': thrust_on,
        'pitch_angle': pitch_angle,
        'gamma_angle': gamma_angle,
        'gamma_command_deg': gamma_cmd,
        'gamma_measured_deg': gamma_meas,
        'velocity_tilt_deg': gamma_meas,
        'blend_alpha': alpha,
        'altitude': altitude,
        'velocity': float(np.linalg.norm(v)),
        'local_vertical': vertical,
        'local_horizontal': compute_local_horizontal(r, v),
        'prograde': prograde,
        'throttle': throttle,
        'v_rel': v_rel,
        'v_rel_mag': v_rel_norm
    }


def compute_orbit_insertion_guidance(r: np.ndarray, v: np.ndarray, t: float, m: float) -> GuidanceOutput:
    """
    Guidance logic for S2 orbit insertion burn.

    Steering law: TWR-adaptive pitch program with altitude feedback.

    The S2 engine has TWR < 1 at ignition (0.50 at 120 t), increasing to
    TWR = 3.5 at burnout (17 t). The pitch program must balance:
    - Building horizontal velocity (requires low pitch = near horizontal)
    - Preventing excessive altitude loss (requires high pitch = near vertical)

    Strategy:
    The pitch angle above local horizontal is computed in two parts:

    1. BASE PITCH from velocity progress toward circular:
       As v_horizontal approaches v_circular, pitch decreases from ~40 deg to 0.
       pitch_base = 40 * (1 - v_h/v_circ)^0.6  [degrees]

    2. ALTITUDE FEEDBACK correction:
       - If radial velocity is negative (descending), add upward bias
       - If altitude drops below minimum safe threshold, increase bias
       - This prevents the perigee from dropping into the atmosphere

    The result is a smooth pitch program that:
    - Starts steep (~40 deg) to gain altitude and arrest the natural fall
    - Gradually flattens as horizontal velocity increases
    - Goes nearly horizontal near orbital velocity
    - Uses altitude feedback as a safety net

    Termination: v_circular reached OR S2 propellant exhausted.
    """
    altitude = float(np.linalg.norm(r) - C.R_EARTH)
    v_inertial = float(np.linalg.norm(v))
    r_mag = float(np.linalg.norm(r))
    vertical = compute_local_vertical(r)

    # Circular velocity at current radius
    v_circular = float(np.sqrt(C.MU_EARTH / r_mag))
    v_deficit = v_circular - v_inertial

    # Decompose velocity into radial and horizontal components
    v_radial = float(np.dot(v, vertical))
    v_horiz_vec = v - v_radial * vertical
    v_horiz = float(np.linalg.norm(v_horiz_vec))

    # Local horizontal direction (in the velocity plane, perpendicular to radial)
    if v_horiz > 10.0:
        horiz_hat = v_horiz_vec / v_horiz
    else:
        _, east, _ = compute_local_frame(r)
        horiz_hat = east

    # =========================================================================
    # TWR-ADAPTIVE PITCH PROGRAM
    # =========================================================================

    # Progress toward circular velocity (0 = just started, 1 = orbital)
    v_progress = float(np.clip(v_horiz / v_circular, 0.0, 1.0))

    # Base pitch angle (degrees above horizontal)
    # Starts at ~15 deg when v_h/v_circ ~ 0.28 (separation velocity)
    # Decreases to 0 as v_h approaches v_circ
    # With TWR=0.83, we need most thrust going horizontal to reach orbit.
    # The small positive pitch angle prevents rapid altitude loss.
    pitch_base_deg = 15.0 * (1.0 - v_progress) ** 0.4

    # Altitude feedback: arrest descent rate
    # If vehicle is descending, add pitch-up correction
    pitch_correction_deg = 0.0
    if v_radial < 0.0:
        # Proportional to descent rate: 1 m/s descent -> 0.15 deg correction
        pitch_correction_deg += min(15.0, -v_radial * 0.15)

    # Altitude floor protection: if altitude drops below safe threshold,
    # add strong upward bias to arrest descent
    min_safe_alt = 80000.0  # 80 km — above atmosphere
    if altitude < min_safe_alt + 20000.0:
        # Ramp up correction as altitude approaches floor
        alt_margin = (altitude - min_safe_alt) / 20000.0  # 1.0 at 100km, 0 at 80km
        alt_margin = float(np.clip(alt_margin, 0.0, 1.0))
        pitch_correction_deg += 20.0 * (1.0 - alt_margin)

    # Terminal flattening: when approaching orbital velocity, actively
    # reduce radial velocity by steering below horizontal.
    # This ensures the orbit circularizes (low eccentricity) rather than
    # being highly elliptical with a low perigee.
    # Start early (3000 m/s deficit) and increase aggressiveness linearly.
    if v_deficit < 4000.0 and v_radial > 10.0:
        # Pitch DOWN below horizontal to arrest radial velocity
        # The closer to orbital speed, the more aggressively we flatten
        flatten_factor = float(np.clip(1.0 - v_deficit / 4000.0, 0.0, 1.0))
        # Pitch down proportional to radial velocity and proximity to v_circular
        pitch_down_deg = flatten_factor * min(20.0, v_radial * 0.05)
        pitch_base_deg -= pitch_down_deg

    # Total pitch angle
    pitch_cmd_deg = pitch_base_deg + pitch_correction_deg
    pitch_cmd_deg = float(np.clip(pitch_cmd_deg, -10.0, 75.0))  # Allow slight negative (below horizontal)
    pitch_cmd_rad = np.radians(pitch_cmd_deg)

    # Compute thrust direction from pitch angle
    # pitch_cmd_rad is angle above local horizontal
    desired_dir = np.cos(pitch_cmd_rad) * horiz_hat + np.sin(pitch_cmd_rad) * vertical
    desired_dir = desired_dir / np.linalg.norm(desired_dir)

    # S2 propellant remaining
    s2_prop_remaining = m - C.STAGE2_DRY_MASS
    thrust_on = s2_prop_remaining > 0.0 and v_deficit > 10.0

    # Throttle shaping: reduce throttle as we approach target to avoid overshoot
    if v_deficit < 100.0 and v_deficit > 10.0:
        throttle = float(np.clip(v_deficit / 100.0, 0.1, 1.0))
    elif v_deficit <= 10.0:
        throttle = 0.0
        thrust_on = False
    else:
        throttle = 1.0

    cos_pitch = np.clip(np.dot(vertical, desired_dir), -1.0, 1.0)
    pitch_angle = float(np.arccos(cos_pitch))

    v_rel = compute_relative_velocity(r, v)
    v_rel_norm = float(np.linalg.norm(v_rel))
    v_vert = float(np.dot(v_rel, vertical))
    v_horiz_vec_rel = v_rel - v_vert * vertical
    v_horiz_rel = float(np.linalg.norm(v_horiz_vec_rel))
    gamma_actual = float(np.degrees(np.arctan2(v_vert, max(v_horiz_rel, 1e-6))))

    if v_inertial > 1.0:
        prograde = v / v_inertial
    else:
        prograde = vertical

    return {
        'thrust_direction': desired_dir,
        'phase': "ORBIT_INSERTION",
        'thrust_on': thrust_on,
        'pitch_angle': pitch_angle,
        'gamma_angle': np.radians(gamma_actual),
        'gamma_command_deg': gamma_actual,
        'gamma_measured_deg': gamma_actual,
        'velocity_tilt_deg': gamma_actual,
        'blend_alpha': 1.0,
        'altitude': altitude,
        'velocity': v_inertial,
        'local_vertical': vertical,
        'local_horizontal': compute_local_horizontal(r, v),
        'prograde': prograde,
        'throttle': throttle,
        'v_rel': v_rel,
        'v_rel_mag': v_rel_norm
    }


def compute_coast_guidance(r: np.ndarray, v: np.ndarray, t: float, m: float) -> GuidanceOutput:
    """
    Guidance logic for Phase II (Coast).

    - Thrust: OFF
    - Attitude: Prograde (aligned with relative velocity)
    """
    altitude = float(np.linalg.norm(r) - C.R_EARTH)
    v_rel = compute_relative_velocity(r, v)
    v_rel_norm = float(np.linalg.norm(v_rel))
    
    # 1. Orientation: Align with Velocity Vector (Prograde)
    if v_rel_norm > 1.0:
        desired_dir = v_rel / v_rel_norm
    else:
        desired_dir = compute_local_vertical(r)
        
    vertical = compute_local_vertical(r)
    cos_pitch = np.clip(np.dot(vertical, desired_dir), -1.0, 1.0)
    pitch_angle = float(np.arccos(cos_pitch))
    
    # Gamma calculation for logging
    v_vert = float(np.dot(v_rel, vertical))
    v_horiz_vec = v_rel - v_vert * vertical
    v_horiz = float(np.linalg.norm(v_horiz_vec))
    gamma_actual = float(np.degrees(np.arctan2(v_vert, v_horiz)))

    return {
        'thrust_direction': desired_dir,
        'phase': "COAST_PHASE",
        'thrust_on': False,
        'pitch_angle': pitch_angle,
        'gamma_angle': np.radians(gamma_actual),
        'gamma_command_deg': gamma_actual, # We "command" what we have since we follow velocity
        'gamma_measured_deg': gamma_actual,
        'velocity_tilt_deg': gamma_actual, 
        'blend_alpha': 1.0, # Pure prograde
        'altitude': altitude,
        'velocity': float(np.linalg.norm(v)),
        'local_vertical': vertical,
        'local_horizontal': compute_local_horizontal(r, v),
        'prograde': desired_dir,
        'throttle': 0.0,
        'v_rel': v_rel,
        'v_rel_mag': v_rel_norm
    }


def _compute_boostback_direction(r: np.ndarray, v: np.ndarray,
                                  launch_site: np.ndarray) -> np.ndarray:
    """
    Compute optimal boostback thrust direction to return to launch site.

    Uses the velocity-to-be-gained (VTG) method:
      v_desired = unit(launch_site - r) * |some speed toward site|
      VTG = v_desired - v
      thrust_dir = -unit(VTG)   (burn retrograde to the velocity error)

    For boostback the primary goal is to cancel downrange velocity and
    add velocity toward the launch site, so we burn opposite to the
    horizontal velocity component relative to the launch-site direction.
    """
    # Vector from vehicle to launch site
    r_to_site = launch_site - r
    r_to_site_norm = np.linalg.norm(r_to_site)

    if r_to_site_norm < 1e3:
        # Already very close — just point retrograde
        v_norm = np.linalg.norm(v)
        if v_norm > 1.0:
            return -v / v_norm
        return compute_local_vertical(r)

    # Desired velocity direction: toward launch site, in the horizontal plane
    vertical = compute_local_vertical(r)

    # Project r_to_site onto horizontal plane
    r_horiz = r_to_site - np.dot(r_to_site, vertical) * vertical
    r_horiz_norm = np.linalg.norm(r_horiz)

    if r_horiz_norm < 1e3:
        # Directly above launch site — just go retrograde vertically
        v_norm = np.linalg.norm(v)
        if v_norm > 1.0:
            return -v / v_norm
        return vertical

    # Target direction: weighted blend of (towards site) and (retrograde)
    # to simultaneously cancel downrange velocity and head home
    towards_site = r_horiz / r_horiz_norm

    # Horizontal velocity component
    v_horiz = v - np.dot(v, vertical) * vertical
    v_horiz_norm = np.linalg.norm(v_horiz)

    # Velocity to be gained: we want v_horiz to point toward site
    # VTG = desired_v_horiz - actual_v_horiz
    # We want zero downrange velocity eventually, so desired is small toward site
    desired_speed = min(200.0, r_horiz_norm / 60.0)  # approach speed
    v_desired = towards_site * desired_speed
    vtg = v_desired - v_horiz

    vtg_norm = np.linalg.norm(vtg)
    if vtg_norm < 1.0:
        # Velocity error negligible — coast
        return -v / max(np.linalg.norm(v), 1.0)

    # Thrust along VTG direction (burn to gain what we lack)
    thrust_dir = vtg / vtg_norm

    # Add a vertical component proportional to descent rate to keep altitude
    v_vert = np.dot(v, vertical)
    if v_vert < -50.0:
        # Falling fast — add upward component
        vert_weight = min(0.3, abs(v_vert) / 1000.0)
        thrust_dir = (1.0 - vert_weight) * thrust_dir + vert_weight * vertical
        thrust_dir /= np.linalg.norm(thrust_dir)

    return thrust_dir


def _compute_suicide_burn_params(r: np.ndarray, v: np.ndarray,
                                  m: float, dry_mass: float) -> dict:
    """
    Compute suicide burn (hoverslam) ignition parameters.

    Physics: For a vehicle descending at speed v at altitude h,
    the required burn altitude for a constant-thrust deceleration
    to zero velocity at h=0 is:

        a_brake = T/m - g      (net deceleration)
        v² = 2 * a_brake * h   (kinematics)
        h_ignite = v² / (2 * a_brake)

    We compute this continuously and command ignition when
    current altitude <= h_ignite * safety_factor.

    Returns dict with: ignite (bool), throttle (float), burn_altitude (float)
    """
    vertical = compute_local_vertical(r)
    altitude = float(np.linalg.norm(r) - C.R_EARTH)

    # Descent speed (positive = downward)
    v_descent = -float(np.dot(v, vertical))

    if v_descent <= 0.0:
        # Not descending yet — no burn needed
        return {'ignite': False, 'throttle': 0.0, 'burn_altitude': 0.0}

    # Net deceleration at current mass (full throttle)
    # T/m - g (accounting for gravity opposing the burn)
    g_local = C.MU_EARTH / np.linalg.norm(r)**2
    thrust_accel = C.THRUST_MAGNITUDE / m  # Using sea-level thrust (conservative)
    a_brake = thrust_accel - g_local

    if a_brake <= 0.0:
        # Thrust cannot overcome gravity — burn immediately at full throttle
        return {'ignite': True, 'throttle': 1.0, 'burn_altitude': altitude}

    # Required burn altitude from kinematics: v² = 2*a*h
    # Use total speed (not just vertical) for conservative estimate since
    # horizontal velocity also needs to be arrested near the ground
    v_total = float(np.linalg.norm(v))
    v_effective = max(v_descent, v_total * 0.8)  # conservative: 80% of total speed
    h_ignite = (v_effective ** 2) / (2.0 * a_brake)

    # Safety margin: ignite 50% early to account for throttle lag, mass change,
    # atmospheric deceleration uncertainty, and engine spool-up time
    safety_factor = 1.5

    # Throttle shaping for soft touchdown
    if altitude <= h_ignite * safety_factor:
        # We need to burn. Compute throttle to achieve desired deceleration.
        # Target: arrive at h=0 with v=0
        # Required decel: v² / (2*h) + g
        if altitude > 10.0:
            a_required = (v_descent**2) / (2.0 * altitude) + g_local
        else:
            # Final meters — full power to arrest
            a_required = thrust_accel

        # Throttle = a_required / thrust_accel (clamped)
        throttle = float(np.clip(a_required / thrust_accel, 0.3, 1.0))

        return {'ignite': True, 'throttle': throttle, 'burn_altitude': h_ignite}

    return {'ignite': False, 'throttle': 0.0, 'burn_altitude': h_ignite}


def compute_booster_guidance(r: np.ndarray, v: np.ndarray, t: float, m: float, phase: str) -> GuidanceOutput:
    """
    Guidance logic for Booster Recovery (Phase III-B).

    Physics-based guidance for each sub-phase:

    FLIP:       Reorient 180 deg retrograde. Zero thrust, RCS/aero torques.
                Transition: when body axis aligns within 15 deg of retrograde.

    BOOSTBACK:  Velocity-to-be-gained (VTG) steering toward launch site.
                Full throttle to cancel downrange velocity and head home.
                Transition: when horizontal velocity toward launch site < threshold.

    COAST:      Ballistic arc. Attitude holds retrograde for entry prep.
                Transition: at entry interface (altitude < 70 km, densifying atmosphere).

    ENTRY:      Retrograde attitude for aerodynamic braking. High-drag belly-first
                orientation. No thrust (save fuel for landing).
                Transition: subsonic and below 5 km.

    LANDING:    Suicide burn (hoverslam) guidance. Physics-based ignition timing:
                h_ignite = v² / (2*(T/m - g)) with throttle modulation.
    """
    altitude = float(np.linalg.norm(r) - C.R_EARTH)

    # Defaults
    thrust_on = False
    throttle = 0.0
    desired_dir = compute_local_vertical(r)

    v_rel = compute_relative_velocity(r, v)
    v_rel_norm = float(np.linalg.norm(v_rel))

    prograde = v_rel / v_rel_norm if v_rel_norm > 1.0 else compute_local_vertical(r)
    retrograde = -prograde
    vertical = compute_local_vertical(r)

    # Launch site for boostback targeting
    launch_site = C.INITIAL_POSITION.copy()

    # --- PHASE LOGIC ---
    if phase == "BOOSTER_FLIP":
        # Point engines retrograde (180-deg flip maneuver)
        # During flip we are in vacuum/near-vacuum, using RCS/reaction wheels
        # Attitude controller handles the rotation; guidance just commands direction
        desired_dir = retrograde
        thrust_on = False

    elif phase == "BOOSTER_BOOSTBACK":
        # VTG steering toward launch site at full throttle
        desired_dir = _compute_boostback_direction(r, v, launch_site)
        thrust_on = True
        throttle = 1.0

    elif phase == "BOOSTER_COAST":
        # Ballistic coast — hold retrograde attitude for entry prep
        # This minimizes AoA at entry interface
        desired_dir = retrograde
        thrust_on = False

    elif phase == "BOOSTER_ENTRY":
        # Entry phase: retrograde attitude with entry burn for deceleration
        # Similar to Falcon 9 "entry burn" at ~70-40 km altitude.
        # This reduces velocity before the final landing burn, lowering
        # the fuel requirement and thermal loads.
        desired_dir = retrograde

        # Entry burn: fire engines retrograde when speed exceeds threshold
        # to reduce velocity before dense atmosphere. Similar to Falcon 9 entry burn.
        # Only burn in upper entry corridor (35-70 km) and reserve fuel for landing.
        propellant_remaining = m - C.STAGE1_DRY_MASS
        landing_fuel_reserve = 20000.0  # kg — enough for ~500 m/s dV at landing mass

        entry_burn_on = (
            altitude > 35000.0 and                      # Upper entry corridor
            v_rel_norm > 800.0 and                      # Only if going fast
            propellant_remaining > landing_fuel_reserve  # Reserve for landing
        )

        if entry_burn_on:
            thrust_on = True
            throttle = 0.7  # Partial throttle to conserve fuel
        else:
            thrust_on = False

    elif phase == "BOOSTER_LANDING":
        # Suicide burn / hoverslam guidance
        # Physics: ignite when h = v²/(2*(T/m - g)) * safety_margin
        burn_params = _compute_suicide_burn_params(r, v, m, C.STAGE1_DRY_MASS)

        # Compute descent rate for intelligent thrust vectoring
        v_descent = -float(np.dot(v, vertical))  # positive = going down
        v_horiz_vec = v - np.dot(v, vertical) * vertical
        v_horiz_mag = float(np.linalg.norm(v_horiz_vec))
        v_total = float(np.linalg.norm(v))

        if burn_params['ignite']:
            # Thrust direction strategy:
            # High altitude: retrograde (cancel total velocity efficiently)
            # Transition: blend toward vertical with anti-horizontal correction
            # Low altitude: mostly vertical to ensure soft touchdown

            if altitude > 2000.0:
                # Above 2km: pure retrograde to cancel all velocity
                desired_dir = retrograde
            elif altitude > 200.0:
                # 200m to 2km: blend from retrograde toward vertical
                # but add a horizontal cancellation component
                blend = (altitude - 200.0) / 1800.0  # 1.0 at 2km, 0.0 at 200m
                desired_dir = blend * retrograde + (1.0 - blend) * vertical

                # Add horizontal correction if significant horizontal speed remains
                if v_horiz_mag > 20.0:
                    horiz_correction = -v_horiz_vec / v_horiz_mag
                    # Tilt toward horizontal cancel (proportional to h_speed ratio)
                    horiz_weight = min(0.3, v_horiz_mag / v_total) * (1.0 - blend)
                    desired_dir = (1.0 - horiz_weight) * desired_dir + horiz_weight * horiz_correction

                norm_d = np.linalg.norm(desired_dir)
                if norm_d > 1e-6:
                    desired_dir /= norm_d
                else:
                    desired_dir = vertical
            else:
                # Below 200m: pure vertical for touchdown
                desired_dir = vertical

            thrust_on = True
            # Use full throttle when velocity is high, modulate near touchdown
            if v_total > 100.0:
                throttle = 1.0  # Full power to decelerate
            else:
                throttle = burn_params['throttle']
        else:
            # Not yet time to ignite — coast in retrograde attitude
            desired_dir = retrograde
            thrust_on = False

    # --- Common flight-path angle computation ---
    v_vert = float(np.dot(v_rel, vertical))
    v_horiz_vec = v_rel - v_vert * vertical
    v_horiz = float(np.linalg.norm(v_horiz_vec))

    if v_rel_norm > 1.0:
        gamma_deg = float(np.degrees(np.arctan2(v_vert, max(v_horiz, 1e-6))))
    else:
        gamma_deg = 90.0 if np.dot(v, vertical) >= 0 else -90.0

    cos_pitch = np.clip(np.dot(vertical, desired_dir), -1.0, 1.0)
    pitch_angle = float(np.arccos(cos_pitch))

    return {
        'thrust_direction': desired_dir,
        'phase': phase,
        'thrust_on': thrust_on,
        'pitch_angle': pitch_angle,
        'gamma_angle': np.radians(gamma_deg),
        'gamma_command_deg': gamma_deg,
        'gamma_measured_deg': gamma_deg,
        'velocity_tilt_deg': float(np.degrees(np.arctan2(v_horiz, abs(v_vert)))) if v_rel_norm > 1.0 else 0.0,
        'blend_alpha': 1.0,
        'altitude': altitude,
        'velocity': float(np.linalg.norm(v)),
        'local_vertical': vertical,
        'local_horizontal': compute_local_horizontal(r, v),
        'prograde': prograde,
        'throttle': throttle,
        'v_rel': v_rel,
        'v_rel_mag': v_rel_norm
    }

