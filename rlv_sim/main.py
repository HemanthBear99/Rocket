"""
RLV Phase-I Ascent Simulation - Main Entry Point

This module implements the main simulation loop with:
- Single continuous simulation
- Correct execution order per timestep
- Data logging
- Logging framework for diagnostics

Coordinate Frames:
- Position/Velocity: Earth-Centered Inertial (ECI) frame
- Attitude: Body frame aligned with vehicle axes
- Quaternion convention: [w, x, y, z] (scalar-first)
"""

import logging
import time
import os
import csv
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from . import constants as C
from .config import SimulationConfig, create_default_config
from .state import State, create_initial_state
from .guidance import (
    compute_guidance_output, reset_guidance, create_guidance_state,
    GuidanceState, compute_local_vertical, compute_coast_guidance,
    compute_booster_guidance, compute_orbit_insertion_guidance,
)
from .control import compute_control_output, compute_commanded_quaternion
from .integrators import integrate
from .validation import validate_state, ValidationError, compute_total_energy, validate_energy_conservation
from .mass import is_propellant_exhausted, compute_inertia_tensor
from .frames import rotate_vector_by_quaternion
from .actuator import ActuatorState, update_actuator
from .forces import compute_atmosphere_properties
from .mission_manager import MissionManager, MissionPhase
from .utils import compute_relative_velocity
from .recovery import target_landing_site_eci, great_circle_distance_m

# Configure module logger
logger = logging.getLogger(__name__)


@dataclass
class SimulationLog:
    """Container for logged simulation data."""
    time: List[float] = field(default_factory=list)
    altitude: List[float] = field(default_factory=list)
    downrange: List[float] = field(default_factory=list)
    downrange_ground: List[float] = field(default_factory=list)  # Ground track distance
    velocity: List[float] = field(default_factory=list)
    velocity_rel: List[float] = field(default_factory=list)
    velocity_horizontal: List[float] = field(default_factory=list)
    velocity_vertical: List[float] = field(default_factory=list)
    radial_velocity: List[float] = field(default_factory=list)
    horizontal_velocity_inertial: List[float] = field(default_factory=list)
    velocity_x: List[float] = field(default_factory=list)
    velocity_y: List[float] = field(default_factory=list)
    velocity_z: List[float] = field(default_factory=list)
    velocity_rel_x: List[float] = field(default_factory=list)
    velocity_rel_y: List[float] = field(default_factory=list)
    velocity_rel_z: List[float] = field(default_factory=list)
    mass: List[float] = field(default_factory=list)
    pitch_angle: List[float] = field(default_factory=list)
    attitude_error: List[float] = field(default_factory=list)
    torque_magnitude: List[float] = field(default_factory=list)
    position_x: List[float] = field(default_factory=list)
    position_y: List[float] = field(default_factory=list)
    position_z: List[float] = field(default_factory=list)
    omega_x: List[float] = field(default_factory=list)
    omega_y: List[float] = field(default_factory=list)
    omega_z: List[float] = field(default_factory=list)
    quaternion_norm: List[float] = field(default_factory=list)
    actual_pitch_angle: List[float] = field(default_factory=list)
    gamma_command_deg: List[float] = field(default_factory=list)
    gamma_actual_deg: List[float] = field(default_factory=list)
    velocity_tilt_deg: List[float] = field(default_factory=list)
    # Diagnostics: commanded vs actual
    commanded_thrust_x: List[float] = field(default_factory=list)
    commanded_thrust_y: List[float] = field(default_factory=list)
    commanded_thrust_z: List[float] = field(default_factory=list)
    commanded_quat_w: List[float] = field(default_factory=list)
    commanded_quat_x: List[float] = field(default_factory=list)
    commanded_quat_y: List[float] = field(default_factory=list)
    commanded_quat_z: List[float] = field(default_factory=list)
    actual_quat_w: List[float] = field(default_factory=list)
    actual_quat_x: List[float] = field(default_factory=list)
    actual_quat_y: List[float] = field(default_factory=list)
    actual_quat_z: List[float] = field(default_factory=list)
    inertial_thrust_x: List[float] = field(default_factory=list)
    inertial_thrust_y: List[float] = field(default_factory=list)
    inertial_thrust_z: List[float] = field(default_factory=list)
    prograde_x: List[float] = field(default_factory=list)
    prograde_y: List[float] = field(default_factory=list)
    prograde_z: List[float] = field(default_factory=list)
    flight_path_angle_deg: List[float] = field(default_factory=list)
    throttle: List[float] = field(default_factory=list)
    thrust_on: List[float] = field(default_factory=list)
    # Extended telemetry (Phase 0.3)
    mach_number: List[float] = field(default_factory=list)
    dynamic_pressure: List[float] = field(default_factory=list)          # Pa
    angle_of_attack_deg: List[float] = field(default_factory=list)      # deg
    q_alpha: List[float] = field(default_factory=list)                  # Pa·rad
    heating_rate: List[float] = field(default_factory=list)             # W/m²
    latitude_deg: List[float] = field(default_factory=list)
    longitude_deg: List[float] = field(default_factory=list)
    ignition_altitude_prediction_m: List[float] = field(default_factory=list)
    propellant_remaining_kg: List[float] = field(default_factory=list)
    phase_name: List[str] = field(default_factory=list)

    def append(self, state: State, guidance: dict, control: dict):
        """Log data from current timestep."""
        self.time.append(state.t)
        self.altitude.append(state.altitude / 1000)  # km
        self.velocity.append(state.speed)
        # Air-relative velocity
        v_rel = guidance.get('v_rel', state.v)  # guidance may stash v_rel
        v_rel = np.asarray(v_rel)
        self.velocity_rel.append(np.linalg.norm(v_rel))
        self.velocity_rel_x.append(v_rel[0])
        self.velocity_rel_y.append(v_rel[1])
        self.velocity_rel_z.append(v_rel[2])
        vertical = compute_local_vertical(state.r)
        v_vert = float(np.dot(v_rel, vertical))
        v_horiz_vec = v_rel - v_vert * vertical
        v_horiz = float(np.linalg.norm(v_horiz_vec))
        self.velocity_horizontal.append(v_horiz)
        self.velocity_vertical.append(v_vert)
        v_radial = float(np.dot(state.v, vertical))
        v_horiz_inertial_vec = state.v - v_radial * vertical
        self.radial_velocity.append(v_radial)
        self.horizontal_velocity_inertial.append(float(np.linalg.norm(v_horiz_inertial_vec)))
        self.velocity_x.append(state.v[0])
        self.velocity_y.append(state.v[1])
        self.velocity_z.append(state.v[2])
        self.mass.append(state.m)
        self.pitch_angle.append(np.degrees(guidance['pitch_angle']))
        self.attitude_error.append(control.get('error_degrees', 0.0))
        self.torque_magnitude.append(control.get('torque_magnitude', 0.0))
        self.position_x.append(state.r[0])
        self.position_y.append(state.r[1])
        self.position_z.append(state.r[2])
        self.omega_x.append(state.omega[0])
        self.omega_y.append(state.omega[1])
        self.omega_z.append(state.omega[2])
        self.quaternion_norm.append(np.linalg.norm(state.q))
        # Downrange ground distance (great-circle) from launch point
        r_hat = state.r / np.linalg.norm(state.r)
        r0_hat = C.INITIAL_POSITION / np.linalg.norm(C.INITIAL_POSITION)
        central_angle = np.clip(np.arccos(np.clip(np.dot(r_hat, r0_hat), -1.0, 1.0)), 0.0, np.pi)
        downrange = C.R_EARTH * central_angle
        self.downrange.append(downrange / 1000)  # km

        # Ground Track Downrange (accounting for Earth rotation)
        # Rotate ECI position (state.r) to ECEF by angle -omega*t
        theta = C.EARTH_ROTATION_RATE * state.t
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        x_ecef = state.r[0] * cos_t + state.r[1] * sin_t
        y_ecef = -state.r[0] * sin_t + state.r[1] * cos_t
        z_ecef = state.r[2]
        r_ecef = np.array([x_ecef, y_ecef, z_ecef])
        
        # Angle from initial position (assumed at R_EARTH, 0, 0 in ECEF)
        r_ecef_norm = np.linalg.norm(r_ecef)
        if r_ecef_norm > C.ZERO_TOLERANCE:
             r_ecef_hat = r_ecef / r_ecef_norm
             # r0_ecef is [1, 0, 0] since launch site rotates with Earth
             central_angle_ground = np.arccos(np.clip(r_ecef_hat[0], -1.0, 1.0))
             downrange_ground = C.R_EARTH * central_angle_ground
             self.downrange_ground.append(downrange_ground / 1000.0)
        else:
             self.downrange_ground.append(0.0)

        # Compute and log actual pitch angle (Body Z vs Local Vertical)
        body_z_inertial = rotate_vector_by_quaternion(C.BODY_Z_AXIS, state.q)
        cos_pitch = np.clip(np.dot(vertical, body_z_inertial), -1.0, 1.0)
        self.actual_pitch_angle.append(np.degrees(np.arccos(cos_pitch)))

        # Flight-path angles
        gamma_cmd_deg = np.degrees(float(guidance.get('gamma_angle', 0.0)))
        self.gamma_command_deg.append(gamma_cmd_deg)
        # Actual gamma from horizontal: atan2(v_vert, v_horiz)
        gamma_actual = np.degrees(np.arctan2(v_vert, max(v_horiz, C.ZERO_TOLERANCE)))
        self.gamma_actual_deg.append(gamma_actual)
        # Velocity tilt from vertical (0=vertical, 90=horizontal)
        tilt = np.degrees(np.arctan2(v_horiz, abs(v_vert)))
        self.velocity_tilt_deg.append(tilt)

        # Diagnostics: commanded thrust direction (unit), commanded quaternion, actual quaternion
        cmd_thrust_dir = guidance.get('thrust_direction', np.array([0.0, 0.0, 1.0]))
        cmd_quat = control.get('q_commanded', np.array([1.0, 0.0, 0.0, 0.0]))

        self.commanded_thrust_x.append(float(cmd_thrust_dir[0]))
        self.commanded_thrust_y.append(float(cmd_thrust_dir[1]))
        self.commanded_thrust_z.append(float(cmd_thrust_dir[2]))

        self.commanded_quat_w.append(float(cmd_quat[0]))
        self.commanded_quat_x.append(float(cmd_quat[1]))
        self.commanded_quat_y.append(float(cmd_quat[2]))
        self.commanded_quat_z.append(float(cmd_quat[3]))

        self.actual_quat_w.append(float(state.q[0]))
        self.actual_quat_x.append(float(state.q[1]))
        self.actual_quat_y.append(float(state.q[2]))
        self.actual_quat_z.append(float(state.q[3]))

        # Compute ACTUAL inertial thrust vector (N) from vehicle attitude, not commanded
        throttle_val = float(guidance.get('throttle', 1.0))
        thrust_on_val = bool(guidance.get('thrust_on', True))
        # Determine thrust magnitude based on phase (S1 vs S2 engine)
        phase_str = guidance.get('phase', '')
        is_s2_phase = phase_str in ('ORBIT_INSERTION', 'COAST_PHASE') and state.m < C.STAGE2_WET_MASS * 1.1
        thrust_mag_nominal = C.STAGE2_THRUST if is_s2_phase else C.THRUST_MAGNITUDE
        thrust_mag = thrust_mag_nominal if thrust_on_val else 0.0
        # Get actual thrust direction from vehicle quaternion
        actual_thrust_dir = rotate_vector_by_quaternion(C.BODY_Z_AXIS, state.q)
        thrust_vec = actual_thrust_dir * (throttle_val * thrust_mag)
        self.inertial_thrust_x.append(float(thrust_vec[0]))
        self.inertial_thrust_y.append(float(thrust_vec[1]))
        self.inertial_thrust_z.append(float(thrust_vec[2]))

        # Prograde/local horizontal
        prograde = guidance.get('local_horizontal', np.array([0.0, 0.0, 1.0]))
        self.prograde_x.append(float(prograde[0]))
        self.prograde_y.append(float(prograde[1]))
        self.prograde_z.append(float(prograde[2]))

        # Flight path angle (gamma) in degrees
        gamma = guidance.get('gamma_angle', np.radians(gamma_actual))
        self.flight_path_angle_deg.append(np.degrees(float(gamma)))

        # Throttle / thrust_on
        self.throttle.append(throttle_val)
        self.thrust_on.append(1.0 if thrust_on_val else 0.0)

        # ── Extended telemetry (Phase 0.3) ──
        alt_m = state.altitude
        T_atm, P_atm, rho_atm, a_atm = compute_atmosphere_properties(alt_m)

        # Mach number
        v_rel_mag = float(np.linalg.norm(v_rel))
        mach = v_rel_mag / a_atm if a_atm > 1.0 else 0.0
        self.mach_number.append(mach)

        # Dynamic pressure
        q_dyn = 0.5 * rho_atm * v_rel_mag ** 2
        self.dynamic_pressure.append(q_dyn)

        # Angle of attack (angle between body +Z and v_rel)
        body_z = rotate_vector_by_quaternion(C.BODY_Z_AXIS, state.q)
        if v_rel_mag > 10.0:
            v_rel_hat = v_rel / v_rel_mag
            cos_aoa = float(np.clip(np.dot(body_z, v_rel_hat), -1.0, 1.0))
            aoa_rad = float(np.arccos(cos_aoa))
        else:
            aoa_rad = 0.0
        self.angle_of_attack_deg.append(np.degrees(aoa_rad))

        # Q-alpha (structural load indicator)
        self.q_alpha.append(q_dyn * aoa_rad)

        # Aerodynamic heating rate (Sutton-Graves, placeholder — uses config later)
        # q_dot = k * sqrt(rho / r_nose) * v^3  (simplified)
        r_nose = C.REFERENCE_DIAMETER / 2.0  # nose radius ~ half diameter
        if rho_atm > 1e-12 and v_rel_mag > 100.0:
            q_heat = 1.7415e-4 * np.sqrt(rho_atm / max(r_nose, 0.01)) * v_rel_mag ** 3
        else:
            q_heat = 0.0
        self.heating_rate.append(q_heat)

        # Latitude / Longitude (approximate spherical)
        r_mag = float(np.linalg.norm(state.r))
        if r_mag > 1.0:
            lat = float(np.degrees(np.arcsin(np.clip(state.r[2] / r_mag, -1.0, 1.0))))
            lon = float(np.degrees(np.arctan2(state.r[1], state.r[0])))
            # Adjust for Earth rotation to get geographic longitude
            lon_geo = lon - np.degrees(C.EARTH_ROTATION_RATE * state.t)
            # Wrap to [-180, 180]
            lon_geo = ((lon_geo + 180.0) % 360.0) - 180.0
        else:
            lat = 0.0
            lon_geo = 0.0
        self.latitude_deg.append(lat)
        self.longitude_deg.append(lon_geo)
        self.ignition_altitude_prediction_m.append(
            float(guidance.get('ignition_altitude_prediction_m', 0.0))
        )
        self.propellant_remaining_kg.append(
            float(guidance.get('propellant_remaining_kg', 0.0))
        )

        # Phase name
        self.phase_name.append(guidance.get('phase', 'UNKNOWN'))

    def to_csv(self, filename: str):
        """Write logged diagnostics to CSV for offline analysis."""
        os.makedirs(os.path.dirname(filename) or '.', exist_ok=True)
        header = [
            'time', 'altitude_km', 'downrange_km', 'downrange_ground_km',
            'velocity_inertial', 'velocity_rel', 'vel_horiz', 'vel_vert',
            'radial_velocity_mps', 'horizontal_velocity_inertial_mps',
            'mass',
            'pos_x', 'pos_y', 'pos_z', 'vel_x', 'vel_y', 'vel_z',
            'vel_rel_x', 'vel_rel_y', 'vel_rel_z',
            'pitch_cmd_deg', 'pitch_actual_deg',
            'attitude_error_deg', 'torque_Nm',
            'qnorm',
            'gamma_cmd_deg', 'gamma_actual_deg', 'velocity_tilt_deg',
            'cmd_thrust_x', 'cmd_thrust_y', 'cmd_thrust_z',
            'cmd_quat_w', 'cmd_quat_x', 'cmd_quat_y', 'cmd_quat_z',
            'act_quat_w', 'act_quat_x', 'act_quat_y', 'act_quat_z',
            'inertial_thrust_x', 'inertial_thrust_y', 'inertial_thrust_z',
            'prograde_x', 'prograde_y', 'prograde_z',
            'flight_path_angle_deg', 'throttle', 'thrust_on',
            'mach', 'dynamic_pressure_Pa', 'aoa_deg', 'q_alpha_Pa_rad',
            'heating_rate_W_m2', 'latitude_deg', 'longitude_deg',
            'ignition_altitude_prediction_m', 'propellant_remaining_kg', 'phase'
        ]

        with open(filename, 'w', newline='') as fh:
            writer = csv.writer(fh)
            writer.writerow(header)
            for i in range(len(self.time)):
                row = [
                    self.time[i], self.altitude[i], self.downrange[i], self.downrange_ground[i],
                    self.velocity[i], self.velocity_rel[i], self.velocity_horizontal[i], self.velocity_vertical[i],
                    self.radial_velocity[i], self.horizontal_velocity_inertial[i],
                    self.mass[i],
                    self.position_x[i], self.position_y[i], self.position_z[i],
                    self.velocity_x[i], self.velocity_y[i], self.velocity_z[i],
                    self.velocity_rel_x[i], self.velocity_rel_y[i], self.velocity_rel_z[i],
                    self.pitch_angle[i], self.actual_pitch_angle[i],
                    self.attitude_error[i], self.torque_magnitude[i],
                    self.quaternion_norm[i],
                    self.gamma_command_deg[i], self.gamma_actual_deg[i], self.velocity_tilt_deg[i],
                    self.commanded_thrust_x[i], self.commanded_thrust_y[i], self.commanded_thrust_z[i],
                    self.commanded_quat_w[i], self.commanded_quat_x[i], self.commanded_quat_y[i], self.commanded_quat_z[i],
                    self.actual_quat_w[i], self.actual_quat_x[i], self.actual_quat_y[i], self.actual_quat_z[i],
                    self.inertial_thrust_x[i], self.inertial_thrust_y[i], self.inertial_thrust_z[i],
                    self.prograde_x[i], self.prograde_y[i], self.prograde_z[i],
                    self.flight_path_angle_deg[i], self.throttle[i], self.thrust_on[i],
                    self.mach_number[i], self.dynamic_pressure[i], self.angle_of_attack_deg[i],
                    self.q_alpha[i], self.heating_rate[i], self.latitude_deg[i], self.longitude_deg[i],
                    self.ignition_altitude_prediction_m[i], self.propellant_remaining_kg[i],
                    self.phase_name[i]
                ]
                writer.writerow(row)


def check_termination(state: State, max_time: float, mission_mgr: MissionManager = None) -> tuple:
    """
    Check if simulation should terminate.

    Termination criteria depend on vehicle type and mission phase:
    - ascent: Stops at apogee (legacy single-stage sim)
    - orbiter: Stops when orbit is achieved or S2 propellant exhausted
    - booster: Stops at touchdown

    Args:
        state: Current state
        max_time: Maximum allowed simulation time (s)
        mission_mgr: Mission Manager instance

    Returns:
        (should_terminate, reason) tuple
    """
    phase = mission_mgr.get_phase()

    # ===== ORBIT ACHIEVED (orbiter vehicle) =====
    if phase == MissionPhase.ORBIT_ACHIEVED:
        return True, "ORBIT ACHIEVED - Mission Complete"
    if phase == MissionPhase.ORBIT_FAILED:
        reason = getattr(mission_mgr, 'orbit_failure_reason', None) or "Target orbit not achieved"
        return True, f"ORBIT INSERTION FAILED - {reason}"

    # ===== BOOSTER LANDING =====
    if mission_mgr.vehicle_type == "booster":
        if phase == MissionPhase.BOOSTER_LANDING and state.altitude <= 0.0:
            cfg = mission_mgr.config
            v_touchdown_rel = float(np.linalg.norm(compute_relative_velocity(state.r, state.v)))
            target_site = target_landing_site_eci(
                state.t,
                cfg.booster_landing_target_downrange_km,
            )
            site_error_m = great_circle_distance_m(state.r, target_site)
            pad_hit = site_error_m <= cfg.booster_pad_tolerance_m

            if cfg.booster_enforce_pad_landing and not pad_hit:
                return True, (
                    f"OFFSITE LANDING - Missed pad by {site_error_m/1000.0:.2f} km "
                    f"at {v_touchdown_rel:.2f} m/s"
                )

            if v_touchdown_rel < 5.0:
                return True, (
                    f"LANDING SUCCESS - Touchdown at {v_touchdown_rel:.2f} m/s "
                    f"(site error {site_error_m:.0f} m)"
                )
            elif v_touchdown_rel < 50.0:
                return True, (
                    f"HARD LANDING - Touchdown at {v_touchdown_rel:.1f} m/s "
                    f"(site error {site_error_m:.0f} m)"
                )
            else:
                return True, (
                    f"CRASH LANDING - Impact at {v_touchdown_rel:.0f} m/s "
                    f"(site error {site_error_m:.0f} m)"
                )

    # ===== ASCENT-ONLY VEHICLE: stop at apogee (legacy mode) =====
    # "ascent" runs the full mission including orbit insertion
    # "ascent_only" is the legacy mode that stops at apogee
    if mission_mgr.vehicle_type == "ascent_only":
        vertical = compute_local_vertical(state.r)
        v_vert = float(np.dot(state.v, vertical))
        # Stop if we are falling back down (Apogee reached)
        if state.t > 100.0 and v_vert <= 0.0:
            return True, "Apogee Reached (v_vert <= 0)"

    # ===== MAX TIME =====
    if state.t >= max_time:
        return True, "Maximum simulation time reached"

    # ===== CRASH =====
    if state.altitude < C.CRASH_ALTITUDE_TOLERANCE:
        return True, "CRASH - Below Earth's surface"

    return False, None


def simulation_step(state: State, actuator: ActuatorState, mission_mgr: MissionManager,
                    dt: float, dry_mass: float = C.DRY_MASS, stage: int = 1,
                    vehicle_model: str = "stacked",
                    gs: GuidanceState = None,
                    config: SimulationConfig = None) -> tuple:
    """
    Execute one complete simulation timestep.

    Execution order (as per spec):
    1. Guidance → thrust direction
    2. Attitude command → quaternion
    3. Control torque
    4. (Force computation happens in integrator)
    5. (Rotational dynamics in integrator)
    6. (Quaternion integration in integrator)
    7. (Translational dynamics in integrator)
    8. (Mass update in integrator)

    Args:
        state: Current state
        actuator: Gimbal actuator state
        mission_mgr: Mission phase manager
        dt: Time step
        dry_mass: Current dry mass limit
        stage: Engine stage (1 = S1, 2 = S2)
        gs: Per-vehicle guidance state (created if None)
        config: Simulation configuration (uses default if None)

    Returns:
        (new_state, guidance_output, control_output, actuator, guidance_state) tuple
    """
    if gs is None:
        gs = create_guidance_state()
    if config is None:
        config = create_default_config()

    # Step 1: Guidance
    phase = mission_mgr.get_phase()

    if phase == MissionPhase.ASCENT:
        # S1 MECO mass = stacked dry + landing fuel reserve
        meco_mass = C.DRY_MASS + C.STAGE1_LANDING_FUEL_RESERVE
        guidance, gs = compute_guidance_output(state.r, state.v, state.t, state.m,
                                              meco_mass=meco_mass, gs=gs)
    elif phase in [MissionPhase.COAST, MissionPhase.STAGE_SEPARATION,
                   MissionPhase.S2_COAST_TO_APOGEE]:
        guidance, gs = compute_coast_guidance(state.r, state.v, state.t, state.m, gs=gs)
    elif phase == MissionPhase.ORBIT_INSERTION:
        guidance, gs = compute_orbit_insertion_guidance(state.r, state.v, state.t, state.m, gs=gs)
    elif phase in [MissionPhase.BOOSTER_FLIP, MissionPhase.BOOSTER_BOOSTBACK,
                   MissionPhase.BOOSTER_COAST, MissionPhase.BOOSTER_ENTRY,
                   MissionPhase.BOOSTER_LANDING]:
        guidance, gs = compute_booster_guidance(
            state.r, state.v, state.t, state.m, phase.name, gs=gs, config=config
        )
    else:
        # Default fallback (ORBIT_ACHIEVED, APOGEE_REACHED, etc.)
        guidance, gs = compute_coast_guidance(state.r, state.v, state.t, state.m, gs=gs)
        guidance['phase'] = phase.name

    desired_dir = guidance['thrust_direction']
    if vehicle_model == "booster":
        # Booster recovery needs rapid flips; do not throttle command slew through
        # ascent gimbal-rate limits.
        n_des = np.linalg.norm(desired_dir)
        thrust_dir_cmd = desired_dir / n_des if n_des > 1e-9 else compute_local_vertical(state.r)
        actuator = ActuatorState(thrust_dir=thrust_dir_cmd)
    else:
        actuator = update_actuator(actuator, desired_dir, dt)
        thrust_dir_cmd = actuator.thrust_dir

    # Step 2: Attitude control with gain scheduling
    # Compute current inertia for gain scaling so the controller maintains
    # consistent bandwidth (omega_n ~ 1 rad/s, zeta ~ 0.7) across all vehicle
    # configurations (S1 stacked at 540 t down to S2 dry at 17 t).
    I_tensor = compute_inertia_tensor(state.m, vehicle_model=vehicle_model)
    inertia_repr = I_tensor[0, 0]  # Ixx — representative pitch/yaw inertia

    # Stage-appropriate torque limit:
    # S1: 7.6 MN thrust * 3.7m arm * 0.7 leverage = ~20 MN·m (use 30 MN·m with margin)
    # S2: 981 kN thrust * 2.0m arm * 0.7 leverage = ~1.4 MN·m (use 2 MN·m with margin)
    if stage == 2:
        stage_max_torque = 2.0e6  # 2 MN*m - S2 gimbal + RCS authority
    elif vehicle_model == "booster":
        stage_max_torque = 1.2e8  # Recovery authority: gimbal + RCS + aero surfaces
    else:
        stage_max_torque = C.MAX_TORQUE  # 30 MN*m - S1 multi-engine gimbal
    control_inertia = inertia_repr
    if vehicle_model == "booster":
        control_inertia = 3.0 * inertia_repr

    if vehicle_model == "booster":
        q_cmd = compute_commanded_quaternion(thrust_dir_cmd)
        control = {
            'q_commanded': q_cmd,
            'error_axis': np.zeros(3),
            'error_angle': 0.0,
            'error_degrees': 0.0,
            'torque': np.zeros(3),
            'torque_magnitude': 0.0,
            'saturated': False,
        }
        integration_state = state.copy()
        integration_state.q = q_cmd
        integration_state.omega = np.zeros(3)
    else:
        control = compute_control_output(
            state.q, state.omega, thrust_dir_cmd,
            inertia=control_inertia,
            max_torque=stage_max_torque
        )
        integration_state = state

    requested_thrust_on = bool(guidance.get('thrust_on', False))
    throttle_cmd = float(guidance.get('throttle', 1.0))
    propellant_available = state.m > (dry_mass + 1e-6)
    thrust_active = requested_thrust_on and propellant_available and throttle_cmd > 0.0
    if not propellant_available:
        throttle_cmd = 0.0
    guidance['thrust_on'] = thrust_active
    guidance['throttle'] = throttle_cmd

    # Step 3: Integration (with correct engine stage parameters)
    new_state = integrate(
        integration_state, control['torque'], dt,
        thrust_on=thrust_active,
        method='rk4',
        throttle=throttle_cmd,
        dry_mass=dry_mass,
        stage=stage,
        vehicle_model=vehicle_model,
    )

    return new_state, guidance, control, actuator, gs


def run_simulation(initial_state: Optional[State] = None, dt: float = None, max_time: float = None,
                   verbose: bool = True, coast_to_apogee: bool = True, vehicle_type: str = "ascent",
                   config: SimulationConfig = None) -> tuple:
    """
    Run the complete multi-phase mission simulation.

    Supports three vehicle types:
    - "ascent": S1 ascent only, terminates at apogee (legacy mode)
    - "orbiter": Full mission — S1 ascent → coast → S2 orbit insertion
    - "booster": S1 booster recovery after separation

    Args:
        initial_state: Optional starting state. If None, starts from liftoff default.
        dt: Time step (default from config/constants). Overrides config.dt if given.
        max_time: Maximum simulation time. Overrides config.max_time if given.
        verbose: Print progress updates. Overrides config.verbose if explicitly passed.
        coast_to_apogee: If True, allow coasting after MECO until apogee
        vehicle_type: "ascent", "orbiter", or "booster" to determine mission logic
        config: SimulationConfig instance. If None a default is created.

    Returns:
        (final_state, log, termination_reason) tuple
    """
    if config is None:
        config = create_default_config()

    # Explicit kwargs override config values (backward compatibility)
    if dt is None:
        dt = config.dt
    if max_time is None:
        max_time = config.max_time

    # Initialize state
    if initial_state is not None:
        state = initial_state.copy()
    else:
        state = create_initial_state()

    # Create fresh per-vehicle guidance state (no module-level carry-over)
    gs = create_guidance_state()
    # Also reset module-level default for any legacy callers
    reset_guidance()

    log = SimulationLog()
    actuator = ActuatorState(thrust_dir=compute_local_vertical(state.r))
    mission_mgr = MissionManager(vehicle_type=vehicle_type, initial_mass=state.m, config=config)

    # Initialize energy tracking for conservation check
    E_prev = compute_total_energy(state.r, state.v, state.m)

    # Log startup info
    logger.info(f"Starting simulation: dt={dt}s, max_time={max_time}s, vehicle={vehicle_type}")
    logger.debug(f"Initial state: {state}")

    if verbose:
        print("\n" + "=" * 80)
        print(f"RLV MISSION SIMULATION    | dt={dt}s | T_max={max_time}s | vehicle={vehicle_type}")
        print("=" * 80)
        print(f"{'Time (s)':^10} | {'Alt (km)':^10} | {'Vel (m/s)':^10} | {'Mass (kg)':^12} | {'Phase':<20}")
        print("-" * 80)

    start_time = time.time()
    step_count = 0
    last_print_time = 0
    meco_time = None
    separation_mass_applied = False  # Flag for one-time mass drop event

    # ===== Vehicle Configuration =====
    # Engine stage (1 = S1 engines, 2 = S2 engine)
    # Switches from 1 to 2 after stage separation
    current_stage = 1
    current_vehicle_model = "stacked"

    # Determine initial dry mass based on vehicle type
    if vehicle_type == "booster":
        current_dry_mass = C.STAGE1_DRY_MASS
        current_vehicle_model = "booster"
    elif vehicle_type == "orbiter":
        # Orbiter starts as S2 vehicle (given post-separation state)
        current_dry_mass = C.STAGE2_DRY_MASS
        current_stage = 2
        current_vehicle_model = "orbiter"
    else:
        # Ascent: stacked vehicle, MECO with fuel reserve
        # During ascent, the integrator's dry_mass floor = DRY_MASS + LANDING_RESERVE
        current_dry_mass = C.DRY_MASS + C.STAGE1_LANDING_FUEL_RESERVE

    # Main simulation loop
    while True:
        # Record MECO time (first propellant exhaustion detection)
        if meco_time is None and is_propellant_exhausted(state.m, current_dry_mass):
            meco_time = state.t

        # Check termination
        should_terminate, reason = check_termination(state, max_time, mission_mgr)
        if should_terminate:
            logger.info(f"Simulation terminated: {reason}")
            if verbose:
                print(f"\nTermination: {reason}")
            return state, log, reason

        # Validate state
        try:
            validate_state(state, dry_mass=current_dry_mass)
        except ValidationError as e:
            logger.error(f"Validation failed: {e}")
            if verbose:
                print(f"\nValidation Error: {e}")
            reason = f"Validation failure: {e}"
            break

        # Update Mission Manager
        phase_before = mission_mgr.get_phase()
        mission_mgr.update(state, dt)
        phase_after = mission_mgr.get_phase()

        # ===== STAGE SEPARATION MASS EVENT =====
        # When transitioning past STAGE_SEPARATION into any S2 phase, drop S1 mass.
        # This is a discrete event: the booster (S1 dry mass + landing fuel) separates,
        # and the vehicle continues as S2 only.
        # Note: mission manager may jump from STAGE_SEPARATION directly to ORBIT_INSERTION
        # in a single update() call, so we check for any post-separation phase.
        s2_phases = {MissionPhase.S2_COAST_TO_APOGEE, MissionPhase.ORBIT_INSERTION,
                     MissionPhase.ORBIT_ACHIEVED, MissionPhase.ORBIT_FAILED}
        pre_sep_phases = {MissionPhase.ASCENT, MissionPhase.COAST, MissionPhase.STAGE_SEPARATION}
        if (not separation_mass_applied and
                phase_after in s2_phases and
                phase_before in pre_sep_phases):

            mass_before = state.m
            # After separation: S2 wet mass only (S1 structure + landing fuel go with booster)
            # Vehicle mass drops from ~192,000 kg to 120,000 kg
            s2_mass = C.STAGE2_WET_MASS

            # At separation, both stages were rigidly connected and rotating together.
            # The S2 inherits the same angular velocity as the combined stack.
            # Although the inertia drops dramatically, the S2 controller (with gain
            # scheduling) will adapt its gains to the new inertia automatically.
            # A small damping factor accounts for separation spring perturbation.
            omega_damped = state.omega * 0.5  # Moderate damping from separation springs

            state_dict = {
                'r': state.r.copy(),
                'v': state.v.copy(),
                'q': state.q.copy(),
                'omega': omega_damped,
                'm': s2_mass,
                't': state.t
            }
            state = State(**state_dict)

            # Switch to S2 engine parameters
            current_stage = 2
            current_dry_mass = C.STAGE2_DRY_MASS
            current_vehicle_model = "orbiter"
            separation_mass_applied = True

            logger.info(f"SEPARATION MASS EVENT at t={state.t:.2f}s: "
                       f"mass {mass_before:.0f} -> {state.m:.0f} kg "
                       f"(dropped S1 dry={C.STAGE1_DRY_MASS:.0f}kg + "
                       f"landing fuel={C.STAGE1_LANDING_FUEL_RESERVE:.0f}kg)")
            if verbose:
                print(f"  *** STAGE SEPARATION: {mass_before:.0f} -> {state.m:.0f} kg | "
                      f"Switched to S2 engine ({C.STAGE2_THRUST/1e3:.0f} kN, Isp={C.STAGE2_ISP_VAC}s)")

        # Execute timestep
        state, guidance, control, actuator, gs = simulation_step(
            state, actuator, mission_mgr, dt,
            dry_mass=current_dry_mass, stage=current_stage,
            vehicle_model=current_vehicle_model, gs=gs,
            config=config
        )

        # Abort if gamma tracking diverges badly (only during ascent)
        if mission_mgr.get_phase() == MissionPhase.ASCENT:
            gamma_err = abs(guidance.get('gamma_command_deg', 0.0) - guidance.get('gamma_measured_deg', 0.0))
            if state.t > 60.0 and gamma_err > 70.0:
                reason = f"Guidance divergence: |gamma_error|={gamma_err:.1f} deg"
                logger.error(reason)
                break

        # Log data
        log.append(state, guidance, control)

        # Periodically check energy and print status
        step_count += 1

        if step_count % 100 == 0:
            E_current = compute_total_energy(state.r, state.v, state.m)
            # Only validate energy if thrust is OFF (coasting)
            if not guidance['thrust_on']:
                _validate_energy(E_current, E_prev, dt * 100, state.t)
            E_prev = E_current

            if verbose and state.t - last_print_time >= 10.0:
                _print_status(state, guidance['phase'])
                last_print_time = state.t

    elapsed = time.time() - start_time

    # Log final results
    _log_completion(state, step_count, elapsed, verbose)

    return state, log, reason


def _validate_energy(E_current: float, E_prev: float, time_delta: float, t: float):
    """Helper to validate and log energy conservation."""
    energy_result = validate_energy_conservation(E_current, E_prev, time_delta)
    if not energy_result['valid']:
        logger.warning(f"Energy drift at t={t:.1f}s: "
                      f"relative_error={energy_result['relative_error']:.2e}, "
                      f"dE={energy_result['dE']:.2e} J")


def _print_status(state: State, phase: str):
    """Print a formatted status row."""
    # Format: Time | Alt | Vel | Mass | Phase
    msg = (f"{state.t:10.1f} | {state.altitude/1000:10.1f} | "
           f"{state.speed:10.1f} | {state.m:12.1f} | {phase:<15}")
    print(msg)
    logger.info(msg)


def _log_completion(state: State, steps: int, elapsed: float, verbose: bool):
    """Log and print comparison statistics."""
    logger.info(f"Simulation complete: {steps} steps in {elapsed:.2f}s")
    logger.info(f"Final state: alt={state.altitude/1000:.2f}km, v={state.speed:.1f}m/s")
    
    if verbose:
        print("-" * 80)
        print("SIMULATION COMPLETED")
        print("-" * 80)
        print(f"Final Time:     {state.t:.2f} s")
        print(f"Final Altitude: {state.altitude/1000:.2f} km")
        print(f"Final Velocity: {state.speed:.2f} m/s")
        print(f"Final Mass:     {state.m:.1f} kg")
        print("-" * 80)
        print(f"Steps:       {steps:,}")
        print(f"Wall Time:   {elapsed:.2f} s")
        print(f"Performance: {steps/elapsed:.0f} steps/s" if elapsed > 0 else "Performance: N/A")
        print("=" * 80)


@dataclass
class FullMissionResult:
    """Result of a full integrated mission with dual-vehicle tracking.

    Contains telemetry for all three mission segments:
      - ascent: S1+S2 stacked from liftoff to stage separation
      - orbiter: S2 from separation to orbit insertion (or timeout)
      - booster: S1 from separation to landing (or timeout)

    The ascent log covers t=[0, t_sep].
    The orbiter and booster logs cover t=[t_sep, t_end].
    """
    # Ascent phase (stacked vehicle)
    ascent_log: SimulationLog
    ascent_final_state: State
    ascent_reason: str
    separation_time: float

    # Orbiter (S2) post-separation
    orbiter_log: SimulationLog
    orbiter_final_state: State
    orbiter_reason: str

    # Booster (S1) post-separation
    booster_log: SimulationLog
    booster_final_state: State
    booster_reason: str


def run_full_mission(dt: float = None, max_time: float = None,
                     verbose: bool = True,
                     config: SimulationConfig = None) -> FullMissionResult:
    """
    Run a full integrated mission: S1 ascent → separation → dual tracking.

    After stage separation the simulation forks into two vehicles
    (orbiter and booster) running in lockstep on the same clock.
    Each vehicle has its own GuidanceState, MissionManager and telemetry log.

    Args:
        dt: Timestep (overrides config.dt if given)
        max_time: Maximum mission time (overrides config.max_time if given)
        verbose: Print status updates
        config: SimulationConfig (default created if None)

    Returns:
        FullMissionResult with telemetry for ascent, orbiter and booster.
    """
    if config is None:
        config = create_default_config()
    if dt is None:
        dt = config.dt
    if max_time is None:
        max_time = config.max_time

    # ═══════════════════════════════════════════════════════════════════════
    #  PHASE A: Stacked Ascent (S1 + S2)
    # ═══════════════════════════════════════════════════════════════════════
    if verbose:
        print("\n" + "=" * 90)
        print("FULL MISSION SIMULATION  (Ascent + Orbiter + Booster)")
        print("=" * 90)
        print("--- PHASE A: Stacked Ascent ---")

    state = create_initial_state()
    gs_ascent = create_guidance_state()
    reset_guidance()

    ascent_log = SimulationLog()
    actuator = ActuatorState(thrust_dir=compute_local_vertical(state.r))
    mission_mgr = MissionManager(vehicle_type="ascent", initial_mass=state.m, config=config)

    current_stage = 1
    current_dry_mass = C.DRY_MASS + C.STAGE1_LANDING_FUEL_RESERVE
    step_count = 0
    last_print_time = 0
    separation_time = None
    ascent_reason = "Stage separation"

    start_wall = time.time()

    while True:
        # Check for separation transition
        phase_before = mission_mgr.get_phase()
        mission_mgr.update(state, dt)
        phase_after = mission_mgr.get_phase()

        s2_phases = {MissionPhase.S2_COAST_TO_APOGEE, MissionPhase.ORBIT_INSERTION,
                     MissionPhase.ORBIT_ACHIEVED, MissionPhase.ORBIT_FAILED}
        pre_sep_phases = {MissionPhase.ASCENT, MissionPhase.COAST, MissionPhase.STAGE_SEPARATION}

        if phase_after in s2_phases and phase_before in pre_sep_phases:
            separation_time = state.t
            if verbose:
                print(f"\n  *** STAGE SEPARATION at t={separation_time:.2f}s "
                      f"| Alt={state.altitude/1000:.1f}km "
                      f"| V={state.speed:.0f}m/s ***\n")
            break

        # Max time / crash guard
        if state.t >= max_time:
            ascent_reason = "Maximum simulation time reached (no separation)"
            break
        if state.altitude < C.CRASH_ALTITUDE_TOLERANCE:
            ascent_reason = "CRASH during ascent"
            break

        # Execute timestep
        state, guid_out, ctrl_out, actuator, gs_ascent = simulation_step(
            state, actuator, mission_mgr, dt,
            dry_mass=current_dry_mass, stage=current_stage,
            vehicle_model="stacked", gs=gs_ascent,
            config=config
        )
        ascent_log.append(state, guid_out, ctrl_out)

        step_count += 1
        if verbose and state.t - last_print_time >= 10.0:
            _print_status(state, guid_out.get('phase', '?') if isinstance(guid_out, dict) else '?')
            last_print_time = state.t

    ascent_final = state.copy()

    if separation_time is None:
        # Never reached separation — return what we have
        empty_log = SimulationLog()
        return FullMissionResult(
            ascent_log=ascent_log, ascent_final_state=ascent_final,
            ascent_reason=ascent_reason, separation_time=state.t,
            orbiter_log=empty_log, orbiter_final_state=ascent_final,
            orbiter_reason="No separation",
            booster_log=empty_log, booster_final_state=ascent_final,
            booster_reason="No separation",
        )

    # ═══════════════════════════════════════════════════════════════════════
    #  Fork: Create Orbiter (S2) and Booster (S1) states
    # ═══════════════════════════════════════════════════════════════════════

    # --- Orbiter: S2 vehicle ---
    omega_orbiter = state.omega * 0.5  # Separation spring damping
    orbiter_state = State(
        r=state.r.copy(), v=state.v.copy(), q=state.q.copy(),
        omega=omega_orbiter, m=C.STAGE2_WET_MASS, t=state.t
    )
    gs_orbiter = create_guidance_state()
    # Carry over last ascent direction for smooth transition
    gs_orbiter.last_ascent_direction = gs_ascent.last_ascent_direction
    orbiter_actuator = ActuatorState(thrust_dir=actuator.thrust_dir.copy())
    orbiter_mgr = MissionManager(vehicle_type="orbiter", initial_mass=C.STAGE2_WET_MASS, config=config)
    # Advance mission manager to post-separation state
    orbiter_mgr.update(orbiter_state, dt)

    # --- Booster: S1 vehicle ---
    # Booster gets S1 dry mass + landing fuel reserve
    booster_mass = C.STAGE1_DRY_MASS + C.STAGE1_LANDING_FUEL_RESERVE
    # Apply small separation delta-V (pushes booster slightly retrograde)
    v_hat = state.v / max(np.linalg.norm(state.v), 1.0)
    sep_dv = 1.5  # m/s separation impulse
    booster_v = state.v - sep_dv * v_hat  # Slight retrograde kick
    omega_booster = state.omega * 0.3  # Booster gets less angular rate
    booster_state = State(
        r=state.r.copy(), v=booster_v, q=state.q.copy(),
        omega=omega_booster, m=booster_mass, t=state.t
    )
    gs_booster = create_guidance_state()
    booster_actuator = ActuatorState(thrust_dir=actuator.thrust_dir.copy())
    booster_mgr = MissionManager(vehicle_type="booster", initial_mass=booster_mass, config=config)
    # Force booster into BOOSTER_FLIP phase
    booster_mgr.update(booster_state, dt)

    orbiter_log = SimulationLog()
    booster_log = SimulationLog()

    orbiter_done = False
    booster_done = False
    orbiter_reason = "Running"
    booster_reason = "Running"

    # ═══════════════════════════════════════════════════════════════════════
    #  PHASE B: Dual-Vehicle Lockstep Simulation
    # ═══════════════════════════════════════════════════════════════════════
    if verbose:
        print("--- PHASE B: Dual Vehicle Tracking (Orbiter + Booster) ---")
        print(f"{'Time (s)':^10} | {'Orb Alt km':^10} | {'Orb V m/s':^10} | "
              f"{'Bst Alt km':^10} | {'Bst V m/s':^10} | {'Phase O':^18} | {'Phase B':^18}")
        print("-" * 110)

    dual_step = 0

    while not (orbiter_done and booster_done):
        # ── Orbiter step ──
        if not orbiter_done:
            orb_term, orb_reason = check_termination(orbiter_state, max_time, orbiter_mgr)
            if orb_term:
                orbiter_done = True
                orbiter_reason = orb_reason
            else:
                orbiter_mgr.update(orbiter_state, dt)
                orbiter_state, orb_guid, orb_ctrl, orbiter_actuator, gs_orbiter = simulation_step(
                    orbiter_state, orbiter_actuator, orbiter_mgr, dt,
                    dry_mass=C.STAGE2_DRY_MASS, stage=2,
                    vehicle_model="orbiter", gs=gs_orbiter,
                    config=config
                )
                orbiter_log.append(orbiter_state, orb_guid, orb_ctrl)

        # ── Booster step ──
        if not booster_done:
            bst_term, bst_reason = check_termination(booster_state, max_time, booster_mgr)
            if bst_term:
                booster_done = True
                booster_reason = bst_reason
            else:
                booster_mgr.update(booster_state, dt)
                booster_state, bst_guid, bst_ctrl, booster_actuator, gs_booster = simulation_step(
                    booster_state, booster_actuator, booster_mgr, dt,
                    dry_mass=C.STAGE1_DRY_MASS, stage=1,
                    vehicle_model="booster", gs=gs_booster,
                    config=config
                )
                booster_log.append(booster_state, bst_guid, bst_ctrl)

        dual_step += 1
        step_count += 1

        # Periodic status
        if verbose and dual_step % 2000 == 0:
            orb_alt = orbiter_state.altitude / 1000 if not orbiter_done else -1
            orb_vel = orbiter_state.speed if not orbiter_done else -1
            bst_alt = booster_state.altitude / 1000 if not booster_done else -1
            bst_vel = booster_state.speed if not booster_done else -1
            orb_phase = orb_guid.get('phase', '?') if not orbiter_done else orbiter_reason[:18]
            bst_phase = bst_guid.get('phase', '?') if not booster_done else booster_reason[:18]
            t_now = max(
                orbiter_state.t if not orbiter_done else 0,
                booster_state.t if not booster_done else 0
            )
            print(f"{t_now:10.1f} | {orb_alt:10.1f} | {orb_vel:10.1f} | "
                  f"{bst_alt:10.1f} | {bst_vel:10.1f} | {orb_phase:^18} | {bst_phase:^18}")

    elapsed = time.time() - start_wall

    if verbose:
        print("\n" + "=" * 90)
        print("FULL MISSION SUMMARY")
        print("=" * 90)
        print(f"Separation:    t={separation_time:.2f}s")
        print(f"Orbiter:       {orbiter_reason}")
        print(f"  Final Alt:   {orbiter_state.altitude/1000:.1f} km | V={orbiter_state.speed:.0f} m/s")
        print(f"Booster:       {booster_reason}")
        booster_v_display = float(np.linalg.norm(compute_relative_velocity(booster_state.r, booster_state.v)))
        print(f"  Final Alt:   {booster_state.altitude/1000:.1f} km | V={booster_v_display:.0f} m/s (surface-relative)")
        print(f"Total Steps:   {step_count:,} | Wall Time: {elapsed:.2f}s")
        print("=" * 90)

    return FullMissionResult(
        ascent_log=ascent_log,
        ascent_final_state=ascent_final,
        ascent_reason=ascent_reason,
        separation_time=separation_time,
        orbiter_log=orbiter_log,
        orbiter_final_state=orbiter_state,
        orbiter_reason=orbiter_reason,
        booster_log=booster_log,
        booster_final_state=booster_state,
        booster_reason=booster_reason,
    )


if __name__ == "__main__":
    run_simulation()

