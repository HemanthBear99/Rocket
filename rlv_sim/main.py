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
from typing import List

import numpy as np

from . import constants as C
from .state import State, create_initial_state
from .guidance import compute_guidance_output
from .control import compute_control_output
from .integrators import integrate
from .validation import validate_state, ValidationError, compute_total_energy, validate_energy_conservation
from .mass import is_propellant_exhausted
from .frames import rotate_vector_by_quaternion
from .guidance import compute_local_vertical
from .actuator import ActuatorState, update_actuator

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
        thrust_mag = C.THRUST_MAGNITUDE if thrust_on_val else 0.0
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

    def to_csv(self, filename: str):
        """Write logged diagnostics to CSV for offline analysis."""
        os.makedirs(os.path.dirname(filename) or '.', exist_ok=True)
        header = [
            'time', 'altitude_km', 'downrange_km', 'downrange_ground_km',
            'velocity_inertial', 'velocity_rel', 'vel_horiz', 'vel_vert',
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
            'flight_path_angle_deg', 'throttle', 'thrust_on'
        ]

        with open(filename, 'w', newline='') as fh:
            writer = csv.writer(fh)
            writer.writerow(header)
            for i in range(len(self.time)):
                row = [
                    self.time[i], self.altitude[i], self.downrange[i], self.downrange_ground[i],
                    self.velocity[i], self.velocity_rel[i], self.velocity_horizontal[i], self.velocity_vertical[i],
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
                    self.flight_path_angle_deg[i], self.throttle[i], self.thrust_on[i]
                ]
                writer.writerow(row)


def check_termination(state: State, max_time: float, meco_time: float = None,
                      coast_to_apogee: bool = True) -> tuple:
    """
    Check if simulation should terminate.
    
    Args:
        state: Current state
        max_time: Maximum allowed simulation time (s)
        meco_time: Time when propellant was exhausted (for coast logic)
        coast_to_apogee: If True, allow coasting after MECO until apogee
        
    Returns:
        (should_terminate, reason) tuple
    """
    # Check propellant exhaustion
    if is_propellant_exhausted(state.m):
        if not coast_to_apogee:
            return True, "MECO - Propellant exhausted"
        # During coast, stop at apogee (vertical velocity <= 0) or after 15s of coast
        if meco_time is None:
            meco_time = state.t
        from .guidance import compute_local_vertical
        vertical = compute_local_vertical(state.r)
        v_vert = float(np.dot(state.v, vertical))
        if v_vert <= 0.0 or (state.t - meco_time) > 15.0:
            return True, "Coast complete after MECO"

    # Target energy/velocity cutoff
    if state.m > C.DRY_MASS and state.altitude >= C.TARGET_ALTITUDE and state.speed >= C.TARGET_SPEED:
        return True, "MECO - Target conditions achieved"
    
    # Check maximum time
    if state.t >= max_time:
        return True, "Maximum simulation time reached"
    
    # Check if crashed (altitude below tolerance)
    if state.altitude < C.CRASH_ALTITUDE_TOLERANCE:
        return True, "CRASH - Below Earth's surface"
    
    return False, None


def simulation_step(state: State, actuator: ActuatorState, dt: float) -> tuple:
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
        dt: Time step
        
    Returns:
        (new_state, guidance_output, control_output) tuple
    """
    # Step 1: Guidance
    guidance = compute_guidance_output(state.r, state.v, state.t, state.m)
    desired_dir = guidance['thrust_direction']
    actuator = update_actuator(actuator, desired_dir, dt)
    thrust_dir_cmd = actuator.thrust_dir
    
    # Step 2: Attitude control
    control = compute_control_output(
        state.q, state.omega, thrust_dir_cmd
    )
    
    # Step 3: Integration
    new_state = integrate(
        state, control['torque'], dt,
        thrust_on=guidance['thrust_on'],
        method='rk4',
        throttle=guidance.get('throttle', 1.0)
    )
    
    return new_state, guidance, control, actuator


def run_simulation(dt: float = None, max_time: float = None,
                   verbose: bool = True, coast_to_apogee: bool = True) -> tuple:
    """
    Run the complete Phase-I ascent simulation.
    
    Args:
        dt: Time step (default from constants)
        max_time: Maximum simulation time (default from constants)
        verbose: Print progress updates
        
    Returns:
        (final_state, log, termination_reason) tuple
    """
    if dt is None:
        dt = C.DT
    if max_time is None:
        max_time = C.MAX_TIME
    
    # Initialize
    state = create_initial_state()
    log = SimulationLog()
    actuator = ActuatorState(thrust_dir=compute_local_vertical(state.r))
    
    # [FIX #3] Initialize energy tracking for conservation check
    E_prev = compute_total_energy(state.r, state.v, state.m)
    
    # Log startup info
    logger.info(f"Starting simulation: dt={dt}s, max_time={max_time}s")
    logger.debug(f"Initial state: {state}")
    
    if verbose:
        print("\n" + "=" * 80)
        print(f"RLV PHASE-I SIMULATION    | dt={dt}s | T_max={max_time}s")
        print("=" * 80)
        print(f"{'Time (s)':^10} | {'Alt (km)':^10} | {'Vel (m/s)':^10} | {'Mass (kg)':^12} | {'Phase':<15}")
        print("-" * 80)
    
    start_time = time.time()
    step_count = 0
    last_print_time = 0
    meco_time = None
    
    # Main simulation loop
    while True:
        # Record MECO time (first dry-mass detection)
        if meco_time is None and is_propellant_exhausted(state.m):
            meco_time = state.t

        # Check termination
        terminate, reason = check_termination(state, max_time, meco_time, coast_to_apogee)
        if terminate:
            logger.info(f"Simulation terminated: {reason}")
            if verbose:
                print(f"\nTermination: {reason}")
            break
        
        # Validate state
        try:
            validate_state(state)
        except ValidationError as e:
            logger.error(f"Validation failed: {e}")
            if verbose:
                print(f"\nValidation Error: {e}")
            reason = f"Validation failure: {e}"
            break
        
        # Execute timestep
        state, guidance, control, actuator = simulation_step(state, actuator, dt)

        # Abort if gamma tracking diverges badly after initial 5s
        # Optional gamma divergence check (relaxed)
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


if __name__ == "__main__":
    run_simulation()
