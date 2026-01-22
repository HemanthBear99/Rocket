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
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from . import constants as C
from .state import State, create_initial_state
from .guidance import compute_guidance_output
from .control import compute_control_torque, compute_control_output
from .integrators import integrate
from .validation import validate_state, ValidationError
from .mass import is_propellant_exhausted
from .types import GuidanceOutput, ControlOutput

# Configure module logger
logger = logging.getLogger(__name__)


@dataclass
class SimulationLog:
    """Container for logged simulation data."""
    time: List[float] = field(default_factory=list)
    altitude: List[float] = field(default_factory=list)
    velocity: List[float] = field(default_factory=list)
    velocity_x: List[float] = field(default_factory=list)
    velocity_y: List[float] = field(default_factory=list)
    velocity_z: List[float] = field(default_factory=list)
    mass: List[float] = field(default_factory=list)
    pitch_angle: List[float] = field(default_factory=list)
    attitude_error: List[float] = field(default_factory=list)
    torque_magnitude: List[float] = field(default_factory=list)
    position_x: List[float] = field(default_factory=list)
    position_y: List[float] = field(default_factory=list)
    position_z: List[float] = field(default_factory=list)
    quaternion_norm: List[float] = field(default_factory=list)
    
    def append(self, state: State, guidance: dict, control: dict):
        """Log data from current timestep."""
        self.time.append(state.t)
        self.altitude.append(state.altitude / 1000)  # km
        self.velocity.append(state.speed)
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


def check_termination(state: State) -> tuple:
    """
    Check if simulation should terminate.
    
    Args:
        state: Current state
        
    Returns:
        (should_terminate, reason) tuple
    """
    # Check propellant exhaustion
    if is_propellant_exhausted(state.m):
        return True, "MECO - Propellant exhausted"
    
    # Check maximum time
    if state.t >= C.MAX_TIME:
        return True, "Maximum simulation time reached"
    
    # Check if crashed (altitude < 0)
    if state.altitude < -1000:  # Allow some margin for numerical errors
        return True, "CRASH - Below Earth's surface"
    
    return False, None


def simulation_step(state: State, dt: float) -> tuple:
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
    
    # Step 2-3: Attitude control
    control = compute_control_output(
        state.q, state.omega, guidance['thrust_direction']
    )
    torque = control['torque']
    
    # Steps 4-8: Integration (handles forces, dynamics, quaternion, mass)
    new_state = integrate(
        state, torque, dt,
        thrust_on=guidance['thrust_on'],
        method='rk4'
    )
    
    return new_state, guidance, control


def run_simulation(dt: float = None, max_time: float = None,
                   verbose: bool = True) -> tuple:
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
    
    # Log startup info
    logger.info(f"Starting simulation: dt={dt}s, max_time={max_time}s")
    logger.debug(f"Initial state: {state}")
    
    if verbose:
        print("=" * 60)
        print("RLV Phase-I Ascent Simulation")
        print("=" * 60)
        print(f"Initial state: {state}")
        print(f"Time step: {dt} s")
        print(f"Max time: {max_time} s")
        print("-" * 60)
    
    start_time = time.time()
    step_count = 0
    last_print_time = 0
    
    # Main simulation loop
    while True:
        # Check termination
        terminate, reason = check_termination(state)
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
        state, guidance, control = simulation_step(state, dt)
        
        # Log data
        log.append(state, guidance, control)
        
        step_count += 1
        
        # Progress update every 10 seconds of sim time
        if verbose and state.t - last_print_time >= 10.0:
            print(f"  t={state.t:6.1f}s | alt={state.altitude/1000:6.1f}km | "
                  f"v={state.speed:7.1f}m/s | m={state.m:7.1f}kg | "
                  f"phase={guidance['phase']}")
            last_print_time = state.t
    
    elapsed = time.time() - start_time
    
    # Log final results
    logger.info(f"Simulation complete: {step_count} steps in {elapsed:.2f}s")
    logger.info(f"Final state: alt={state.altitude/1000:.2f}km, v={state.speed:.1f}m/s")
    
    if verbose:
        print("-" * 60)
        print(f"Final state: {state}")
        print(f"Steps: {step_count:,}")
        print(f"Wall time: {elapsed:.2f} s")
        print(f"Performance: {step_count/elapsed:.0f} steps/s")
        print("=" * 60)
    
    return state, log, reason


if __name__ == "__main__":
    run_simulation()
