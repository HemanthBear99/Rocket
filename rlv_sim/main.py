"""
RLV Phase-I Ascent Simulation - Main Entry Point

This module implements the main simulation loop with:
- Single continuous simulation
- Correct execution order per timestep
- Data logging
- Visualization
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import List, Optional
import time

from . import constants as C
from .state import State, create_initial_state
from .guidance import compute_guidance_output
from .control import compute_control_torque
from .integrators import integrate
from .validation import validate_state, ValidationError
from .mass import is_propellant_exhausted


@dataclass
class SimulationLog:
    """Container for logged simulation data."""
    time: List[float] = field(default_factory=list)
    altitude: List[float] = field(default_factory=list)
    velocity: List[float] = field(default_factory=list)
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
    from .control import compute_control_output
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
            if verbose:
                print(f"\nTermination: {reason}")
            break
        
        # Validate state
        try:
            validate_state(state)
        except ValidationError as e:
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
    
    if verbose:
        print("-" * 60)
        print(f"Final state: {state}")
        print(f"Steps: {step_count:,}")
        print(f"Wall time: {elapsed:.2f} s")
        print(f"Performance: {step_count/elapsed:.0f} steps/s")
        print("=" * 60)
    
    return state, log, reason


def plot_results(log: SimulationLog, save_path: Optional[str] = None):
    """
    Generate standard plots of simulation results.
    
    Args:
        log: Simulation log data
        save_path: Optional path to save figure
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('RLV Phase-I Ascent Simulation Results', fontsize=14)
    
    # Altitude vs Time
    ax = axes[0, 0]
    ax.plot(log.time, log.altitude, 'b-')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Altitude (km)')
    ax.set_title('Altitude Profile')
    ax.grid(True, alpha=0.3)
    
    # Velocity vs Time
    ax = axes[0, 1]
    ax.plot(log.time, log.velocity, 'r-')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Velocity (m/s)')
    ax.set_title('Velocity Profile')
    ax.grid(True, alpha=0.3)
    
    # Mass vs Time
    ax = axes[0, 2]
    ax.plot(log.time, log.mass, 'g-')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Mass (kg)')
    ax.set_title('Mass Profile')
    ax.grid(True, alpha=0.3)
    
    # Pitch Angle vs Time
    ax = axes[1, 0]
    ax.plot(log.time, log.pitch_angle, 'm-')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Pitch Angle (deg)')
    ax.set_title('Guidance Pitch Angle')
    ax.grid(True, alpha=0.3)
    
    # Attitude Error vs Time
    ax = axes[1, 1]
    ax.plot(log.time, log.attitude_error, 'c-')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Attitude Error (deg)')
    ax.set_title('Attitude Tracking Error')
    ax.grid(True, alpha=0.3)
    
    # Quaternion Norm vs Time (validation)
    ax = axes[1, 2]
    ax.plot(log.time, log.quaternion_norm, 'k-')
    ax.axhline(y=1.0, color='r', linestyle='--', alpha=0.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Quaternion Norm')
    ax.set_title('Quaternion Norm (should = 1.0)')
    ax.set_ylim([0.999, 1.001])
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Figure saved to: {save_path}")
    
    plt.show()


def plot_trajectory_3d(log: SimulationLog, save_path: Optional[str] = None):
    """
    Plot 3D trajectory.
    
    Args:
        log: Simulation log data
        save_path: Optional path to save figure
    """
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Convert to km
    x = np.array(log.position_x) / 1000
    y = np.array(log.position_y) / 1000
    z = np.array(log.position_z) / 1000
    
    # Plot trajectory
    ax.plot(x, y, z, 'b-', linewidth=1)
    ax.scatter(x[0], y[0], z[0], c='g', s=100, marker='o', label='Start')
    ax.scatter(x[-1], y[-1], z[-1], c='r', s=100, marker='x', label='End')
    
    # Plot Earth sphere (simplified)
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    R = C.R_EARTH / 1000
    xe = R * np.outer(np.cos(u), np.sin(v))
    ye = R * np.outer(np.sin(u), np.sin(v))
    ze = R * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(xe, ye, ze, alpha=0.3, color='blue')
    
    ax.set_xlabel('X (km)')
    ax.set_ylabel('Y (km)')
    ax.set_zlabel('Z (km)')
    ax.set_title('3D Trajectory')
    ax.legend()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Figure saved to: {save_path}")
    
    plt.show()


def main():
    """Main entry point."""
    # Run simulation
    final_state, log, reason = run_simulation(verbose=True)
    
    # Plot results
    if len(log.time) > 0:
        plot_results(log)
        plot_trajectory_3d(log)
    
    return final_state, log, reason


if __name__ == "__main__":
    main()
