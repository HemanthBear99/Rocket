"""
Reusable Launch Vehicle (RLV) Phase-I Ascent Trajectory Visualization Module.

This module provides professional, publication-quality plotting functionality 
for RLV ascent simulation data. All plots follow aerospace industry standards
with proper labeling, units, and physics-based annotations.

Author: Generated for Academic Use
Date: 2026-02-02
"""

import os
import sys
from pathlib import Path
from typing import List, Tuple, Optional, NamedTuple
from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for batch processing
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from rlv_sim.main import run_simulation
from rlv_sim import constants as C


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class TrajectoryData:
    """Container for processed trajectory data used in plotting.

    Attributes:
        time: Time array in seconds
        altitude: Altitude array in kilometers
        velocity: Inertial velocity magnitude in m/s
        velocity_rel: Relative (airspeed) velocity magnitude in m/s
        mass: Vehicle mass in kg
        pitch_angle: Commanded pitch angle from vertical in degrees
        actual_pitch: Actual body pitch angle from vertical in degrees
        attitude_error: Attitude tracking error in degrees
        torque: Control torque magnitude in N·m
        gamma_cmd: Commanded flight path angle from horizontal in degrees
        gamma_actual: Actual flight path angle from horizontal in degrees
        gamma_rel: Computed relative flight path angle from horizontal in degrees
        position: Position vector array [n x 3] in meters (ECI)
        velocity_vec: Velocity vector array [n x 3] in m/s (ECI)
        velocity_rel_vec: Relative velocity vector array [n x 3] in m/s
        downrange: Downrange distance in kilometers
        dynamic_pressure: Dynamic pressure in Pascals
        thrust_force: Actual thrust magnitude in N
        # Extended telemetry for research-grade plots
        omega_x: Angular velocity body X component (rad/s)
        omega_y: Angular velocity body Y component (rad/s)
        omega_z: Angular velocity body Z component (rad/s)
        quaternion_norm: Quaternion norm history (should be 1.0)
        velocity_horizontal: Horizontal velocity component (m/s)
        velocity_vertical: Vertical velocity component (m/s)
        velocity_x: ECI velocity X (m/s)
        velocity_y: ECI velocity Y (m/s)
        velocity_z: ECI velocity Z (m/s)
        throttle: Throttle setting (0.0 to 1.0)
        thrust_on: Thrust active flag (0 or 1)
        commanded_quat: Commanded quaternion [n x 4]
        actual_quat: Actual quaternion [n x 4]
        mach_number: Mach number
        temperature: Atmospheric temperature (K)
        pressure: Atmospheric pressure (Pa)
        density: Atmospheric density (kg/m^3)
        speed_of_sound: Speed of sound (m/s)
        downrange_ground: Ground-track downrange (km)
    """
    time: np.ndarray
    altitude: np.ndarray
    velocity: np.ndarray
    velocity_rel: np.ndarray
    mass: np.ndarray
    pitch_angle: np.ndarray
    actual_pitch: np.ndarray
    attitude_error: np.ndarray
    torque: np.ndarray
    gamma_cmd: np.ndarray
    gamma_actual: np.ndarray
    gamma_rel: np.ndarray
    position: np.ndarray
    velocity_vec: np.ndarray
    velocity_rel_vec: np.ndarray
    downrange: np.ndarray
    dynamic_pressure: np.ndarray
    thrust_force: np.ndarray  # Actual thrust magnitude in N
    # Extended fields
    omega_x: np.ndarray = None
    omega_y: np.ndarray = None
    omega_z: np.ndarray = None
    quaternion_norm: np.ndarray = None
    velocity_horizontal: np.ndarray = None
    velocity_vertical: np.ndarray = None
    velocity_x: np.ndarray = None
    velocity_y: np.ndarray = None
    velocity_z: np.ndarray = None
    throttle: np.ndarray = None
    thrust_on: np.ndarray = None
    commanded_quat: np.ndarray = None
    actual_quat: np.ndarray = None
    mach_number: np.ndarray = None
    temperature: np.ndarray = None
    pressure: np.ndarray = None
    density: np.ndarray = None
    speed_of_sound: np.ndarray = None
    downrange_ground: np.ndarray = None


# =============================================================================
# Configuration
# =============================================================================

def configure_plot_style() -> None:
    """Configure matplotlib for professional aerospace plots.
    
    Sets publication-quality defaults suitable for academic papers
    and technical reports.
    """
    plt.rcParams.update({
        'figure.figsize': (10, 6),
        'figure.dpi': 300,  # High resolution for publications
        'savefig.dpi': 300,
        'axes.grid': True,
        'axes.axisbelow': True,
        'grid.alpha': 0.3,
        'grid.linestyle': '-',
        'grid.linewidth': 0.5,
        'font.size': 11,
        'axes.titlesize': 13,
        'axes.labelsize': 12,
        'legend.fontsize': 10,
        'legend.framealpha': 0.95,
        'legend.edgecolor': 'gray',
        'lines.linewidth': 1.8,
        'lines.markersize': 6,
        'axes.linewidth': 0.8,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
    })


# =============================================================================
# Data Processing
# =============================================================================

def extract_log_data(log) -> TrajectoryData:
    """Extract and process simulation log data for plotting.
    
    Args:
        log: Simulation log object with trajectory data
        
    Returns:
        TrajectoryData object with processed arrays
        
    Raises:
        AttributeError: If required log attributes are missing
    """
    # Extract raw arrays
    time = np.array(log.time)
    altitude = np.array(log.altitude)  # km
    velocity = np.array(log.velocity)  # m/s (inertial)
    mass = np.array(log.mass)  # kg
    pitch_angle = np.array(log.pitch_angle)  # deg from vertical
    attitude_error = np.array(log.attitude_error)  # deg
    torque = np.array(log.torque_magnitude)  # N·m
    
    # Position and velocity vectors
    pos_x = np.array(log.position_x)
    pos_y = np.array(log.position_y)
    pos_z = np.array(log.position_z)
    vel_x = np.array(log.velocity_x)
    vel_y = np.array(log.velocity_y)
    vel_z = np.array(log.velocity_z)
    
    # Thrust actual
    th_x = np.array(log.inertial_thrust_x)
    th_y = np.array(log.inertial_thrust_y)
    th_z = np.array(log.inertial_thrust_z)
    thrust_vec = np.column_stack((th_x, th_y, th_z))
    thrust_mag = np.linalg.norm(thrust_vec, axis=1)
    
    position = np.column_stack((pos_x, pos_y, pos_z))
    velocity_vec = np.column_stack((vel_x, vel_y, vel_z))
    
    # Relative velocity (accounting for Earth rotation AND wind)
    from rlv_sim.utils import compute_relative_velocity
    
    # Vectorized computation of relative velocity
    # We need to loop or map because compute_relative_velocity expects single vectors
    # or we can rely on numpy broadcasting if r/v are arrays.
    # rlv_sim.utils.compute_relative_velocity uses np.cross and _wind_vector.
    # _wind_vector uses norm(r). If r is (N,3), norm(r) is (N,).
    # Let's verify if utils supports broadcasting.
    # _wind_vector: alt = np.linalg.norm(r) - ... -> Returns scalar if r is 1D.
    # So we probably need to list comp it or update utils.
    # For safety in plot script, list comp is fine.
    
    velocity_rel_vec = np.array([
        compute_relative_velocity(p, v) 
        for p, v in zip(position, velocity_vec)
    ])
    velocity_rel = np.linalg.norm(velocity_rel_vec, axis=1)
    
    # Flight path angle calculations
    r_mag = np.linalg.norm(position, axis=1)
    r_hat = position / r_mag[:, np.newaxis]
    
    # Relative flight path angle (from horizontal)
    v_rel_radial = np.sum(velocity_rel_vec * r_hat, axis=1)
    sin_gamma_rel = np.clip(v_rel_radial / np.maximum(velocity_rel, 1.0), -1.0, 1.0)
    gamma_rel = np.degrees(np.arcsin(sin_gamma_rel))
    
    # Commanded and actual gamma
    gamma_cmd = np.array(getattr(log, 'gamma_command_deg', gamma_rel))
    gamma_actual = np.array(getattr(log, 'gamma_actual_deg', gamma_rel))
    
    # Downrange calculation
    downrange = np.sqrt((pos_x - pos_x[0])**2 + (pos_y - pos_y[0])**2) / 1000.0
    
    # Dynamic pressure
    from rlv_sim.forces import compute_atmosphere_properties
    dynamic_pressure = np.array([
        0.5 * compute_atmosphere_properties(h * 1000.0)[2] * v_r**2
        for h, v_r in zip(altitude, velocity_rel)
    ])
    
    # Actual pitch angle
    actual_pitch = np.array(getattr(log, 'actual_pitch_angle', pitch_angle))
    
    # =========================================================================
    # Extended telemetry for research-grade individual plots
    # =========================================================================
    omega_x = np.array(getattr(log, 'omega_x', [0.0]*len(time)))
    omega_y = np.array(getattr(log, 'omega_y', [0.0]*len(time)))
    omega_z = np.array(getattr(log, 'omega_z', [0.0]*len(time)))
    quaternion_norm = np.array(getattr(log, 'quaternion_norm', [1.0]*len(time)))

    velocity_horizontal = np.array(getattr(log, 'velocity_horizontal', [0.0]*len(time)))
    velocity_vertical = np.array(getattr(log, 'velocity_vertical', [0.0]*len(time)))

    throttle_arr = np.array(getattr(log, 'throttle', [1.0]*len(time)))
    thrust_on_arr = np.array(getattr(log, 'thrust_on', [1.0]*len(time)))

    # Commanded and actual quaternions
    cq_w = np.array(getattr(log, 'commanded_quat_w', [1.0]*len(time)))
    cq_x = np.array(getattr(log, 'commanded_quat_x', [0.0]*len(time)))
    cq_y = np.array(getattr(log, 'commanded_quat_y', [0.0]*len(time)))
    cq_z = np.array(getattr(log, 'commanded_quat_z', [0.0]*len(time)))
    aq_w = np.array(getattr(log, 'actual_quat_w', [1.0]*len(time)))
    aq_x = np.array(getattr(log, 'actual_quat_x', [0.0]*len(time)))
    aq_y = np.array(getattr(log, 'actual_quat_y', [0.0]*len(time)))
    aq_z = np.array(getattr(log, 'actual_quat_z', [0.0]*len(time)))
    commanded_quat = np.column_stack((cq_w, cq_x, cq_y, cq_z))
    actual_quat = np.column_stack((aq_w, aq_x, aq_y, aq_z))

    # Atmospheric properties along trajectory
    atm_temp = np.zeros(len(time))
    atm_pressure = np.zeros(len(time))
    atm_density = np.zeros(len(time))
    atm_sos = np.zeros(len(time))
    mach_arr = np.zeros(len(time))
    for i, (h, v_r) in enumerate(zip(altitude, velocity_rel)):
        T_atm, P_atm, rho_atm, a_atm = compute_atmosphere_properties(h * 1000.0)
        atm_temp[i] = T_atm
        atm_pressure[i] = P_atm
        atm_density[i] = rho_atm
        atm_sos[i] = a_atm
        mach_arr[i] = v_r / a_atm if a_atm > 0 else 0.0

    downrange_ground = np.array(getattr(log, 'downrange_ground', downrange))

    return TrajectoryData(
        time=time,
        altitude=altitude,
        velocity=velocity,
        velocity_rel=velocity_rel,
        mass=mass,
        pitch_angle=pitch_angle,
        actual_pitch=actual_pitch,
        attitude_error=attitude_error,
        torque=torque,
        gamma_cmd=gamma_cmd,
        gamma_actual=gamma_actual,
        gamma_rel=gamma_rel,
        position=position,
        velocity_vec=velocity_vec,
        velocity_rel_vec=velocity_rel_vec,
        downrange=downrange,
        dynamic_pressure=dynamic_pressure,
        thrust_force=thrust_mag,
        # Extended
        omega_x=omega_x,
        omega_y=omega_y,
        omega_z=omega_z,
        quaternion_norm=quaternion_norm,
        velocity_horizontal=velocity_horizontal,
        velocity_vertical=velocity_vertical,
        velocity_x=vel_x,
        velocity_y=vel_y,
        velocity_z=vel_z,
        throttle=throttle_arr,
        thrust_on=thrust_on_arr,
        commanded_quat=commanded_quat,
        actual_quat=actual_quat,
        mach_number=mach_arr,
        temperature=atm_temp,
        pressure=atm_pressure,
        density=atm_density,
        speed_of_sound=atm_sos,
        downrange_ground=downrange_ground,
    )


def compute_gravity_turn_start(data: TrajectoryData, threshold: float = 0.1) -> float:
    """Determine the start time of the gravity turn maneuver.
    
    Args:
        data: TrajectoryData object
        threshold: Pitch rate threshold in deg/s to detect gravity turn
        
    Returns:
        Time in seconds when gravity turn begins (as Python float)
    """
    if len(data.time) < 2:
        return 0.0
    
    pitch_rate = np.abs(np.diff(data.pitch_angle))
    indices = np.where(pitch_rate > threshold)[0]
    
    if len(indices) > 0:
        return float(data.time[indices[0]])
    return float(data.time[min(20, len(data.time) - 1)])


# =============================================================================
# Individual Plot Functions
# =============================================================================

def plot_altitude_profile(data: TrajectoryData, output_dir: str) -> str:
    """Generate altitude vs time profile plot.
    
    Args:
        data: TrajectoryData object
        output_dir: Directory to save the plot
        
    Returns:
        Path to saved plot file
    """
    fig, ax = plt.subplots()
    
    ax.fill_between(data.time, 0, data.altitude, alpha=0.25, color='#1f77b4')
    ax.plot(data.time, data.altitude, 'b-', linewidth=2, label='Altitude')
    
    # Key events
    ax.scatter([data.time[0]], [data.altitude[0]], 
               c='green', s=80, marker='o', zorder=5, label='Liftoff')
    ax.scatter([data.time[-1]], [data.altitude[-1]], 
               c='red', s=80, marker='x', zorder=5, 
               label=f'MECO ({data.altitude[-1]:.1f} km)')
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Altitude (km)')
    ax.set_title('Altitude Profile', fontweight='bold')
    ax.legend(loc='lower right', framealpha=0.95)
    ax.set_xlim(0, data.time[-1])
    ax.set_ylim(0, None)
    
    plt.tight_layout()
    path = os.path.join(output_dir, '01_altitude_profile.png')
    fig.savefig(path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    
    return path


def plot_velocity_profile(data: TrajectoryData, output_dir: str) -> str:
    """Generate velocity profile comparing inertial and relative velocities.
    
    Args:
        data: TrajectoryData object
        output_dir: Directory to save the plot
        
    Returns:
        Path to saved plot file
    """
    fig, ax = plt.subplots()
    
    ax.plot(data.time, data.velocity, 'r-', linewidth=2, 
            label='Inertial Velocity')
    ax.plot(data.time, data.velocity_rel, 'g--', linewidth=2, 
            label='Relative Velocity (Airspeed)')
    
    ax.scatter([data.time[0]], [data.velocity[0]], 
               c='red', s=50, marker='o', zorder=5)
    ax.scatter([data.time[0]], [data.velocity_rel[0]], 
               c='green', s=50, marker='o', zorder=5)
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Velocity (m/s)')
    ax.set_title('Velocity Profile: Inertial vs Relative', fontweight='bold')
    ax.legend(loc='lower right', framealpha=0.95)
    ax.set_xlim(0, data.time[-1])
    ax.set_ylim(0, None)
    
    # Annotation
    ax.text(0.02, 0.98, 
            f'Difference due to Earth rotation\n(~{data.velocity[0]:.0f} m/s at equator)',
            transform=ax.transAxes, fontsize=9,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    plt.tight_layout()
    path = os.path.join(output_dir, '02_velocity_profile.png')
    fig.savefig(path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    
    return path


def plot_mass_profile(data: TrajectoryData, output_dir: str) -> str:
    """Generate vehicle mass vs time profile.
    
    Args:
        data: TrajectoryData object
        output_dir: Directory to save the plot
        
    Returns:
        Path to saved plot file
    """
    fig, ax = plt.subplots()
    
    mass_tonnes = data.mass / 1000.0
    
    ax.fill_between(data.time, 0, mass_tonnes, alpha=0.25, color='#2ca02c')
    ax.plot(data.time, mass_tonnes, 'g-', linewidth=2, label='Vehicle Mass')
    ax.axhline(y=C.DRY_MASS/1000, color='orange', linestyle='--', 
               linewidth=1.5, label=f'Dry Mass ({C.DRY_MASS/1000:.0f} t)')
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Mass (tonnes)')
    ax.set_title('Vehicle Mass Profile', fontweight='bold')
    ax.legend(loc='upper right', framealpha=0.95)
    ax.set_xlim(0, data.time[-1])
    ax.set_ylim(0, None)
    
    plt.tight_layout()
    path = os.path.join(output_dir, '03_mass_profile.png')
    fig.savefig(path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    
    return path


def plot_pitch_angle(data: TrajectoryData, output_dir: str) -> str:
    """Generate pitch angle evolution plot.
    
    Args:
        data: TrajectoryData object
        output_dir: Directory to save the plot
        
    Returns:
        Path to saved plot file
    """
    fig, ax = plt.subplots()
    
    gravity_turn_time = compute_gravity_turn_start(data)
    
    ax.fill_between(data.time, 0, data.pitch_angle, alpha=0.25, color='#9467bd')
    ax.plot(data.time, data.pitch_angle, color='purple', linewidth=2, 
            label='Pitch Angle (from Vertical)')
    ax.axvline(x=gravity_turn_time, color='gray', linestyle=':', 
               linewidth=1.5, label='Gravity Turn Start')
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Pitch Angle (° from Vertical)')
    ax.set_title('Pitch Angle Evolution', fontweight='bold')
    ax.legend(loc='upper left', framealpha=0.95)
    ax.set_xlim(0, data.time[-1])
    ax.set_ylim(0, 95)
    
    plt.tight_layout()
    path = os.path.join(output_dir, '04_pitch_angle.png')
    fig.savefig(path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    
    return path


def plot_attitude_error(data: TrajectoryData, output_dir: str) -> str:
    """Generate attitude control error plot.
    
    Args:
        data: TrajectoryData object
        output_dir: Directory to save the plot
        
    Returns:
        Path to saved plot file
    """
    fig, ax = plt.subplots()
    
    ax.plot(data.time, data.attitude_error, 'c-', linewidth=1.5, 
            label='Attitude Error')
    ax.axhline(y=1.0, color='orange', linestyle='--', 
               label='1° Threshold', linewidth=1.5)
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Error (°)')
    ax.set_title('Attitude Control Error', fontweight='bold')
    ax.set_xlim(0, data.time[-1])
    ax.legend(loc='upper right', framealpha=0.95)
    
    plt.tight_layout()
    path = os.path.join(output_dir, '05_attitude_error.png')
    fig.savefig(path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    
    return path


def plot_control_torque(data: TrajectoryData, output_dir: str) -> str:
    """Generate control torque magnitude plot.
    
    Args:
        data: TrajectoryData object
        output_dir: Directory to save the plot
        
    Returns:
        Path to saved plot file
    """
    fig, ax = plt.subplots()
    
    torque_mn = data.torque / 1e6
    
    ax.plot(data.time, torque_mn, color='#ff7f0e', linewidth=1.5, 
            label='Control Torque')
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Torque (MN·m)')
    ax.set_title('Control Torque Magnitude', fontweight='bold')
    ax.set_xlim(0, data.time[-1])
    
    plt.tight_layout()
    path = os.path.join(output_dir, '06_control_torque.png')
    fig.savefig(path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    
    return path


def plot_trajectory_local(data: TrajectoryData, output_dir: str) -> str:
    """Generate 2D trajectory plot (downrange vs altitude).
    
    Args:
        data: TrajectoryData object
        output_dir: Directory to save the plot
        
    Returns:
        Path to saved plot file
    """
    fig, ax = plt.subplots()
    
    ax.plot(data.downrange, data.altitude, 'b-', linewidth=2, 
            label='Ascent Trajectory')
    
    ax.set_xlabel('Downrange (km)')
    ax.set_ylabel('Altitude (km)')
    ax.set_title('Ground Track (Local Frame)', fontweight='bold')
    ax.legend(loc='lower right', framealpha=0.95)
    ax.set_xlim(0, None)
    ax.set_ylim(0, None)
    
    plt.tight_layout()
    path = os.path.join(output_dir, '07_trajectory_local.png')
    fig.savefig(path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    
    return path


def plot_trajectory_3d(data: TrajectoryData, output_dir: str) -> str:
    """Generate 3D trajectory plot.
    
    Args:
        data: TrajectoryData object
        output_dir: Directory to save the plot
        
    Returns:
        Path to saved plot file
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    pos_x = data.position[:, 0]
    pos_y = data.position[:, 1]
    
    east = (pos_y - pos_y[0]) / 1000
    north = (pos_x - pos_x[0]) / 1000
    
    ax.plot(east, north, data.altitude, linewidth=2, color='#1f77b4')
    
    ax.set_xlabel('East (km)')
    ax.set_ylabel('North (km)')
    ax.set_zlabel('Altitude (km)')
    ax.set_title('3D Trajectory', fontweight='bold')
    
    plt.tight_layout()
    path = os.path.join(output_dir, '08_trajectory_3d.png')
    fig.savefig(path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    
    return path


def plot_thrust_vs_gravity(data: TrajectoryData, output_dir: str) -> str:
    """Generate thrust vs gravity force comparison plot.
    
    Args:
        data: TrajectoryData object
        output_dir: Directory to save the plot
        
    Returns:
        Path to saved plot file
    """
    fig, ax = plt.subplots()
    
    thrust_force = data.thrust_force / 1e6
    r_center = data.altitude * 1000.0 + C.R_EARTH
    gravity_force = (C.MU_EARTH * data.mass / (r_center**2)) / 1e6
    
    ax.plot(data.time, thrust_force, 'r-', linewidth=2, label='Thrust')
    ax.plot(data.time, gravity_force, 'b-', linewidth=2, label='Weight')
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Force (MN)')
    ax.set_title('Thrust vs Weight', fontweight='bold')
    ax.legend(loc='upper right', framealpha=0.95)
    ax.set_xlim(0, data.time[-1])
    
    plt.tight_layout()
    path = os.path.join(output_dir, '09_thrust_vs_gravity.png')
    fig.savefig(path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    
    return path


def plot_flight_path_angle(data: TrajectoryData, output_dir: str) -> str:
    """Generate flight path angle evolution plot.
    
    Args:
        data: TrajectoryData object
        output_dir: Directory to save the plot
        
    Returns:
        Path to saved plot file
    """
    fig, ax = plt.subplots()
    
    ax.plot(data.time, data.gamma_rel, 'b-', linewidth=2.5, 
            label=r'$\gamma_{relative}$ (Primary)')
    ax.plot(data.time, data.gamma_cmd, 'g--', linewidth=1.5, alpha=0.7,
            label=r'$\gamma_{command}$')
    ax.plot(data.time, data.gamma_actual, 'r:', linewidth=1.5, alpha=0.7,
            label=r'$\gamma_{actual}$')
    
    ax.axhline(y=90, color='gray', linestyle=':', alpha=0.5)
    ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel(r'Flight Path Angle $\gamma$ (° from Horizontal)')
    ax.set_title('Flight Path Angle Evolution', fontweight='bold')
    ax.legend(loc='center right', framealpha=0.95)
    ax.set_xlim(0, data.time[-1])
    ax.set_ylim(-5, 100)
    
    # Physics explanation
    textstr = (r'$\gamma$ Definition:' + '\n'
               r'$\gamma = 90°$: Vertical climb' + '\n'
               r'$\gamma = 0°$: Horizontal flight' + '\n'
               r'$\gamma < 0°$: Descent')
    ax.text(0.02, 0.35, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.95))
    
    plt.tight_layout()
    path = os.path.join(output_dir, '10_flight_path_angle.png')
    fig.savefig(path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    
    return path


def plot_dynamic_pressure(data: TrajectoryData, output_dir: str) -> str:
    """Generate dynamic pressure plot with structural limit.
    
    Args:
        data: TrajectoryData object
        output_dir: Directory to save the plot
        
    Returns:
        Path to saved plot file
    """
    fig, ax = plt.subplots()
    
    q_kpa = data.dynamic_pressure / 1000.0
    limit_kpa = C.MAX_DYNAMIC_PRESSURE / 1000.0
    
    ax.plot(data.time, q_kpa, color='#9467bd', linewidth=2, 
            label='Dynamic Pressure')
    ax.axhline(y=limit_kpa, color='red', linestyle='--', linewidth=1.5,
               label=f'Structural Limit ({limit_kpa:.0f} kPa)')
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Dynamic Pressure (kPa)')
    ax.set_title('Dynamic Pressure (q)', fontweight='bold')
    ax.legend(loc='upper right', framealpha=0.95)
    ax.set_xlim(0, data.time[-1])
    ax.set_ylim(0, None)
    
    plt.tight_layout()
    path = os.path.join(output_dir, '11_dynamic_pressure.png')
    fig.savefig(path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    
    return path


def plot_physics_check(data: TrajectoryData, output_dir: str) -> str:
    """Generate physics validation plot (angle of attack analysis).
    
    Args:
        data: TrajectoryData object
        output_dir: Directory to save the plot
        
    Returns:
        Path to saved plot file
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    
    # Compute velocity angle from vertical
    v_rel_mag = np.linalg.norm(data.velocity_rel_vec, axis=1)
    r_mag = np.linalg.norm(data.position, axis=1)
    r_hat = data.position / r_mag[:, np.newaxis]
    v_rel_radial = np.sum(data.velocity_rel_vec * r_hat, axis=1)
    cos_theta = np.clip(v_rel_radial / np.maximum(v_rel_mag, 1.0), -1.0, 1.0)
    velocity_angle = np.degrees(np.arccos(cos_theta))
    
    # Subplot 1: Pitch vs Velocity Angle
    ax1.plot(data.time, data.pitch_angle, 'purple', linewidth=2,
             label='Pitch Angle $\\theta$ (Body Z)')
    ax1.plot(data.time, velocity_angle, 'g--', linewidth=2,
             label='Velocity Angle (Wind-Relative)')
    ax1.set_ylabel('Angle from Vertical (°)')
    ax1.set_title('Pitch vs Wind-Relative Velocity Vector', fontweight='bold')
    ax1.legend(loc='lower right', framealpha=0.95)
    ax1.set_ylim(0, 95)
    ax1.grid(True, which='both', alpha=0.3)
    
    # Subplot 2: Angle of Attack Estimate with Q context
    alpha = np.abs(data.pitch_angle - velocity_angle)
    
    # Mask AoA where dynamic pressure is negligible (q < 100 Pa)
    # This hides the misleading "90 degree" spike at liftoff where
    # velocity is just Earth rotation and airspeed is effectively zero.
    q_threshold = 100.0  # Pa
    alpha_masked = np.where(data.dynamic_pressure > q_threshold, alpha, np.nan)
    
    # Primary Axis: AoA
    line1 = ax2.plot(data.time, alpha_masked, 'r-', linewidth=2, 
             label='Angle of Attack (Wind-Relative)')
    ax2.axhline(y=0, color='k', linestyle='-', linewidth=0.8)
    line2 = ax2.axhline(y=10, color='orange', linestyle='--', 
                label='10° Safety Threshold')
    
    ax2.set_ylabel('Angle of Attack (°)')
    # Use non-masked max for y-limit, but cap at reasonable value
    ax2.set_ylim(0, min(15, max(12, np.nanmax(alpha_masked)*1.2 if not np.all(np.isnan(alpha_masked)) else 12)))
    
    # Secondary Axis: Dynamic Pressure
    ax2b = ax2.twinx()
    q_kpa = data.dynamic_pressure / 1000.0
    
    # Fill background based on Q
    # We want to highlight High-Q region
    # Create a filled curve
    ax2b.fill_between(data.time, 0, q_kpa, color='gray', alpha=0.15, label='Dynamic Pressure (q)')
    line3 = ax2b.plot(data.time, q_kpa, color='gray', linestyle=':', linewidth=1, label='Dynamic Pressure')
    
    ax2b.set_ylabel('Dynamic Pressure (kPa)', color='gray')
    ax2b.tick_params(axis='y', labelcolor='gray')
    ax2b.set_ylim(0, None)
    
    # Combine legends
    lines = line1 + [line2] + line3
    lbls = [l.get_label() for l in lines]
    ax2.legend(lines, lbls, loc='upper right', framealpha=0.95)
    
    ax2.set_xlabel('Time (s)')
    ax2.set_title('Angle of Attack & Dynamic Pressure', fontweight='bold')
    
    # Annotations for Guidance Phases
    # Prograde Lock starts roughly when Q > 10kPa or Alt > 10km
    # Let's find time where Q > 10kPa
    high_q_idx = np.where(q_kpa > 10.0)[0]
    if len(high_q_idx) > 0:
        t_lock = data.time[high_q_idx[0]]
        ax2.axvline(x=t_lock, color='blue', linestyle='-.', alpha=0.5)
        ax2.text(t_lock + 2, ax2.get_ylim()[1]*0.8, "High-Q / Prograde Lock", 
                 color='blue', fontsize=9, rotation=0)

    plt.tight_layout()
    path = os.path.join(output_dir, '12_physics_validation.png')
    fig.savefig(path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    
    return path


def plot_pitch_gamma_diagnostic(data: TrajectoryData, output_dir: str) -> str:
    """Generate diagnostic plot comparing thrust direction vs velocity direction.
    
    This is the key diagnostic for verifying pitch/trajectory coupling:
    - Thrust pitch (from vertical): Actual body Z vs local vertical
    - Velocity tilt (from vertical): atan(v_horiz/v_vert)
    - Gamma command (from horizontal): Guidance target
    
    If these don't align properly, there's a frame or coupling issue.
    
    Args:
        data: TrajectoryData object
        output_dir: Directory to save the plot
        
    Returns:
        Path to saved plot file
    """
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    fig.suptitle('Pitch/Gamma Diagnostic: Thrust Direction vs Velocity Direction', 
                 fontsize=14, fontweight='bold')
    
    # Compute velocity tilt from vertical (0 = vertical, 90 = horizontal)
    r_mag = np.linalg.norm(data.position, axis=1)
    r_hat = data.position / r_mag[:, np.newaxis]
    
    # Use relative velocity for vt calculation
    v_rel_radial = np.sum(data.velocity_rel_vec * r_hat, axis=1)  # vertical component
    v_rel_tangent = np.sqrt(np.maximum(data.velocity_rel**2 - v_rel_radial**2, 0))  # horizontal
    velocity_tilt_from_vertical = np.degrees(np.arctan2(v_rel_tangent, np.maximum(v_rel_radial, 1.0)))
    
    # Pitch angles  
    # data.pitch_angle = commanded pitch from vertical
    # data.actual_pitch = actual body pitch from vertical
    pitch_cmd = data.pitch_angle
    pitch_actual = data.actual_pitch
    
    # Gamma (from horizontal) -> pitch from vertical = 90 - gamma
    gamma_implies_pitch = 90.0 - data.gamma_cmd
    
    # ===== Subplot 1: All angles on same scale =====
    ax1.plot(data.time, pitch_cmd, 'b-', linewidth=2, label='Pitch Command (from vertical)')
    ax1.plot(data.time, pitch_actual, 'b--', linewidth=1.5, alpha=0.7, label='Pitch Actual (from vertical)')
    ax1.plot(data.time, velocity_tilt_from_vertical, 'g-', linewidth=2, label='Velocity Tilt (from vertical)')
    ax1.plot(data.time, gamma_implies_pitch, 'r:', linewidth=2, label='90° - γ_cmd')
    
    ax1.set_ylabel('Angle from Vertical (°)')
    ax1.set_title('Comparison: Thrust Pitch vs Velocity Tilt')
    ax1.legend(loc='upper left', fontsize=9, framealpha=0.95)
    ax1.set_ylim(0, 95)
    ax1.axhline(45, color='gray', linestyle=':', alpha=0.3)
    
    # ===== Subplot 2: Gamma angles (from horizontal) =====
    gamma_measured = 90.0 - velocity_tilt_from_vertical
    ax2.plot(data.time, data.gamma_cmd, 'purple', linewidth=2, label='γ Command')
    ax2.plot(data.time, gamma_measured, 'orange', linewidth=2, label='γ Measured (from velocity)')
    ax2.plot(data.time, data.gamma_rel, 'k--', linewidth=1.5, alpha=0.7, label='γ Relative (logged)')
    
    ax2.set_ylabel('Flight Path Angle γ (° from horizontal)')
    ax2.set_title('Flight Path Angle: Command vs Actual')
    ax2.legend(loc='upper right', fontsize=9, framealpha=0.95)
    ax2.set_ylim(-5, 100)
    ax2.axhline(90, color='gray', linestyle=':', alpha=0.3, label='Vertical')
    ax2.axhline(0, color='gray', linestyle=':', alpha=0.3, label='Horizontal')
    
    # ===== Subplot 3: Alignment error =====
    # Difference between thrust pitch and velocity tilt indicates angle of attack
    pitch_velocity_diff = np.abs(pitch_actual - velocity_tilt_from_vertical)
    gamma_tracking_error = np.abs(data.gamma_cmd - gamma_measured)
    
    ax3.plot(data.time, pitch_velocity_diff, 'r-', linewidth=2, 
             label='|Thrust Pitch - Velocity Tilt|')
    ax3.plot(data.time, gamma_tracking_error, 'b--', linewidth=1.5,
             label='|γ_cmd - γ_meas|')
    ax3.axhline(10, color='orange', linestyle='--', linewidth=1.5, label='10° Warning')
    
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Angle Difference (°)')
    ax3.set_title('Alignment & Tracking Error')
    ax3.legend(loc='upper right', fontsize=9, framealpha=0.95)
    ax3.set_ylim(0, None)
    
    # Text summary box
    final_pitch = pitch_actual[-1] if len(pitch_actual) > 0 else 0
    final_v_tilt = velocity_tilt_from_vertical[-1] if len(velocity_tilt_from_vertical) > 0 else 0
    final_gamma = data.gamma_cmd[-1] if len(data.gamma_cmd) > 0 else 0
    max_downrange = data.downrange[-1] if len(data.downrange) > 0 else 0
    
    summary = (f"At MECO:\n"
               f"  Pitch (from vert): {final_pitch:.1f}°\n"
               f"  Vel tilt (from vert): {final_v_tilt:.1f}°\n"
               f"  γ command: {final_gamma:.1f}° (from horiz)\n"
               f"  Downrange: {max_downrange:.1f} km\n\n"
               f"Expected: pitch ≈ vel_tilt ≈ (90-γ)")
    
    ax3.text(0.98, 0.97, summary, transform=ax3.transAxes, fontsize=9,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.95))
    
    plt.tight_layout()
    path = os.path.join(output_dir, '14_pitch_gamma_diagnostic.png')
    fig.savefig(path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    
    return path

def plot_comprehensive_dashboard(data: TrajectoryData, output_dir: str) -> str:
    """Generate comprehensive 6-panel dashboard summary.
    
    Args:
        data: TrajectoryData object
        output_dir: Directory to save the plot
        
    Returns:
        Path to saved plot file
    """
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('RLV Phase-I Ascent Summary', fontsize=16, fontweight='bold')
    
    # Panel 1: Altitude
    ax = axes[0, 0]
    ax.fill_between(data.time, 0, data.altitude, alpha=0.25, color='#1f77b4')
    ax.plot(data.time, data.altitude, 'b-', linewidth=2)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Altitude (km)')
    ax.set_title('Altitude Profile')
    ax.set_xlim(0, data.time[-1])
    
    # Panel 2: Velocity
    ax = axes[0, 1]
    ax.plot(data.time, data.velocity, 'r-', linewidth=2, label='Inertial')
    ax.plot(data.time, data.velocity_rel, 'g--', linewidth=2, label='Relative')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Velocity (m/s)')
    ax.set_title('Velocity Profile')
    ax.legend(fontsize=8, framealpha=0.95)
    ax.set_xlim(0, data.time[-1])
    
    # Panel 3: Mass
    ax = axes[0, 2]
    mass_tonnes = data.mass / 1000.0
    ax.fill_between(data.time, 0, mass_tonnes, alpha=0.25, color='#2ca02c')
    ax.plot(data.time, mass_tonnes, 'g-', linewidth=2)
    ax.axhline(y=C.DRY_MASS/1000, color='orange', linestyle='--', linewidth=1)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Mass (tonnes)')
    ax.set_title('Vehicle Mass')
    ax.set_xlim(0, data.time[-1])
    
    # Panel 4: Pitch
    ax = axes[1, 0]
    ax.plot(data.time, data.pitch_angle, 'purple', linewidth=2)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Pitch (° from Vertical)')
    ax.set_title('Pitch Angle')
    ax.set_xlim(0, data.time[-1])
    ax.set_ylim(0, 95)
    
    # Panel 5: Flight Path Angle
    ax = axes[1, 1]
    ax.plot(data.time, data.gamma_rel, 'b-', linewidth=2)
    ax.axhline(y=90, color='gray', linestyle=':', alpha=0.5)
    ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel(r'$\gamma$ (° from Horizontal)')
    ax.set_title('Flight Path Angle')
    ax.set_xlim(0, data.time[-1])
    ax.set_ylim(-5, 100)
    
    # Panel 6: Trajectory
    ax = axes[1, 2]
    ax.plot(data.downrange, data.altitude, 'b-', linewidth=2)
    ax.set_xlabel('Downrange (km)')
    ax.set_ylabel('Altitude (km)')
    ax.set_title('Ground Track')
    ax.set_xlim(0, None)
    ax.set_ylim(0, None)
    
    plt.tight_layout()
    path = os.path.join(output_dir, '13_comprehensive_summary.png')
    fig.savefig(path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    
    return path


# =============================================================================
# README-Specific Plots
# =============================================================================

def plot_ascent_profile(data: TrajectoryData, output_dir: str) -> str:
    """Generate the ascent_profile.png dashboard referenced in README.
    
    Combined 4-panel view of Altitude, Velocity, Mass, and Pitch.
    
    Args:
        data: TrajectoryData object
        output_dir: Directory to save the plot
        
    Returns:
        Path to saved plot file
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('RLV Phase-I Ascent Profile Dashboard', fontsize=16, fontweight='bold')
    
    # Panel 1: Altitude
    ax = axes[0, 0]
    ax.fill_between(data.time, 0, data.altitude, alpha=0.3, color='#1f77b4')
    ax.plot(data.time, data.altitude, 'b-', linewidth=2.5)
    ax.scatter([data.time[-1]], [data.altitude[-1]], c='red', s=100, marker='*', 
               zorder=5, label=f'MECO: {data.altitude[-1]:.1f} km')
    ax.set_xlabel('Time (s)', fontsize=11)
    ax.set_ylabel('Altitude (km)', fontsize=11)
    ax.set_title('Altitude Profile', fontweight='bold', fontsize=12)
    ax.legend(loc='lower right', fontsize=10)
    ax.set_xlim(0, data.time[-1])
    ax.set_ylim(0, None)
    ax.grid(True, alpha=0.3)
    
    # Panel 2: Velocity
    ax = axes[0, 1]
    ax.plot(data.time, data.velocity, 'r-', linewidth=2.5, label='Inertial')
    ax.plot(data.time, data.velocity_rel, 'g--', linewidth=2, label='Relative')
    ax.scatter([data.time[-1]], [data.velocity[-1]], c='red', s=100, marker='*', 
               zorder=5, label=f'MECO: {data.velocity[-1]:.0f} m/s')
    ax.set_xlabel('Time (s)', fontsize=11)
    ax.set_ylabel('Velocity (m/s)', fontsize=11)
    ax.set_title('Velocity Profile', fontweight='bold', fontsize=12)
    ax.legend(loc='lower right', fontsize=10)
    ax.set_xlim(0, data.time[-1])
    ax.set_ylim(0, None)
    ax.grid(True, alpha=0.3)
    
    # Panel 3: Mass
    ax = axes[1, 0]
    mass_tonnes = data.mass / 1000.0
    ax.fill_between(data.time, 0, mass_tonnes, alpha=0.3, color='#2ca02c')
    ax.plot(data.time, mass_tonnes, 'g-', linewidth=2.5)
    ax.axhline(y=C.DRY_MASS/1000, color='orange', linestyle='--', linewidth=2,
               label=f'Dry Mass: {C.DRY_MASS/1000:.0f} t')
    ax.set_xlabel('Time (s)', fontsize=11)
    ax.set_ylabel('Mass (tonnes)', fontsize=11)
    ax.set_title('Vehicle Mass', fontweight='bold', fontsize=12)
    ax.legend(loc='upper right', fontsize=10)
    ax.set_xlim(0, data.time[-1])
    ax.set_ylim(0, None)
    ax.grid(True, alpha=0.3)
    
    # Panel 4: Pitch
    ax = axes[1, 1]
    ax.fill_between(data.time, 0, data.pitch_angle, alpha=0.3, color='#9467bd')
    ax.plot(data.time, data.pitch_angle, color='purple', linewidth=2.5)
    ax.set_xlabel('Time (s)', fontsize=11)
    ax.set_ylabel('Pitch Angle (° from Vertical)', fontsize=11)
    ax.set_title('Pitch Angle Evolution', fontweight='bold', fontsize=12)
    ax.set_xlim(0, data.time[-1])
    ax.set_ylim(0, 95)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    path = os.path.join(output_dir, 'ascent_profile.png')
    fig.savefig(path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    
    return path


def plot_control_dynamics(data: TrajectoryData, output_dir: str) -> str:
    """Generate the control_dynamics.png referenced in README.
    
    Combined view of Attitude Error, Control Torque, and Guidance Commands.
    
    Args:
        data: TrajectoryData object
        output_dir: Directory to save the plot
        
    Returns:
        Path to saved plot file
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    fig.suptitle('Control System Performance', fontsize=16, fontweight='bold')
    
    # Panel 1: Attitude Error
    ax = axes[0]
    ax.plot(data.time, data.attitude_error, 'c-', linewidth=2, label='Attitude Error')
    ax.axhline(y=1.0, color='orange', linestyle='--', linewidth=2, label='1° Threshold')
    max_error = np.max(data.attitude_error)
    ax.axhline(y=max_error, color='red', linestyle=':', linewidth=1.5, alpha=0.7,
               label=f'Max: {max_error:.2f}°')
    ax.set_ylabel('Attitude Error (°)', fontsize=11)
    ax.set_title('Attitude Tracking Error', fontweight='bold', fontsize=12)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, None)
    
    # Panel 2: Control Torque
    ax = axes[1]
    torque_mn = data.torque / 1e6
    ax.plot(data.time, torque_mn, color='#ff7f0e', linewidth=2, label='Control Torque')
    ax.axhline(y=C.MAX_TORQUE/1e6, color='red', linestyle='--', linewidth=1.5,
               label=f'Saturation: {C.MAX_TORQUE/1e6:.1f} MN·m')
    ax.set_ylabel('Torque (MN·m)', fontsize=11)
    ax.set_title('Control Torque Magnitude', fontweight='bold', fontsize=12)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, None)
    
    # Panel 3: Guidance Commands (gamma tracking)
    ax = axes[2]
    ax.plot(data.time, data.gamma_cmd, 'b-', linewidth=2.5, label='γ Command')
    ax.plot(data.time, data.gamma_rel, 'g--', linewidth=2, label='γ Actual')
    ax.axhline(y=90, color='gray', linestyle=':', alpha=0.5)
    ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('Time (s)', fontsize=11)
    ax.set_ylabel('Flight Path Angle (°)', fontsize=11)
    ax.set_title('Guidance Commands', fontweight='bold', fontsize=12)
    ax.legend(loc='center right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-5, 100)
    
    plt.tight_layout()
    path = os.path.join(output_dir, 'control_dynamics.png')
    fig.savefig(path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    
    return path


def plot_flight_path_readme(data: TrajectoryData, output_dir: str) -> str:
    """Generate the 11_flight_path_angle.png referenced in README.
    
    Flight path angle evolution with correct filename for README.
    
    Args:
        data: TrajectoryData object
        output_dir: Directory to save the plot
        
    Returns:
        Path to saved plot file
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    
    ax.plot(data.time, data.gamma_rel, 'b-', linewidth=3, 
            label=r'$\gamma_{relative}$ (Primary)')
    ax.plot(data.time, data.gamma_cmd, 'g--', linewidth=2, alpha=0.8,
            label=r'$\gamma_{command}$')
    ax.plot(data.time, data.gamma_actual, 'r:', linewidth=2, alpha=0.8,
            label=r'$\gamma_{actual}$')
    
    ax.axhline(y=90, color='gray', linestyle=':', alpha=0.5, linewidth=1.5)
    ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5, linewidth=1.5)
    
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel(r'Flight Path Angle $\gamma$ (° from Horizontal)', fontsize=12)
    ax.set_title('Flight Path Angle Evolution', fontweight='bold', fontsize=14)
    ax.legend(loc='center right', fontsize=11, framealpha=0.95)
    ax.set_xlim(0, data.time[-1])
    ax.set_ylim(-5, 100)
    ax.grid(True, alpha=0.3)
    
    # Physics explanation box
    textstr = (r'$\gamma$ Definition:' + '\n'
               r'$\gamma = 90°$: Vertical climb' + '\n'
               r'$\gamma = 0°$: Horizontal flight' + '\n'
               r'$\gamma < 0°$: Descent')
    ax.text(0.02, 0.35, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.95))
    
    plt.tight_layout()
    path = os.path.join(output_dir, '11_flight_path_angle.png')
    fig.savefig(path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    
    return path


# =============================================================================
# NEW INDIVIDUAL RESEARCH PLOTS — Publication-Quality Standalone Figures
# =============================================================================

def plot_quaternion_norm(data: TrajectoryData, output_dir: str) -> str:
    """Quaternion norm history — validates numerical integration integrity.

    A deviation from ||q|| = 1 indicates numerical drift in the attitude
    integrator. This is a key validation metric for any quaternion-based
    6-DOF simulation.
    """
    fig, ax = plt.subplots()

    norm_err = np.maximum(np.abs(data.quaternion_norm - 1.0), 1e-16)
    ax.semilogy(data.time, norm_err, color='#d62728', linewidth=1.8,
                label='|  ||q|| - 1  |')
    ax.axhline(y=C.QUATERNION_NORM_TOL, color='orange', linestyle='--',
               linewidth=1.5, label=f'Tolerance ({C.QUATERNION_NORM_TOL:.0e})')

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Quaternion Norm Error  |  ||q|| - 1  |')
    ax.set_title('Quaternion Norm Integrity Check', fontweight='bold')
    ax.legend(loc='upper right', framealpha=0.95)
    ax.set_xlim(0, data.time[-1])

    textstr = (f'Max deviation: {np.max(norm_err):.2e}\n'
               f'Mean deviation: {np.mean(norm_err):.2e}\n'
               f'RK4 + renormalization')
    ax.text(0.02, 0.97, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.95))

    plt.tight_layout()
    path = os.path.join(output_dir, '18_quaternion_norm.png')
    fig.savefig(path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    return path


def plot_angular_velocity_x(data: TrajectoryData, output_dir: str) -> str:
    """Body-frame angular velocity X component (roll rate)."""
    fig, ax = plt.subplots()

    ax.plot(data.time, np.degrees(data.omega_x), color='#1f77b4', linewidth=1.8)
    ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel(r'$\omega_x$ (°/s)')
    ax.set_title('Angular Velocity — Roll Rate ($\\omega_x$)', fontweight='bold')
    ax.set_xlim(0, data.time[-1])

    plt.tight_layout()
    path = os.path.join(output_dir, '19_omega_x_roll.png')
    fig.savefig(path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    return path


def plot_angular_velocity_y(data: TrajectoryData, output_dir: str) -> str:
    """Body-frame angular velocity Y component (pitch rate)."""
    fig, ax = plt.subplots()

    ax.plot(data.time, np.degrees(data.omega_y), color='#ff7f0e', linewidth=1.8)
    ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel(r'$\omega_y$ (°/s)')
    ax.set_title('Angular Velocity — Pitch Rate ($\\omega_y$)', fontweight='bold')
    ax.set_xlim(0, data.time[-1])

    plt.tight_layout()
    path = os.path.join(output_dir, '20_omega_y_pitch.png')
    fig.savefig(path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    return path


def plot_angular_velocity_z(data: TrajectoryData, output_dir: str) -> str:
    """Body-frame angular velocity Z component (yaw rate)."""
    fig, ax = plt.subplots()

    ax.plot(data.time, np.degrees(data.omega_z), color='#2ca02c', linewidth=1.8)
    ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel(r'$\omega_z$ (°/s)')
    ax.set_title('Angular Velocity — Yaw Rate ($\\omega_z$)', fontweight='bold')
    ax.set_xlim(0, data.time[-1])

    plt.tight_layout()
    path = os.path.join(output_dir, '21_omega_z_yaw.png')
    fig.savefig(path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    return path


def plot_acceleration_profile(data: TrajectoryData, output_dir: str) -> str:
    """Total acceleration magnitude in g-units vs time.

    Shows the net acceleration experienced by the vehicle, critical for
    structural load and payload design constraints.
    """
    fig, ax = plt.subplots()

    # Compute acceleration from velocity derivative (central difference)
    dt = np.diff(data.time)
    dv = np.diff(data.velocity)
    accel = np.zeros(len(data.time))
    accel[1:] = dv / dt
    accel[0] = accel[1]
    accel_g = accel / C.G0

    ax.plot(data.time, accel_g, color='#d62728', linewidth=1.8, label='Net Acceleration')
    ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Acceleration (g)')
    ax.set_title('Vehicle Acceleration Profile', fontweight='bold')
    ax.legend(loc='upper left', framealpha=0.95)
    ax.set_xlim(0, data.time[-1])

    textstr = f'Peak acceleration: {np.max(accel_g):.2f} g'
    ax.text(0.98, 0.97, textstr, transform=ax.transAxes, fontsize=10,
            ha='right', va='top',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.95))

    plt.tight_layout()
    path = os.path.join(output_dir, '22_acceleration_profile.png')
    fig.savefig(path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    return path


def plot_twr(data: TrajectoryData, output_dir: str) -> str:
    """Thrust-to-Weight Ratio vs time.

    TWR > 1 is required for liftoff; TWR increases as propellant is consumed.
    This is a fundamental figure of merit for launch vehicle performance.
    """
    fig, ax = plt.subplots()

    r_center = data.altitude * 1000.0 + C.R_EARTH
    weight = C.MU_EARTH * data.mass / (r_center ** 2)
    twr = data.thrust_force / np.maximum(weight, 1.0)

    ax.plot(data.time, twr, color='#e377c2', linewidth=2, label='TWR')
    ax.axhline(y=1.0, color='red', linestyle='--', linewidth=1.5, label='TWR = 1 (hover)')
    ax.fill_between(data.time, 1.0, twr, where=(twr > 1.0),
                    alpha=0.15, color='green', label='Excess thrust')

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Thrust-to-Weight Ratio')
    ax.set_title('Thrust-to-Weight Ratio (TWR)', fontweight='bold')
    ax.legend(loc='upper left', framealpha=0.95)
    ax.set_xlim(0, data.time[-1])
    ax.set_ylim(0, None)

    textstr = (f'Liftoff TWR: {twr[0]:.2f}\n'
               f'MECO TWR: {twr[-1]:.2f}')
    ax.text(0.98, 0.50, textstr, transform=ax.transAxes, fontsize=10,
            ha='right', va='center',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.95))

    plt.tight_layout()
    path = os.path.join(output_dir, '23_thrust_to_weight.png')
    fig.savefig(path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    return path


def plot_specific_orbital_energy(data: TrajectoryData, output_dir: str) -> str:
    """Specific orbital energy (MJ/kg) vs time.

    E = v^2/2 - mu/r.  This must monotonically increase during powered ascent.
    Positive energy means escape trajectory; orbital insertion requires
    targeting a specific negative energy corresponding to the desired orbit.
    """
    fig, ax = plt.subplots()

    r_mag = np.linalg.norm(data.position, axis=1)
    v_mag = data.velocity
    specific_energy = 0.5 * v_mag ** 2 - C.MU_EARTH / r_mag
    specific_energy_mj = specific_energy / 1e6

    ax.plot(data.time, specific_energy_mj, color='#17becf', linewidth=2,
            label='Specific Orbital Energy')
    ax.axhline(y=0, color='red', linestyle='--', linewidth=1.5, label='Escape energy')

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Specific Orbital Energy (MJ/kg)')
    ax.set_title(r'Specific Orbital Energy $\varepsilon = \frac{v^2}{2} - \frac{\mu}{r}$',
                 fontweight='bold')
    ax.legend(loc='lower right', framealpha=0.95)
    ax.set_xlim(0, data.time[-1])

    plt.tight_layout()
    path = os.path.join(output_dir, '24_specific_orbital_energy.png')
    fig.savefig(path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    return path


def plot_velocity_eci_x(data: TrajectoryData, output_dir: str) -> str:
    """ECI velocity X-component vs time."""
    fig, ax = plt.subplots()

    ax.plot(data.time, data.velocity_x, color='#1f77b4', linewidth=1.8)
    ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('$V_X$ (m/s)')
    ax.set_title('ECI Velocity X-Component (Radial at Launch)', fontweight='bold')
    ax.set_xlim(0, data.time[-1])

    plt.tight_layout()
    path = os.path.join(output_dir, '25_velocity_eci_x.png')
    fig.savefig(path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    return path


def plot_velocity_eci_y(data: TrajectoryData, output_dir: str) -> str:
    """ECI velocity Y-component vs time."""
    fig, ax = plt.subplots()

    ax.plot(data.time, data.velocity_y, color='#ff7f0e', linewidth=1.8)
    ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('$V_Y$ (m/s)')
    ax.set_title('ECI Velocity Y-Component (East at Launch)', fontweight='bold')
    ax.set_xlim(0, data.time[-1])

    plt.tight_layout()
    path = os.path.join(output_dir, '26_velocity_eci_y.png')
    fig.savefig(path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    return path


def plot_velocity_eci_z(data: TrajectoryData, output_dir: str) -> str:
    """ECI velocity Z-component vs time."""
    fig, ax = plt.subplots()

    ax.plot(data.time, data.velocity_z, color='#2ca02c', linewidth=1.8)
    ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('$V_Z$ (m/s)')
    ax.set_title('ECI Velocity Z-Component (North at Launch)', fontweight='bold')
    ax.set_xlim(0, data.time[-1])

    plt.tight_layout()
    path = os.path.join(output_dir, '27_velocity_eci_z.png')
    fig.savefig(path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    return path


def plot_horizontal_vs_vertical_velocity(data: TrajectoryData, output_dir: str) -> str:
    """Horizontal vs vertical velocity decomposition.

    This shows the transition from vertical ascent to horizontal flight,
    which is the hallmark of a gravity turn trajectory.
    """
    fig, ax = plt.subplots()

    ax.plot(data.time, data.velocity_vertical, color='#1f77b4', linewidth=2,
            label='Vertical (radial)')
    ax.plot(data.time, data.velocity_horizontal, color='#ff7f0e', linewidth=2,
            label='Horizontal (tangential)')
    ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)

    # Mark crossover point
    diff = np.abs(data.velocity_horizontal - data.velocity_vertical)
    if len(diff) > 5:
        crossover_idx = np.argmin(diff[5:]) + 5  # skip first few noisy points
        ax.axvline(x=data.time[crossover_idx], color='purple', linestyle='-.',
                   alpha=0.6, linewidth=1.5,
                   label=f'Crossover t={data.time[crossover_idx]:.0f}s')

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Velocity Component (m/s)')
    ax.set_title('Horizontal vs Vertical Velocity Decomposition', fontweight='bold')
    ax.legend(loc='upper left', framealpha=0.95)
    ax.set_xlim(0, data.time[-1])

    plt.tight_layout()
    path = os.path.join(output_dir, '28_horiz_vs_vert_velocity.png')
    fig.savefig(path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    return path


def plot_mach_number(data: TrajectoryData, output_dir: str) -> str:
    """Mach number vs time.

    Key regimes: subsonic (M<0.8), transonic (0.8<M<1.3), supersonic (M>1.3),
    hypersonic (M>5). Annotated with regime boundaries.
    """
    fig, ax = plt.subplots()

    ax.plot(data.time, data.mach_number, color='#d62728', linewidth=2, label='Mach Number')

    # Regime boundaries
    ax.axhline(y=1.0, color='orange', linestyle='--', linewidth=1.5, label='M = 1 (sonic)')
    ax.axhline(y=5.0, color='blue', linestyle='-.', linewidth=1.2, label='M = 5 (hypersonic)')

    # Fill regime backgrounds
    ax.axhspan(0, 0.8, alpha=0.04, color='green')
    ax.axhspan(0.8, 1.3, alpha=0.06, color='orange')
    ax.axhspan(1.3, 5.0, alpha=0.04, color='red')
    ax.axhspan(5.0, max(15, np.max(data.mach_number)*1.1), alpha=0.04, color='blue')

    ax.text(data.time[-1]*0.02, 0.4, 'Subsonic', fontsize=8, color='green')
    ax.text(data.time[-1]*0.02, 1.05, 'Transonic', fontsize=8, color='orange')
    ax.text(data.time[-1]*0.02, 3.0, 'Supersonic', fontsize=8, color='red')
    if np.max(data.mach_number) > 5.5:
        ax.text(data.time[-1]*0.02, 6.0, 'Hypersonic', fontsize=8, color='blue')

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Mach Number')
    ax.set_title('Mach Number Profile', fontweight='bold')
    ax.legend(loc='upper left', framealpha=0.95)
    ax.set_xlim(0, data.time[-1])
    ax.set_ylim(0, None)

    plt.tight_layout()
    path = os.path.join(output_dir, '29_mach_number.png')
    fig.savefig(path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    return path


def plot_atmospheric_temperature(data: TrajectoryData, output_dir: str) -> str:
    """Atmospheric temperature vs altitude (US Standard Atmosphere 1976)."""
    fig, ax = plt.subplots()

    ax.plot(data.temperature, data.altitude, color='#d62728', linewidth=2)
    ax.axhline(y=11.0, color='gray', linestyle='--', linewidth=1.2,
               label='Tropopause (11 km)')

    ax.set_xlabel('Temperature (K)')
    ax.set_ylabel('Altitude (km)')
    ax.set_title('Atmospheric Temperature vs Altitude', fontweight='bold')
    ax.legend(loc='upper right', framealpha=0.95)
    ax.set_ylim(0, None)

    plt.tight_layout()
    path = os.path.join(output_dir, '30_atm_temperature.png')
    fig.savefig(path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    return path


def plot_atmospheric_density(data: TrajectoryData, output_dir: str) -> str:
    """Atmospheric density (log scale) vs altitude."""
    fig, ax = plt.subplots()

    # Mask zero/negative density for log plot
    valid = data.density > 0
    ax.semilogy(data.altitude[valid], data.density[valid],
                color='#9467bd', linewidth=2)

    ax.set_xlabel('Altitude (km)')
    ax.set_ylabel(r'Density $\rho$ (kg/m$^3$)')
    ax.set_title('Atmospheric Density vs Altitude (Log Scale)', fontweight='bold')
    ax.set_xlim(0, None)

    textstr = (f'Sea level: {data.density[0]:.3f} kg/m$^3$\n'
               f'At MECO: {data.density[-1]:.2e} kg/m$^3$')
    ax.text(0.50, 0.97, textstr, transform=ax.transAxes, fontsize=10,
            ha='center', va='top',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.95))

    plt.tight_layout()
    path = os.path.join(output_dir, '31_atm_density.png')
    fig.savefig(path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    return path


def plot_atmospheric_pressure(data: TrajectoryData, output_dir: str) -> str:
    """Atmospheric pressure (log scale) vs altitude."""
    fig, ax = plt.subplots()

    valid = data.pressure > 0
    ax.semilogy(data.altitude[valid], data.pressure[valid] / 1000.0,
                color='#17becf', linewidth=2)

    ax.set_xlabel('Altitude (km)')
    ax.set_ylabel('Pressure (kPa)')
    ax.set_title('Atmospheric Pressure vs Altitude (Log Scale)', fontweight='bold')
    ax.set_xlim(0, None)

    textstr = f'Sea level: {data.pressure[0]/1000:.1f} kPa'
    ax.text(0.50, 0.97, textstr, transform=ax.transAxes, fontsize=10,
            ha='center', va='top',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.95))

    plt.tight_layout()
    path = os.path.join(output_dir, '32_atm_pressure.png')
    fig.savefig(path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    return path


def plot_speed_of_sound(data: TrajectoryData, output_dir: str) -> str:
    """Speed of sound vs altitude along trajectory."""
    fig, ax = plt.subplots()

    ax.plot(data.altitude, data.speed_of_sound, color='#8c564b', linewidth=2)
    ax.axhline(y=11.0, color='gray', linestyle='--', linewidth=1.2, alpha=0.5)

    ax.set_xlabel('Altitude (km)')
    ax.set_ylabel('Speed of Sound (m/s)')
    ax.set_title('Speed of Sound vs Altitude', fontweight='bold')
    ax.set_xlim(0, None)

    plt.tight_layout()
    path = os.path.join(output_dir, '33_speed_of_sound.png')
    fig.savefig(path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    return path


def plot_throttle_history(data: TrajectoryData, output_dir: str) -> str:
    """Engine throttle setting vs time."""
    fig, ax = plt.subplots()

    throttle_pct = data.throttle * 100.0
    ax.plot(data.time, throttle_pct, color='#e377c2', linewidth=2, label='Throttle')
    ax.fill_between(data.time, 0, throttle_pct, alpha=0.2, color='#e377c2')

    # Show thrust-on envelope
    ax.plot(data.time, data.thrust_on * 100.0, 'k--', linewidth=1.0,
            alpha=0.4, label='Engine ON flag')

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Throttle (%)')
    ax.set_title('Engine Throttle History', fontweight='bold')
    ax.legend(loc='upper right', framealpha=0.95)
    ax.set_xlim(0, data.time[-1])
    ax.set_ylim(-5, 110)

    plt.tight_layout()
    path = os.path.join(output_dir, '34_throttle_history.png')
    fig.savefig(path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    return path


def plot_inertia_variation(data: TrajectoryData, output_dir: str) -> str:
    """Moment of inertia variation with mass (linear interpolation model).

    Shows how the principal moments of inertia change as propellant is consumed.
    This directly affects the control system bandwidth and damping.
    """
    fig, ax = plt.subplots()

    # Compute inertia via linear interpolation (same as mass.py model)
    mass_frac = (data.mass - C.DRY_MASS) / (C.INITIAL_MASS - C.DRY_MASS)
    mass_frac = np.clip(mass_frac, 0, 1)
    Ixx = C.IXX_EMPTY + mass_frac * (C.IXX_FULL - C.IXX_EMPTY)
    Izz = C.IZZ_EMPTY + mass_frac * (C.IZZ_FULL - C.IZZ_EMPTY)

    ax.plot(data.time, Ixx / 1e6, color='#1f77b4', linewidth=2,
            label=r'$I_{xx} = I_{yy}$ (pitch/yaw)')
    ax.plot(data.time, Izz / 1e6, color='#ff7f0e', linewidth=2,
            label=r'$I_{zz}$ (roll)')

    ax.set_xlabel('Time (s)')
    ax.set_ylabel(r'Moment of Inertia ($\times 10^6$ kg·m²)')
    ax.set_title('Principal Moments of Inertia vs Time', fontweight='bold')
    ax.legend(loc='upper right', framealpha=0.95)
    ax.set_xlim(0, data.time[-1])
    ax.set_ylim(0, None)

    textstr = (f'Full: Ixx={C.IXX_FULL:.2e}\n'
               f'Empty: Ixx={C.IXX_EMPTY:.2e}\n'
               f'Ratio: {C.IXX_FULL/C.IXX_EMPTY:.1f}x')
    ax.text(0.50, 0.97, textstr, transform=ax.transAxes, fontsize=9,
            ha='center', va='top',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.95))

    plt.tight_layout()
    path = os.path.join(output_dir, '35_inertia_variation.png')
    fig.savefig(path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    return path


def plot_commanded_vs_actual_quat_w(data: TrajectoryData, output_dir: str) -> str:
    """Commanded vs Actual quaternion scalar component q_w."""
    fig, ax = plt.subplots()

    ax.plot(data.time, data.commanded_quat[:, 0], 'b-', linewidth=2,
            label=r'$q_w^{cmd}$')
    ax.plot(data.time, data.actual_quat[:, 0], 'r--', linewidth=1.5,
            label=r'$q_w^{actual}$')

    ax.set_xlabel('Time (s)')
    ax.set_ylabel(r'Quaternion $q_w$ (scalar)')
    ax.set_title('Commanded vs Actual Quaternion — Scalar $q_w$', fontweight='bold')
    ax.legend(loc='best', framealpha=0.95)
    ax.set_xlim(0, data.time[-1])

    plt.tight_layout()
    path = os.path.join(output_dir, '36_quat_w_cmd_vs_actual.png')
    fig.savefig(path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    return path


def plot_commanded_vs_actual_quat_x(data: TrajectoryData, output_dir: str) -> str:
    """Commanded vs Actual quaternion q_x component."""
    fig, ax = plt.subplots()

    ax.plot(data.time, data.commanded_quat[:, 1], 'b-', linewidth=2,
            label=r'$q_x^{cmd}$')
    ax.plot(data.time, data.actual_quat[:, 1], 'r--', linewidth=1.5,
            label=r'$q_x^{actual}$')

    ax.set_xlabel('Time (s)')
    ax.set_ylabel(r'Quaternion $q_x$')
    ax.set_title('Commanded vs Actual Quaternion — $q_x$', fontweight='bold')
    ax.legend(loc='best', framealpha=0.95)
    ax.set_xlim(0, data.time[-1])

    plt.tight_layout()
    path = os.path.join(output_dir, '37_quat_x_cmd_vs_actual.png')
    fig.savefig(path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    return path


def plot_commanded_vs_actual_quat_y(data: TrajectoryData, output_dir: str) -> str:
    """Commanded vs Actual quaternion q_y component."""
    fig, ax = plt.subplots()

    ax.plot(data.time, data.commanded_quat[:, 2], 'b-', linewidth=2,
            label=r'$q_y^{cmd}$')
    ax.plot(data.time, data.actual_quat[:, 2], 'r--', linewidth=1.5,
            label=r'$q_y^{actual}$')

    ax.set_xlabel('Time (s)')
    ax.set_ylabel(r'Quaternion $q_y$')
    ax.set_title('Commanded vs Actual Quaternion — $q_y$', fontweight='bold')
    ax.legend(loc='best', framealpha=0.95)
    ax.set_xlim(0, data.time[-1])

    plt.tight_layout()
    path = os.path.join(output_dir, '38_quat_y_cmd_vs_actual.png')
    fig.savefig(path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    return path


def plot_commanded_vs_actual_quat_z(data: TrajectoryData, output_dir: str) -> str:
    """Commanded vs Actual quaternion q_z component."""
    fig, ax = plt.subplots()

    ax.plot(data.time, data.commanded_quat[:, 3], 'b-', linewidth=2,
            label=r'$q_z^{cmd}$')
    ax.plot(data.time, data.actual_quat[:, 3], 'r--', linewidth=1.5,
            label=r'$q_z^{actual}$')

    ax.set_xlabel('Time (s)')
    ax.set_ylabel(r'Quaternion $q_z$')
    ax.set_title('Commanded vs Actual Quaternion — $q_z$', fontweight='bold')
    ax.legend(loc='best', framealpha=0.95)
    ax.set_xlim(0, data.time[-1])

    plt.tight_layout()
    path = os.path.join(output_dir, '39_quat_z_cmd_vs_actual.png')
    fig.savefig(path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    return path


def plot_downrange_distance(data: TrajectoryData, output_dir: str) -> str:
    """Downrange distance vs time."""
    fig, ax = plt.subplots()

    ax.plot(data.time, data.downrange, color='#8c564b', linewidth=2,
            label='Downrange (ECI projection)')

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Downrange Distance (km)')
    ax.set_title('Downrange Distance vs Time', fontweight='bold')
    ax.legend(loc='upper left', framealpha=0.95)
    ax.set_xlim(0, data.time[-1])
    ax.set_ylim(0, None)

    textstr = f'Final downrange: {data.downrange[-1]:.1f} km'
    ax.text(0.98, 0.05, textstr, transform=ax.transAxes, fontsize=10,
            ha='right', va='bottom',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.95))

    plt.tight_layout()
    path = os.path.join(output_dir, '40_downrange_distance.png')
    fig.savefig(path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    return path


def plot_mass_flow_rate(data: TrajectoryData, output_dir: str) -> str:
    """Propellant mass flow rate vs time.

    Computed from mass derivative. Shows constant flow rate during powered
    flight and zero during coast.
    """
    fig, ax = plt.subplots()

    # Compute mass flow rate from finite differences
    dt = np.diff(data.time)
    dm = np.diff(data.mass)
    mdot = np.zeros(len(data.time))
    mdot[1:] = -dm / dt  # Positive = fuel being consumed
    mdot[0] = mdot[1]

    ax.plot(data.time, mdot, color='#bcbd22', linewidth=2, label='Mass Flow Rate')
    ax.axhline(y=C.MASS_FLOW_RATE, color='red', linestyle='--', linewidth=1.5,
               label=f'Nominal: {C.MASS_FLOW_RATE:.1f} kg/s')

    ax.set_xlabel('Time (s)')
    ax.set_ylabel(r'$\dot{m}$ (kg/s)')
    ax.set_title('Propellant Mass Flow Rate', fontweight='bold')
    ax.legend(loc='upper right', framealpha=0.95)
    ax.set_xlim(0, data.time[-1])
    ax.set_ylim(0, None)

    plt.tight_layout()
    path = os.path.join(output_dir, '41_mass_flow_rate.png')
    fig.savefig(path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    return path


def plot_propellant_fraction(data: TrajectoryData, output_dir: str) -> str:
    """Propellant fraction remaining vs time.

    Shows the fraction of total propellant still on board.
    """
    fig, ax = plt.subplots()

    prop_remaining = (data.mass - C.DRY_MASS) / C.PROPELLANT_MASS * 100.0
    prop_remaining = np.clip(prop_remaining, 0, 100)

    ax.plot(data.time, prop_remaining, color='#7f7f7f', linewidth=2,
            label='Propellant Remaining')
    ax.fill_between(data.time, 0, prop_remaining, alpha=0.2, color='#7f7f7f')
    ax.axhline(y=10, color='red', linestyle='--', linewidth=1.5,
               label='10% Reserve Warning')

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Propellant Remaining (%)')
    ax.set_title('Propellant Fraction Remaining', fontweight='bold')
    ax.legend(loc='upper right', framealpha=0.95)
    ax.set_xlim(0, data.time[-1])
    ax.set_ylim(0, 105)

    textstr = f'At MECO: {prop_remaining[-1]:.1f}%'
    ax.text(0.98, 0.50, textstr, transform=ax.transAxes, fontsize=10,
            ha='right', va='center',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.95))

    plt.tight_layout()
    path = os.path.join(output_dir, '42_propellant_fraction.png')
    fig.savefig(path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    return path


def plot_drag_coefficient_vs_mach(data: TrajectoryData, output_dir: str) -> str:
    """Drag coefficient Cd vs Mach number (as experienced along trajectory).

    Shows the Mach-dependent Cd look-up table values, including the transonic
    drag rise — a critical phenomenon for launch vehicles.
    """
    fig, ax = plt.subplots()

    # Compute Cd from look-up table along trajectory
    cd_trajectory = np.interp(data.mach_number, C.MACH_BREAKPOINTS, C.CD_VALUES)

    ax.plot(data.mach_number, cd_trajectory, 'o', markersize=1.5, alpha=0.3,
            color='#1f77b4', label='Trajectory Cd(M)')

    # Also plot the reference table
    mach_fine = np.linspace(0, max(np.max(data.mach_number), 10), 500)
    cd_fine = np.interp(mach_fine, C.MACH_BREAKPOINTS, C.CD_VALUES)
    ax.plot(mach_fine, cd_fine, 'r-', linewidth=2, label='Cd(M) Look-up Table')

    ax.axvline(x=1.0, color='gray', linestyle=':', linewidth=1.2, alpha=0.6,
               label='M = 1')

    ax.set_xlabel('Mach Number')
    ax.set_ylabel(r'Drag Coefficient $C_D$')
    ax.set_title('Drag Coefficient vs Mach Number', fontweight='bold')
    ax.legend(loc='upper right', framealpha=0.95)
    ax.set_xlim(0, None)
    ax.set_ylim(0, None)

    plt.tight_layout()
    path = os.path.join(output_dir, '43_drag_coeff_vs_mach.png')
    fig.savefig(path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    return path


def plot_energy_budget(data: TrajectoryData, output_dir: str) -> str:
    """Energy budget: kinetic, potential, and total specific mechanical energy.

    Tracks how thrust converts chemical energy into kinetic and potential energy.
    Losses due to gravity and drag are visible as the gap between input and total.
    """
    fig, ax = plt.subplots()

    r_mag = np.linalg.norm(data.position, axis=1)
    KE = 0.5 * data.velocity ** 2  # Specific kinetic energy (J/kg)
    PE = -C.MU_EARTH / r_mag       # Specific potential energy (J/kg)
    TE = KE + PE                    # Total specific mechanical energy

    # Convert to MJ/kg
    ax.plot(data.time, KE / 1e6, color='#d62728', linewidth=2, label='Kinetic Energy')
    ax.plot(data.time, PE / 1e6, color='#1f77b4', linewidth=2, label='Potential Energy')
    ax.plot(data.time, TE / 1e6, color='#2ca02c', linewidth=2.5,
            label='Total Mechanical Energy')
    ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Specific Energy (MJ/kg)')
    ax.set_title('Specific Mechanical Energy Budget', fontweight='bold')
    ax.legend(loc='center left', framealpha=0.95)
    ax.set_xlim(0, data.time[-1])

    plt.tight_layout()
    path = os.path.join(output_dir, '44_energy_budget.png')
    fig.savefig(path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    return path


def plot_altitude_vs_velocity(data: TrajectoryData, output_dir: str) -> str:
    """Altitude vs velocity (state-space trajectory).

    This phase portrait shows the trajectory in altitude-velocity space,
    a common representation in astrodynamics for performance analysis.
    """
    fig, ax = plt.subplots()

    # Color by time
    sc = ax.scatter(data.velocity, data.altitude, c=data.time, cmap='viridis',
                    s=3, alpha=0.8)
    cbar = plt.colorbar(sc, ax=ax, label='Time (s)')

    ax.scatter([data.velocity[0]], [data.altitude[0]], c='green', s=80,
               marker='o', zorder=5, label='Liftoff')
    ax.scatter([data.velocity[-1]], [data.altitude[-1]], c='red', s=80,
               marker='*', zorder=5,
               label=f'MECO ({data.velocity[-1]:.0f} m/s, {data.altitude[-1]:.0f} km)')

    ax.set_xlabel('Inertial Velocity (m/s)')
    ax.set_ylabel('Altitude (km)')
    ax.set_title('Altitude-Velocity State Space', fontweight='bold')
    ax.legend(loc='upper left', framealpha=0.95)
    ax.set_xlim(0, None)
    ax.set_ylim(0, None)

    plt.tight_layout()
    path = os.path.join(output_dir, '45_altitude_vs_velocity.png')
    fig.savefig(path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    return path


def plot_gravity_loss(data: TrajectoryData, output_dir: str) -> str:
    """Cumulative gravity loss vs time.

    Gravity loss = integral of g * sin(gamma) dt.
    This is one of the dominant velocity losses for launch vehicles.
    """
    fig, ax = plt.subplots()

    r_mag = np.linalg.norm(data.position, axis=1)
    g_local = C.MU_EARTH / (r_mag ** 2)

    # gamma_rel is from horizontal, so sin(gamma) gives vertical component
    gamma_rad = np.radians(data.gamma_rel)
    gravity_loss_rate = g_local * np.sin(gamma_rad)

    # Cumulative integration (trapezoidal)
    cumulative_loss = np.zeros(len(data.time))
    for i in range(1, len(data.time)):
        dt = data.time[i] - data.time[i - 1]
        cumulative_loss[i] = cumulative_loss[i - 1] + 0.5 * (
            gravity_loss_rate[i] + gravity_loss_rate[i - 1]) * dt

    ax.plot(data.time, cumulative_loss, color='#d62728', linewidth=2,
            label='Cumulative Gravity Loss')

    ax.set_xlabel('Time (s)')
    ax.set_ylabel(r'Gravity Loss $\Delta V_{grav}$ (m/s)')
    ax.set_title('Cumulative Gravity Loss', fontweight='bold')
    ax.legend(loc='upper left', framealpha=0.95)
    ax.set_xlim(0, data.time[-1])
    ax.set_ylim(0, None)

    textstr = f'Total gravity loss: {cumulative_loss[-1]:.0f} m/s'
    ax.text(0.98, 0.50, textstr, transform=ax.transAxes, fontsize=10,
            ha='right', va='center',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.95))

    plt.tight_layout()
    path = os.path.join(output_dir, '46_gravity_loss.png')
    fig.savefig(path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    return path


def plot_drag_loss(data: TrajectoryData, output_dir: str) -> str:
    """Cumulative drag loss vs time.

    Drag loss = integral of (D/m) dt where D = 0.5 * rho * Cd * A * v_rel^2.
    """
    fig, ax = plt.subplots()

    cd_traj = np.interp(data.mach_number, C.MACH_BREAKPOINTS, C.CD_VALUES)
    drag_force = 0.5 * data.density * cd_traj * C.REFERENCE_AREA * data.velocity_rel ** 2
    drag_decel = drag_force / data.mass  # m/s^2

    cumulative_drag = np.zeros(len(data.time))
    for i in range(1, len(data.time)):
        dt = data.time[i] - data.time[i - 1]
        cumulative_drag[i] = cumulative_drag[i - 1] + 0.5 * (
            drag_decel[i] + drag_decel[i - 1]) * dt

    ax.plot(data.time, cumulative_drag, color='#ff7f0e', linewidth=2,
            label='Cumulative Drag Loss')

    ax.set_xlabel('Time (s)')
    ax.set_ylabel(r'Drag Loss $\Delta V_{drag}$ (m/s)')
    ax.set_title('Cumulative Drag Loss', fontweight='bold')
    ax.legend(loc='upper left', framealpha=0.95)
    ax.set_xlim(0, data.time[-1])
    ax.set_ylim(0, None)

    textstr = f'Total drag loss: {cumulative_drag[-1]:.0f} m/s'
    ax.text(0.98, 0.50, textstr, transform=ax.transAxes, fontsize=10,
            ha='right', va='center',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.95))

    plt.tight_layout()
    path = os.path.join(output_dir, '47_drag_loss.png')
    fig.savefig(path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    return path


def plot_delta_v_budget(data: TrajectoryData, output_dir: str) -> str:
    """Delta-V budget breakdown: ideal, gravity loss, drag loss, achieved.

    Visualizes the Tsiolkovsky rocket equation delta-V and the losses,
    giving a complete propulsion performance picture.
    """
    fig, ax = plt.subplots()

    # Ideal delta-V (Tsiolkovsky)
    mass_ratio = data.mass[0] / data.mass
    ideal_dv = C.ISP * C.G0 * np.log(mass_ratio)

    # Achieved delta-V (actual velocity gain)
    achieved_dv = data.velocity - data.velocity[0]

    # Gravity loss (cumulative)
    r_mag = np.linalg.norm(data.position, axis=1)
    g_local = C.MU_EARTH / (r_mag ** 2)
    gamma_rad = np.radians(data.gamma_rel)
    grav_rate = g_local * np.sin(gamma_rad)
    grav_loss = np.zeros(len(data.time))
    for i in range(1, len(data.time)):
        dt = data.time[i] - data.time[i - 1]
        grav_loss[i] = grav_loss[i - 1] + 0.5 * (grav_rate[i] + grav_rate[i - 1]) * dt

    # Drag loss (cumulative)
    cd_traj = np.interp(data.mach_number, C.MACH_BREAKPOINTS, C.CD_VALUES)
    drag_force = 0.5 * data.density * cd_traj * C.REFERENCE_AREA * data.velocity_rel ** 2
    drag_decel = drag_force / data.mass
    drag_loss = np.zeros(len(data.time))
    for i in range(1, len(data.time)):
        dt = data.time[i] - data.time[i - 1]
        drag_loss[i] = drag_loss[i - 1] + 0.5 * (drag_decel[i] + drag_decel[i - 1]) * dt

    ax.plot(data.time, ideal_dv, 'k-', linewidth=2.5, label='Ideal (Tsiolkovsky)')
    ax.plot(data.time, achieved_dv, 'b-', linewidth=2, label='Achieved')
    ax.plot(data.time, grav_loss, 'r--', linewidth=1.5, label='Gravity Loss')
    ax.plot(data.time, drag_loss, 'g--', linewidth=1.5, label='Drag Loss')

    ax.set_xlabel('Time (s)')
    ax.set_ylabel(r'$\Delta V$ (m/s)')
    ax.set_title(r'$\Delta V$ Budget Breakdown', fontweight='bold')
    ax.legend(loc='upper left', framealpha=0.95)
    ax.set_xlim(0, data.time[-1])
    ax.set_ylim(0, None)

    textstr = (f'Ideal: {ideal_dv[-1]:.0f} m/s\n'
               f'Achieved: {achieved_dv[-1]:.0f} m/s\n'
               f'Gravity: -{grav_loss[-1]:.0f} m/s\n'
               f'Drag: -{drag_loss[-1]:.0f} m/s')
    ax.text(0.02, 0.97, textstr, transform=ax.transAxes, fontsize=10,
            ha='left', va='top',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.95))

    plt.tight_layout()
    path = os.path.join(output_dir, '48_delta_v_budget.png')
    fig.savefig(path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    return path


def plot_position_eci_x(data: TrajectoryData, output_dir: str) -> str:
    """ECI position X-component vs time."""
    fig, ax = plt.subplots()

    ax.plot(data.time, data.position[:, 0] / 1000.0, color='#1f77b4', linewidth=1.8)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('$r_X$ (km)')
    ax.set_title('ECI Position X-Component (Radial at Launch)', fontweight='bold')
    ax.set_xlim(0, data.time[-1])

    plt.tight_layout()
    path = os.path.join(output_dir, '49_position_eci_x.png')
    fig.savefig(path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    return path


def plot_position_eci_y(data: TrajectoryData, output_dir: str) -> str:
    """ECI position Y-component vs time."""
    fig, ax = plt.subplots()

    ax.plot(data.time, data.position[:, 1] / 1000.0, color='#ff7f0e', linewidth=1.8)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('$r_Y$ (km)')
    ax.set_title('ECI Position Y-Component (East at Launch)', fontweight='bold')
    ax.set_xlim(0, data.time[-1])

    plt.tight_layout()
    path = os.path.join(output_dir, '50_position_eci_y.png')
    fig.savefig(path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    return path


def plot_position_eci_z(data: TrajectoryData, output_dir: str) -> str:
    """ECI position Z-component vs time."""
    fig, ax = plt.subplots()

    ax.plot(data.time, data.position[:, 2] / 1000.0, color='#2ca02c', linewidth=1.8)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('$r_Z$ (km)')
    ax.set_title('ECI Position Z-Component (North at Launch)', fontweight='bold')
    ax.set_xlim(0, data.time[-1])

    plt.tight_layout()
    path = os.path.join(output_dir, '51_position_eci_z.png')
    fig.savefig(path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    return path


def plot_natural_frequency(data: TrajectoryData, output_dir: str) -> str:
    """Control system natural frequency and damping ratio vs time.

    Shows how the PD controller's effective bandwidth and damping change
    as the inertia tensor varies with propellant depletion.
    ωn = sqrt(Kp / (2*I)), ζ = Kd / (2*sqrt(Kp*I/2))
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    mass_frac = (data.mass - C.DRY_MASS) / (C.INITIAL_MASS - C.DRY_MASS)
    mass_frac = np.clip(mass_frac, 0, 1)
    Ixx = C.IXX_EMPTY + mass_frac * (C.IXX_FULL - C.IXX_EMPTY)

    omega_n = np.sqrt(C.KP_ATTITUDE / (2.0 * Ixx))
    zeta = C.KD_ATTITUDE / (2.0 * np.sqrt(C.KP_ATTITUDE * Ixx / 2.0))

    ax1.plot(data.time, omega_n, color='#1f77b4', linewidth=2)
    ax1.set_ylabel(r'$\omega_n$ (rad/s)')
    ax1.set_title('Control System Natural Frequency', fontweight='bold')

    ax2.plot(data.time, zeta, color='#d62728', linewidth=2)
    ax2.axhline(y=1.0, color='gray', linestyle='--', linewidth=1.2,
                label='Critical damping')
    ax2.axhline(y=0.7, color='green', linestyle=':', linewidth=1.2,
                label='Design target (0.7)')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel(r'Damping Ratio $\zeta$')
    ax2.set_title('Control System Damping Ratio', fontweight='bold')
    ax2.legend(loc='upper left', framealpha=0.95)

    ax1.set_xlim(0, data.time[-1])
    ax2.set_xlim(0, data.time[-1])

    plt.tight_layout()
    path = os.path.join(output_dir, '52_control_bandwidth.png')
    fig.savefig(path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    return path


def plot_geocentric_radius(data: TrajectoryData, output_dir: str) -> str:
    """Geocentric radius vs time."""
    fig, ax = plt.subplots()

    r_mag = np.linalg.norm(data.position, axis=1) / 1000.0  # km
    ax.plot(data.time, r_mag, color='#17becf', linewidth=2, label='Geocentric Radius')
    ax.axhline(y=C.R_EARTH / 1000.0, color='orange', linestyle='--',
               linewidth=1.5, label=f'Earth Surface ({C.R_EARTH/1000:.0f} km)')

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Geocentric Radius (km)')
    ax.set_title('Geocentric Radius vs Time', fontweight='bold')
    ax.legend(loc='lower right', framealpha=0.95)
    ax.set_xlim(0, data.time[-1])

    plt.tight_layout()
    path = os.path.join(output_dir, '53_geocentric_radius.png')
    fig.savefig(path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    return path


def plot_specific_angular_momentum(data: TrajectoryData, output_dir: str) -> str:
    """Specific angular momentum magnitude vs time.

    h = ||r x v||.  For a Keplerian orbit, h is constant. During powered
    flight, thrust modifies h. This is a fundamental orbital element.
    """
    fig, ax = plt.subplots()

    h_vec = np.cross(data.position, data.velocity_vec)
    h_mag = np.linalg.norm(h_vec, axis=1) / 1e9  # Convert to 10^9 m^2/s

    ax.plot(data.time, h_mag, color='#9467bd', linewidth=2,
            label=r'$h = ||r \times v||$')

    ax.set_xlabel('Time (s)')
    ax.set_ylabel(r'Specific Angular Momentum ($\times 10^9$ m$^2$/s)')
    ax.set_title('Specific Angular Momentum', fontweight='bold')
    ax.legend(loc='lower right', framealpha=0.95)
    ax.set_xlim(0, data.time[-1])

    plt.tight_layout()
    path = os.path.join(output_dir, '54_angular_momentum.png')
    fig.savefig(path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    return path


def plot_dynamic_pressure_vs_altitude(data: TrajectoryData, output_dir: str) -> str:
    """Dynamic pressure vs altitude (instead of time).

    Shows the Max-Q region in altitude space.
    """
    fig, ax = plt.subplots()

    q_kpa = data.dynamic_pressure / 1000.0
    ax.plot(data.altitude, q_kpa, color='#9467bd', linewidth=2,
            label='Dynamic Pressure')
    ax.axhline(y=C.MAX_DYNAMIC_PRESSURE / 1000.0, color='red', linestyle='--',
               linewidth=1.5, label=f'Limit ({C.MAX_DYNAMIC_PRESSURE/1000:.0f} kPa)')

    # Mark Max-Q
    maxq_idx = np.argmax(q_kpa)
    ax.scatter([data.altitude[maxq_idx]], [q_kpa[maxq_idx]], c='red', s=100,
               marker='*', zorder=5,
               label=f'Max-Q: {q_kpa[maxq_idx]:.1f} kPa @ {data.altitude[maxq_idx]:.1f} km')

    ax.set_xlabel('Altitude (km)')
    ax.set_ylabel('Dynamic Pressure (kPa)')
    ax.set_title('Dynamic Pressure vs Altitude', fontweight='bold')
    ax.legend(loc='upper right', framealpha=0.95)
    ax.set_xlim(0, None)
    ax.set_ylim(0, None)

    plt.tight_layout()
    path = os.path.join(output_dir, '55_q_vs_altitude.png')
    fig.savefig(path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    return path


def plot_effective_isp(data: TrajectoryData, output_dir: str) -> str:
    """Effective specific impulse vs altitude.

    ISP varies with ambient pressure: Isp_eff = T / (mdot * g0)
    where T = T_vac - (T_vac - T_sl) * (P_amb / P_sl).
    """
    fig, ax = plt.subplots()

    T_sl = C.THRUST_MAGNITUDE
    T_vac = C.ISP_VAC * C.G0 * C.MASS_FLOW_RATE
    P_sl = C.ATM_P0

    T_eff = T_vac - (T_vac - T_sl) * np.clip(data.pressure / P_sl, 0, 1)
    Isp_eff = T_eff / (C.MASS_FLOW_RATE * C.G0)

    ax.plot(data.altitude, Isp_eff, color='#e377c2', linewidth=2)
    ax.axhline(y=C.ISP, color='blue', linestyle='--', linewidth=1.2,
               label=f'Sea Level Isp = {C.ISP:.0f} s')
    ax.axhline(y=C.ISP_VAC, color='red', linestyle='--', linewidth=1.2,
               label=f'Vacuum Isp = {C.ISP_VAC:.0f} s')

    ax.set_xlabel('Altitude (km)')
    ax.set_ylabel('Effective Isp (s)')
    ax.set_title('Effective Specific Impulse vs Altitude', fontweight='bold')
    ax.legend(loc='center right', framealpha=0.95)
    ax.set_xlim(0, None)

    plt.tight_layout()
    path = os.path.join(output_dir, '56_effective_isp.png')
    fig.savefig(path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    return path


def plot_altitude_rate(data: TrajectoryData, output_dir: str) -> str:
    """Rate of altitude change (vertical velocity) vs time."""
    fig, ax = plt.subplots()

    # Compute altitude rate from finite differences
    dt = np.diff(data.time)
    dh = np.diff(data.altitude) * 1000.0  # Convert km to m
    hdot = np.zeros(len(data.time))
    hdot[1:] = dh / dt
    hdot[0] = hdot[1]

    ax.plot(data.time, hdot, color='#8c564b', linewidth=1.8, label='Altitude Rate')
    ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)

    ax.set_xlabel('Time (s)')
    ax.set_ylabel(r'$\dot{h}$ (m/s)')
    ax.set_title('Altitude Rate of Change', fontweight='bold')
    ax.legend(loc='upper right', framealpha=0.95)
    ax.set_xlim(0, data.time[-1])

    plt.tight_layout()
    path = os.path.join(output_dir, '57_altitude_rate.png')
    fig.savefig(path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    return path


def plot_ground_track(data: TrajectoryData, output_dir: str) -> str:
    """Ground track projection (East vs North from launch site).

    Shows the 2D footprint of the trajectory projected onto Earth's surface.
    """
    fig, ax = plt.subplots()

    east = (data.position[:, 1] - data.position[0, 1]) / 1000.0
    north = (data.position[:, 0] - data.position[0, 0]) / 1000.0

    sc = ax.scatter(east, north, c=data.time, cmap='plasma', s=4, alpha=0.8)
    plt.colorbar(sc, ax=ax, label='Time (s)')

    ax.scatter([0], [0], c='green', s=100, marker='^', zorder=5, label='Launch Site')
    ax.scatter([east[-1]], [north[-1]], c='red', s=100, marker='*',
               zorder=5, label='MECO projection')

    ax.set_xlabel('East (km)')
    ax.set_ylabel('North (km)')
    ax.set_title('Ground Track Projection', fontweight='bold')
    ax.legend(loc='upper left', framealpha=0.95)
    ax.set_aspect('equal')

    plt.tight_layout()
    path = os.path.join(output_dir, '58_ground_track.png')
    fig.savefig(path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    return path


# =============================================================================
# Main Generation Function
# =============================================================================

def generate_all_plots(log, output_dir: str = "plots") -> List[str]:
    """Generate all trajectory and telemetry plots.
    
    This is the main entry point for plot generation. It creates a complete
    set of publication-quality plots for RLV ascent trajectory analysis.
    
    Args:
        log: Simulation log object containing trajectory data
        output_dir: Directory to save plots (created if doesn't exist)
        
    Returns:
        List of paths to saved plot files
        
    Example:
        >>> from rlv_sim.main import run_simulation
        >>> from scripts.plot_generator import generate_all_plots
        >>> state, log, reason = run_simulation()
        >>> plot_files = generate_all_plots(log, "output/plots")
        >>> print(f"Generated {len(plot_files)} plots")
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Configure plotting style
    configure_plot_style()
    
    # Extract and process data
    data = extract_log_data(log)
    
    # Generate all plots
    saved_files = []
    
    plot_functions = [
        # --- Original 17 plots ---
        plot_altitude_profile,
        plot_velocity_profile,
        plot_mass_profile,
        plot_pitch_angle,
        plot_attitude_error,
        plot_control_torque,
        plot_trajectory_local,
        plot_trajectory_3d,
        plot_thrust_vs_gravity,
        plot_flight_path_angle,
        plot_dynamic_pressure,
        plot_physics_check,
        plot_pitch_gamma_diagnostic,
        plot_comprehensive_dashboard,
        plot_ascent_profile,
        plot_control_dynamics,
        plot_flight_path_readme,
        # --- NEW Individual Research Plots (18-58) ---
        plot_quaternion_norm,
        plot_angular_velocity_x,
        plot_angular_velocity_y,
        plot_angular_velocity_z,
        plot_acceleration_profile,
        plot_twr,
        plot_specific_orbital_energy,
        plot_velocity_eci_x,
        plot_velocity_eci_y,
        plot_velocity_eci_z,
        plot_horizontal_vs_vertical_velocity,
        plot_mach_number,
        plot_atmospheric_temperature,
        plot_atmospheric_density,
        plot_atmospheric_pressure,
        plot_speed_of_sound,
        plot_throttle_history,
        plot_inertia_variation,
        plot_commanded_vs_actual_quat_w,
        plot_commanded_vs_actual_quat_x,
        plot_commanded_vs_actual_quat_y,
        plot_commanded_vs_actual_quat_z,
        plot_downrange_distance,
        plot_mass_flow_rate,
        plot_propellant_fraction,
        plot_drag_coefficient_vs_mach,
        plot_energy_budget,
        plot_altitude_vs_velocity,
        plot_gravity_loss,
        plot_drag_loss,
        plot_delta_v_budget,
        plot_position_eci_x,
        plot_position_eci_y,
        plot_position_eci_z,
        plot_natural_frequency,
        plot_geocentric_radius,
        plot_specific_angular_momentum,
        plot_dynamic_pressure_vs_altitude,
        plot_effective_isp,
        plot_altitude_rate,
        plot_ground_track,
    ]
    
    for plot_func in plot_functions:
        try:
            filepath = plot_func(data, output_dir)
            saved_files.append(filepath)
        except Exception as e:
            print(f"Warning: Failed to generate {plot_func.__name__}: {e}")
    
    return saved_files


def main() -> None:
    """Run simulation and generate all plots."""
    print("=" * 70)
    print("  RLV Phase-I Trajectory Visualization — Research Publication Suite")
    print("=" * 70)

    print("\nRunning ascent simulation...")
    final_state, log, reason = run_simulation(verbose=True)

    print(f"\nSimulation complete: {reason}")
    print(f"Final altitude: {final_state.altitude/1000:.1f} km")
    print(f"Final velocity: {final_state.speed:.1f} m/s")

    print("\nGenerating publication-quality plots...")
    output_dir = "plots"
    saved_files = generate_all_plots(log, output_dir)

    print(f"\n{'=' * 70}")
    print(f"  Generated {len(saved_files)} publication-quality plots in '{output_dir}/'")
    print(f"{'=' * 70}")
    for i, path in enumerate(saved_files, 1):
        print(f"  {i:2d}. {os.path.basename(path)}")
    print(f"{'=' * 70}")
    print(f"  All plots saved at 300 DPI — ready for research paper submission.")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
