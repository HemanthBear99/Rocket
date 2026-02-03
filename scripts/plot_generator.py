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
    
    position = np.column_stack((pos_x, pos_y, pos_z))
    velocity_vec = np.column_stack((vel_x, vel_y, vel_z))
    
    # Relative velocity (accounting for Earth rotation)
    omega_vec = np.array([0.0, 0.0, C.EARTH_ROTATION_RATE])
    velocity_rel_vec = velocity_vec - np.cross(omega_vec, position)
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
        dynamic_pressure=dynamic_pressure
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
    
    thrust_force = np.ones_like(data.time) * C.THRUST_MAGNITUDE / 1e6
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
             label='Pitch Angle $\\theta$')
    ax1.plot(data.time, velocity_angle, 'g--', linewidth=2,
             label='Velocity Angle')
    ax1.set_ylabel('Angle from Vertical (°)')
    ax1.set_title('Pitch vs Velocity Vector Alignment', fontweight='bold')
    ax1.legend(loc='upper left', framealpha=0.95)
    ax1.set_ylim(0, 95)
    
    # Subplot 2: Angle of Attack Estimate
    alpha = np.abs(data.pitch_angle - velocity_angle)
    ax2.plot(data.time, alpha, 'r-', linewidth=2, 
             label='Estimated Angle of Attack')
    ax2.axhline(y=0, color='k', linestyle='-', linewidth=0.8)
    ax2.axhline(y=10, color='orange', linestyle='--', 
                label='10° Awareness Threshold')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Angle of Attack (°)')
    ax2.set_title('Estimated Angle of Attack', fontweight='bold')
    ax2.legend(loc='upper right', framealpha=0.95)
    ax2.set_ylim(0, None)
    
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
    print("=" * 60)
    print("RLV Phase-I Trajectory Visualization")
    print("=" * 60)
    
    print("\nRunning ascent simulation...")
    final_state, log, reason = run_simulation(verbose=True)
    
    print(f"\nSimulation complete: {reason}")
    print(f"Final altitude: {final_state.altitude/1000:.1f} km")
    print(f"Final velocity: {final_state.speed:.1f} m/s")
    
    print("\nGenerating plots...")
    output_dir = "plots"
    saved_files = generate_all_plots(log, output_dir)
    
    print(f"\n{'='*60}")
    print(f"Generated {len(saved_files)} plots in '{output_dir}/':")
    print(f"{'='*60}")
    for i, path in enumerate(saved_files, 1):
        print(f"  {i:2d}. {os.path.basename(path)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
