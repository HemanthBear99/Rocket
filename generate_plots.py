"""
RLV Phase-I Ascent Simulation - Publication-Quality Plot Generation

Generates individual, publication-ready figures for research papers.
Each plot is saved as a separate high-resolution PNG file.

Refactored to use base plotting pattern for DRY compliance.
"""

import os
import logging
from typing import Callable, Optional, Any
from functools import wraps

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 - needed for 3D projection

from rlv_sim import run_simulation
from rlv_sim import constants as C

logger = logging.getLogger(__name__)

# Publication style settings
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'legend.fontsize': 11,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})


# =============================================================================
# BASE PLOTTING INFRASTRUCTURE
# =============================================================================

def create_plot(
    filename: str,
    figsize: tuple = (8, 5),
    projection: Optional[str] = None
) -> Callable:
    """
    Decorator factory for creating standardized plots.
    
    Handles common boilerplate:
    - Figure creation with consistent sizing
    - Saving to output directory
    - Closing figure to free memory
    - Error handling and logging
    
    Args:
        filename: Output filename (without directory)
        figsize: Figure size tuple (width, height)
        projection: Optional projection type ('3d' for 3D plots)
    
    Returns:
        Decorator function
    """
    def decorator(plot_func: Callable) -> Callable:
        @wraps(plot_func)
        def wrapper(log: Any, save_dir: str, **kwargs) -> str:
            try:
                fig = plt.figure(figsize=figsize)
                if projection:
                    ax = fig.add_subplot(111, projection=projection)
                else:
                    fig, ax = plt.subplots(figsize=figsize)
                    plt.close()  # Close duplicate from subplots
                    fig = plt.figure(figsize=figsize)
                    ax = fig.add_subplot(111)
                
                # Call the actual plotting function
                plot_func(log, ax, fig, **kwargs)
                
                # Common finalization
                path = os.path.join(save_dir, filename)
                plt.savefig(path)
                logger.info(f"Saved: {path}")
                print(f"Saved: {path}")
                plt.close(fig)
                return path
                
            except Exception as e:
                logger.error(f"Error generating {filename}: {e}")
                raise
                
        return wrapper
    return decorator


def setup_time_axis(ax, t: np.ndarray, xlabel: str = 'Time (s)'):
    """Common time axis setup."""
    ax.set_xlabel(xlabel)
    ax.set_xlim([0, max(t)])
    ax.grid(True, alpha=0.3)


def get_time_array(log) -> np.ndarray:
    """Extract time array from log."""
    return np.array(log.time)


# =============================================================================
# PLOT FUNCTIONS
# =============================================================================

@create_plot('01_altitude_profile.png')
def plot_altitude_profile(log, ax, fig):
    """Plot altitude vs time with key flight events."""
    t = get_time_array(log)
    alt = np.array(log.altitude)
    
    ax.plot(t, alt, 'b-', linewidth=2, label='Altitude')
    ax.fill_between(t, 0, alt, alpha=0.2)
    
    # Mark key events
    ax.scatter(t[0], alt[0], c='green', s=100, marker='o', zorder=5, label='Liftoff')
    ax.scatter(t[-1], alt[-1], c='red', s=100, marker='X', zorder=5, label=f'MECO ({alt[-1]:.1f} km)')
    
    ax.set_ylabel('Altitude (km)')
    ax.set_title('Altitude Profile')
    ax.legend(loc='lower right')
    ax.set_ylim([0, None])
    setup_time_axis(ax, t)


@create_plot('02_velocity_profile.png')
def plot_velocity_profile(log, ax, fig):
    """Plot velocity vs time (relative to launch site)."""
    t = get_time_array(log)
    v_inertial = np.array(log.velocity)
    v_initial = v_inertial[0]
    v_relative = v_inertial - v_initial
    
    ax.plot(t, v_relative, 'r-', linewidth=2, label='Velocity (relative to launch site)')
    ax.fill_between(t, 0, v_relative, alpha=0.2, color='red')
    
    ax.scatter(t[0], v_relative[0], c='green', s=100, marker='o', zorder=5, label='Liftoff (v=0)')
    ax.scatter(t[-1], v_relative[-1], c='red', s=100, marker='X', zorder=5, 
               label=f'MECO ({v_relative[-1]:.0f} m/s)')
    
    ax.set_ylabel('Velocity (m/s)')
    ax.set_title('Velocity Profile (Relative to Launch Site)')
    ax.legend(loc='lower right')
    ax.set_ylim([0, None])
    setup_time_axis(ax, t)


@create_plot('03_mass_profile.png')
def plot_mass_profile(log, ax, fig):
    """Plot mass consumption over time."""
    t = get_time_array(log)
    mass = np.array(log.mass)
    
    ax.plot(t, mass/1000, 'g-', linewidth=2, label='Vehicle Mass')
    ax.axhline(y=C.DRY_MASS/1000, color='r', linestyle='--', linewidth=1.5, 
               label=f'Dry Mass ({C.DRY_MASS/1000:.0f} t)')
    ax.fill_between(t, C.DRY_MASS/1000, mass/1000, alpha=0.3, color='green', label='Propellant')
    
    ax.set_ylabel('Mass (tonnes)')
    ax.set_title('Mass Profile')
    ax.legend(loc='upper right')
    setup_time_axis(ax, t)


@create_plot('04_pitch_angle.png')
def plot_pitch_angle(log, ax, fig):
    """Plot guidance pitch angle over time."""
    t = get_time_array(log)
    pitch = np.array(log.pitch_angle)
    
    ax.plot(t, pitch, 'purple', linewidth=2)
    
    # Mark guidance phases
    pitchover_idx = np.argmax(np.array(log.altitude) * 1000 > C.PITCHOVER_START_ALTITUDE)
    gravity_turn_idx = np.argmax(np.array(log.altitude) * 1000 > C.GRAVITY_TURN_START_ALTITUDE)
    
    if pitchover_idx > 0:
        ax.axvline(x=t[pitchover_idx], color='orange', linestyle=':', alpha=0.7, label='Pitchover Start')
    if gravity_turn_idx > 0:
        ax.axvline(x=t[gravity_turn_idx], color='blue', linestyle=':', alpha=0.7, label='Gravity Turn Start')
    
    ax.set_ylabel('Pitch Angle (°)')
    ax.set_title('Guidance Pitch Angle from Vertical')
    ax.legend(loc='upper right')
    ax.set_ylim([0, None])
    setup_time_axis(ax, t)


@create_plot('05_attitude_error.png')
def plot_attitude_error(log, ax, fig):
    """Plot attitude tracking error."""
    t = get_time_array(log)
    error = np.array(log.attitude_error)
    
    ax.plot(t, error, 'c-', linewidth=1.5)
    ax.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='1° threshold')
    
    ax.set_ylabel('Attitude Error (°)')
    ax.set_title('Attitude Tracking Error')
    ax.legend()
    ax.set_ylim([0, max(error) * 1.1])
    setup_time_axis(ax, t)


@create_plot('06_control_torque.png')
def plot_control_torque(log, ax, fig):
    """Plot control torque magnitude."""
    t = get_time_array(log)
    torque = np.array(log.torque_magnitude) / 1e6
    
    ax.plot(t, torque, 'orange', linewidth=1.5)
    ax.axhline(y=C.MAX_TORQUE/1e6, color='r', linestyle='--', alpha=0.7, 
               label=f'Saturation ({C.MAX_TORQUE/1e6:.1f} MN·m)')
    
    ax.set_ylabel('Control Torque (MN·m)')
    ax.set_title('Control Torque Magnitude')
    ax.legend()
    ax.set_ylim([0, None])
    setup_time_axis(ax, t)


@create_plot('07_trajectory_local.png', figsize=(10, 6))
def plot_trajectory_local(log, ax, fig):
    """Plot trajectory in local coordinates (downrange vs altitude)."""
    t = get_time_array(log)
    x = np.array(log.position_x)
    y = np.array(log.position_y)
    z = np.array(log.position_z)
    
    x0, y0, z0 = x[0], y[0], z[0]
    dy = (y - y0) / 1000
    dz = (z - z0) / 1000
    downrange = np.sqrt(dy**2 + dz**2)
    alt = np.array(log.altitude)
    
    scatter = ax.scatter(downrange, alt, c=t, cmap='viridis', s=3)
    ax.plot(downrange, alt, 'b-', alpha=0.3, linewidth=1)
    
    ax.scatter(downrange[0], alt[0], c='lime', s=150, marker='o', 
               edgecolors='black', zorder=5, label='Liftoff')
    ax.scatter(downrange[-1], alt[-1], c='red', s=150, marker='X', 
               edgecolors='black', zorder=5, label='MECO')
    
    gt_idx = np.argmax(alt > C.GRAVITY_TURN_START_ALTITUDE/1000)
    if gt_idx > 0:
        ax.scatter(downrange[gt_idx], alt[gt_idx], c='orange', s=100, marker='s',
                   edgecolors='black', zorder=5, label='Gravity Turn Start')
    
    ax.set_xlabel('Downrange Distance (km)')
    ax.set_ylabel('Altitude (km)')
    ax.set_title('Trajectory: Altitude vs Downrange')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right')
    ax.set_xlim([0, None])
    ax.set_ylim([0, None])
    
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label('Time (s)')


@create_plot('08_trajectory_3d_local.png', figsize=(10, 8), projection='3d')
def plot_trajectory_3d_local(log, ax, fig):
    """Plot 3D trajectory in local coordinates."""
    t = get_time_array(log)
    x = np.array(log.position_x)
    y = np.array(log.position_y)
    z = np.array(log.position_z)
    
    x_local = (x - x[0]) / 1000
    y_local = (y - y[0]) / 1000
    z_local = (z - z[0]) / 1000
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(x_local)))
    for i in range(len(x_local)-1):
        ax.plot(x_local[i:i+2], y_local[i:i+2], z_local[i:i+2], c=colors[i], linewidth=2)
    
    ax.scatter(0, 0, 0, c='lime', s=200, marker='o', label='Liftoff', edgecolors='black')
    ax.scatter(x_local[-1], y_local[-1], z_local[-1], c='red', s=200, 
               marker='X', label='MECO', edgecolors='black')
    
    ax.set_xlabel('Radial (km)')
    ax.set_ylabel('East (km)')
    ax.set_zlabel('North (km)')
    ax.set_title('3D Trajectory (Local Coordinates)')
    ax.legend(loc='upper left')
    
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=0, vmax=max(t)))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.1)
    cbar.set_label('Time (s)')


@create_plot('09_altitude_vs_velocity.png', figsize=(8, 6))
def plot_altitude_vs_velocity(log, ax, fig):
    """Plot altitude vs velocity phase space."""
    t = get_time_array(log)
    alt = np.array(log.altitude)
    vel = np.array(log.velocity)
    v_relative = vel - vel[0]
    
    scatter = ax.scatter(v_relative, alt, c=t, cmap='viridis', s=5)
    ax.plot(v_relative, alt, 'b-', alpha=0.2, linewidth=0.5)
    
    ax.scatter(v_relative[0], alt[0], c='lime', s=200, marker='o', 
               edgecolors='black', zorder=5, label='Liftoff')
    ax.scatter(v_relative[-1], alt[-1], c='red', s=200, marker='X', 
               edgecolors='black', zorder=5, label='MECO')
    
    ax.set_xlabel('Velocity Gained (m/s)')
    ax.set_ylabel('Altitude (km)')
    ax.set_title('Altitude vs Velocity (Phase Space)')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_xlim([0, None])
    ax.set_ylim([0, None])
    
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label('Time (s)')


@create_plot('10_thrust_vs_gravity.png')
def plot_thrust_acceleration(log, ax, fig):
    """Plot thrust and gravity accelerations."""
    t = get_time_array(log)
    mass = np.array(log.mass)
    alt = np.array(log.altitude) * 1000
    
    thrust_accel = C.THRUST_MAGNITUDE / mass
    r = C.R_EARTH + alt
    gravity_accel = C.MU_EARTH / r**2
    net_accel = thrust_accel - gravity_accel
    
    ax.plot(t, thrust_accel, 'r-', linewidth=2, label='Thrust Acceleration')
    ax.plot(t, gravity_accel, 'b-', linewidth=2, label='Gravity')
    ax.plot(t, net_accel, 'g--', linewidth=1.5, label='Net (vertical component)')
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    
    ax.set_ylabel('Acceleration (m/s²)')
    ax.set_title('Thrust vs Gravity Acceleration')
    ax.legend()
    setup_time_axis(ax, t)


@create_plot('11_flight_path_angle.png', figsize=(10, 6))
def plot_flight_path_angle(log, ax, fig):
    """Plot flight path angle evolution."""
    t = get_time_array(log)
    x = np.array(log.position_x)
    y = np.array(log.position_y)
    z = np.array(log.position_z)
    
    vx = np.gradient(x, t)
    vy = np.gradient(y, t)
    vz = np.gradient(z, t)
    
    r = np.sqrt(x**2 + y**2 + z**2)
    rx, ry, rz = x/r, y/r, z/r
    v_radial = vx*rx + vy*ry + vz*rz
    v_total = np.sqrt(vx**2 + vy**2 + vz**2)
    
    flight_path_angle = np.degrees(np.arcsin(np.clip(v_radial / v_total, -1, 1)))
    
    ax.plot(t, flight_path_angle, 'b-', linewidth=2.5, label='Flight Path Angle γ')
    ax.axhline(y=90, color='green', linestyle='--', alpha=0.5, label='Pure Vertical (90°)')
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='Horizontal (0°)')
    
    ax.scatter(t[0], flight_path_angle[0], c='green', s=150, marker='o', 
               zorder=5, edgecolors='black', label=f'Liftoff (γ={flight_path_angle[0]:.1f}°)')
    ax.scatter(t[-1], flight_path_angle[-1], c='red', s=150, marker='X', 
               zorder=5, edgecolors='black', label=f'MECO (γ={flight_path_angle[-1]:.1f}°)')
    
    max_idx = np.argmax(flight_path_angle)
    ax.scatter(t[max_idx], flight_path_angle[max_idx], c='orange', s=120, marker='D',
               zorder=5, edgecolors='black', label=f'Peak (γ={flight_path_angle[max_idx]:.1f}°)')
    
    gt_idx = np.argmax(np.array(log.altitude) > C.GRAVITY_TURN_START_ALTITUDE/1000)
    if gt_idx > 0:
        ax.axvline(x=t[gt_idx], color='purple', linestyle=':', alpha=0.7)
        ax.annotate('Gravity Turn\nStart', xy=(t[gt_idx], flight_path_angle[gt_idx]),
                    xytext=(t[gt_idx]+15, flight_path_angle[gt_idx]-15),
                    fontsize=10, color='purple',
                    arrowprops=dict(arrowstyle='->', color='purple', alpha=0.7))
    
    ax.set_ylabel('Flight Path Angle γ (°)')
    ax.set_title('Flight Path Angle Evolution\n(γ = angle of velocity from local horizontal)')
    ax.legend(loc='lower right', fontsize=9)
    ax.set_ylim([-5, 100])
    setup_time_axis(ax, t)
    
    ax.text(0.02, 0.98, 
            'Physics Explanation:\n'
            '• At liftoff, inertial velocity is tangential (Earth rotation ~465 m/s)\n'
            '• Thrust adds vertical velocity → γ increases (climbing steeper)\n'
            '• After peak, gravity turn bends trajectory → γ decreases\n'
            '• For orbit: γ → 0° (purely horizontal velocity)',
            transform=ax.transAxes, fontsize=8, 
            verticalalignment='top', 
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))


# =============================================================================
# MAIN GENERATION FUNCTION
# =============================================================================

def generate_all_plots(log, final_state, save_dir: str) -> list:
    """Generate all publication-quality plots."""
    os.makedirs(save_dir, exist_ok=True)
    
    print("\n" + "="*60)
    print("Generating Publication-Quality Plots")
    print("="*60)
    print(f"Output directory: {save_dir}\n")
    
    plot_functions = [
        plot_altitude_profile,
        plot_velocity_profile,
        plot_mass_profile,
        plot_pitch_angle,
        plot_attitude_error,
        plot_control_torque,
        plot_trajectory_local,
        plot_trajectory_3d_local,
        plot_altitude_vs_velocity,
        plot_thrust_acceleration,
        plot_flight_path_angle,
    ]
    
    saved = []
    for plot_func in plot_functions:
        try:
            path = plot_func(log, save_dir)
            saved.append(path)
        except Exception as e:
            logger.error(f"Failed to generate {plot_func.__name__}: {e}")
            print(f"[ERROR] {plot_func.__name__}: {e}")
    
    print(f"\n✓ Generated {len(saved)} publication-ready plots")
    return saved


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Running standalone plot generation...")
    final_state, log, reason = run_simulation(verbose=True)
    generate_all_plots(log, final_state, 'plots')
