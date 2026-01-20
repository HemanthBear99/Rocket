"""
RLV Phase-I Ascent Simulation - Publication-Quality Plot Generation

Generates individual, publication-ready figures for research papers.
Each plot is saved as a separate high-resolution PNG file.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from rlv_sim import run_simulation
from rlv_sim import constants as C

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


def plot_altitude_profile(log, save_dir):
    """Plot altitude vs time with key flight events."""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    t = np.array(log.time)
    alt = np.array(log.altitude)
    
    ax.plot(t, alt, 'b-', linewidth=2, label='Altitude')
    ax.fill_between(t, 0, alt, alpha=0.2)
    
    # Mark key events
    ax.scatter(t[0], alt[0], c='green', s=100, marker='o', zorder=5, label='Liftoff')
    ax.scatter(t[-1], alt[-1], c='red', s=100, marker='X', zorder=5, label=f'MECO ({alt[-1]:.1f} km)')
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Altitude (km)')
    ax.set_title('Altitude Profile')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right')
    ax.set_xlim([0, max(t)])
    ax.set_ylim([0, None])
    
    path = os.path.join(save_dir, '01_altitude_profile.png')
    plt.savefig(path)
    print(f"Saved: {path}")
    plt.close()
    return path


def plot_velocity_profile(log, save_dir):
    """Plot velocity vs time (both inertial and relative)."""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    t = np.array(log.time)
    v_inertial = np.array(log.velocity)
    
    # Calculate relative velocity (subtract Earth rotation component)
    # v_rel ≈ v_inertial - 465 m/s (at equator)
    v_initial = v_inertial[0]  # Earth rotation contribution
    v_relative = v_inertial - v_initial
    
    ax.plot(t, v_relative, 'r-', linewidth=2, label='Velocity (relative to launch site)')
    ax.fill_between(t, 0, v_relative, alpha=0.2, color='red')
    
    # Mark key events
    ax.scatter(t[0], v_relative[0], c='green', s=100, marker='o', zorder=5, label='Liftoff (v=0)')
    ax.scatter(t[-1], v_relative[-1], c='red', s=100, marker='X', zorder=5, 
               label=f'MECO ({v_relative[-1]:.0f} m/s)')
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Velocity (m/s)')
    ax.set_title('Velocity Profile (Relative to Launch Site)')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right')
    ax.set_xlim([0, max(t)])
    ax.set_ylim([0, None])
    
    path = os.path.join(save_dir, '02_velocity_profile.png')
    plt.savefig(path)
    print(f"Saved: {path}")
    plt.close()
    return path


def plot_mass_profile(log, save_dir):
    """Plot mass consumption over time."""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    t = np.array(log.time)
    mass = np.array(log.mass)
    
    ax.plot(t, mass/1000, 'g-', linewidth=2, label='Vehicle Mass')
    ax.axhline(y=C.DRY_MASS/1000, color='r', linestyle='--', linewidth=1.5, 
               label=f'Dry Mass ({C.DRY_MASS/1000:.0f} t)')
    ax.fill_between(t, C.DRY_MASS/1000, mass/1000, alpha=0.3, color='green', label='Propellant')
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Mass (tonnes)')
    ax.set_title('Mass Profile')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    ax.set_xlim([0, max(t)])
    
    path = os.path.join(save_dir, '03_mass_profile.png')
    plt.savefig(path)
    print(f"Saved: {path}")
    plt.close()
    return path


def plot_pitch_angle(log, save_dir):
    """Plot guidance pitch angle over time."""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    t = np.array(log.time)
    pitch = np.array(log.pitch_angle)
    
    ax.plot(t, pitch, 'purple', linewidth=2)
    
    # Mark guidance phases
    # Find pitchover region (approximate)
    pitchover_idx = np.argmax(np.array(log.altitude) * 1000 > C.PITCHOVER_START_ALTITUDE)
    gravity_turn_idx = np.argmax(np.array(log.altitude) * 1000 > C.GRAVITY_TURN_START_ALTITUDE)
    
    if pitchover_idx > 0:
        ax.axvline(x=t[pitchover_idx], color='orange', linestyle=':', alpha=0.7, label='Pitchover Start')
    if gravity_turn_idx > 0:
        ax.axvline(x=t[gravity_turn_idx], color='blue', linestyle=':', alpha=0.7, label='Gravity Turn Start')
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Pitch Angle (°)')
    ax.set_title('Guidance Pitch Angle from Vertical')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    ax.set_xlim([0, max(t)])
    ax.set_ylim([0, None])
    
    path = os.path.join(save_dir, '04_pitch_angle.png')
    plt.savefig(path)
    print(f"Saved: {path}")
    plt.close()
    return path


def plot_attitude_error(log, save_dir):
    """Plot attitude tracking error."""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    t = np.array(log.time)
    error = np.array(log.attitude_error)
    
    ax.plot(t, error, 'c-', linewidth=1.5)
    ax.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='1° threshold')
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Attitude Error (°)')
    ax.set_title('Attitude Tracking Error')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_xlim([0, max(t)])
    ax.set_ylim([0, max(error) * 1.1])
    
    path = os.path.join(save_dir, '05_attitude_error.png')
    plt.savefig(path)
    print(f"Saved: {path}")
    plt.close()
    return path


def plot_control_torque(log, save_dir):
    """Plot control torque magnitude."""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    t = np.array(log.time)
    torque = np.array(log.torque_magnitude) / 1e6  # Convert to MN·m
    
    ax.plot(t, torque, 'orange', linewidth=1.5)
    ax.axhline(y=C.MAX_TORQUE/1e6, color='r', linestyle='--', alpha=0.7, 
               label=f'Saturation ({C.MAX_TORQUE/1e6:.1f} MN·m)')
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Control Torque (MN·m)')
    ax.set_title('Control Torque Magnitude')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_xlim([0, max(t)])
    ax.set_ylim([0, None])
    
    path = os.path.join(save_dir, '06_control_torque.png')
    plt.savefig(path)
    print(f"Saved: {path}")
    plt.close()
    return path


def plot_trajectory_local(log, save_dir):
    """Plot trajectory in local coordinates (downrange vs altitude)."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Convert position to local coordinates (relative to launch site)
    x = np.array(log.position_x)
    y = np.array(log.position_y)
    z = np.array(log.position_z)
    t = np.array(log.time)
    
    # Initial position (launch site)
    x0, y0, z0 = x[0], y[0], z[0]
    
    # Local displacements (in km)
    dx = (x - x0) / 1000
    dy = (y - y0) / 1000
    dz = (z - z0) / 1000
    
    # Downrange = horizontal distance from launch site
    downrange = np.sqrt(dy**2 + dz**2)
    
    # Altitude
    alt = np.array(log.altitude)
    
    # Plot with time coloring
    scatter = ax.scatter(downrange, alt, c=t, cmap='viridis', s=3)
    ax.plot(downrange, alt, 'b-', alpha=0.3, linewidth=1)
    
    # Mark events
    ax.scatter(downrange[0], alt[0], c='lime', s=150, marker='o', 
               edgecolors='black', zorder=5, label='Liftoff')
    ax.scatter(downrange[-1], alt[-1], c='red', s=150, marker='X', 
               edgecolors='black', zorder=5, label='MECO')
    
    # Mark gravity turn
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
    
    path = os.path.join(save_dir, '07_trajectory_local.png')
    plt.savefig(path)
    print(f"Saved: {path}")
    plt.close()
    return path


def plot_trajectory_3d_local(log, save_dir):
    """Plot 3D trajectory in local coordinates."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Convert to local coordinates (km)
    x = np.array(log.position_x)
    y = np.array(log.position_y)
    z = np.array(log.position_z)
    t = np.array(log.time)
    
    x_local = (x - x[0]) / 1000
    y_local = (y - y[0]) / 1000
    z_local = (z - z[0]) / 1000
    
    # Color trajectory by time
    colors = plt.cm.viridis(np.linspace(0, 1, len(x_local)))
    for i in range(len(x_local)-1):
        ax.plot(x_local[i:i+2], y_local[i:i+2], z_local[i:i+2], 
                c=colors[i], linewidth=2)
    
    # Markers
    ax.scatter(0, 0, 0, c='lime', s=200, marker='o', 
               label='Liftoff', edgecolors='black')
    ax.scatter(x_local[-1], y_local[-1], z_local[-1], c='red', s=200, 
               marker='X', label='MECO', edgecolors='black')
    
    ax.set_xlabel('Radial (km)')
    ax.set_ylabel('East (km)')
    ax.set_zlabel('North (km)')
    ax.set_title('3D Trajectory (Local Coordinates)')
    ax.legend(loc='upper left')
    
    # Colorbar
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=0, vmax=max(t)))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.1)
    cbar.set_label('Time (s)')
    
    path = os.path.join(save_dir, '08_trajectory_3d_local.png')
    plt.savefig(path)
    print(f"Saved: {path}")
    plt.close()
    return path


def plot_altitude_vs_velocity(log, save_dir):
    """Plot altitude vs velocity phase space."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    t = np.array(log.time)
    alt = np.array(log.altitude)
    vel = np.array(log.velocity)
    
    # Use relative velocity
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
    
    path = os.path.join(save_dir, '09_altitude_vs_velocity.png')
    plt.savefig(path)
    print(f"Saved: {path}")
    plt.close()
    return path


def plot_thrust_acceleration(log, save_dir):
    """Plot thrust and gravity accelerations."""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    t = np.array(log.time)
    mass = np.array(log.mass)
    alt = np.array(log.altitude) * 1000  # Convert to meters
    
    # Thrust acceleration (approximate - assumes constant thrust)
    # In reality this varies with altitude due to Isp changes
    thrust_accel = C.THRUST_MAGNITUDE / mass
    
    # Gravity acceleration
    r = C.R_EARTH + alt
    gravity_accel = C.MU_EARTH / r**2
    
    # Net acceleration
    net_accel = thrust_accel - gravity_accel
    
    ax.plot(t, thrust_accel, 'r-', linewidth=2, label='Thrust Acceleration')
    ax.plot(t, gravity_accel, 'b-', linewidth=2, label='Gravity')
    ax.plot(t, net_accel, 'g--', linewidth=1.5, label='Net (vertical component)')
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Acceleration (m/s²)')
    ax.set_title('Thrust vs Gravity Acceleration')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_xlim([0, max(t)])
    
    path = os.path.join(save_dir, '10_thrust_vs_gravity.png')
    plt.savefig(path)
    print(f"Saved: {path}")
    plt.close()
    return path


def plot_flight_path_angle(log, save_dir):
    """Plot flight path angle - shows the gravity turn from vertical to horizontal."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    t = np.array(log.time)
    x = np.array(log.position_x)
    y = np.array(log.position_y)
    z = np.array(log.position_z)
    
    # Compute velocity components from position derivatives
    vx = np.gradient(x, t)
    vy = np.gradient(y, t)
    vz = np.gradient(z, t)
    
    # Compute radial direction (local vertical) at each point
    r = np.sqrt(x**2 + y**2 + z**2)
    rx, ry, rz = x/r, y/r, z/r  # Unit radial vector
    
    # Radial (vertical) velocity component
    v_radial = vx*rx + vy*ry + vz*rz
    
    # Total velocity magnitude
    v_total = np.sqrt(vx**2 + vy**2 + vz**2)
    
    # Flight path angle = angle of velocity from local horizontal
    # Positive = climbing, Negative = descending, 0 = horizontal
    flight_path_angle = np.degrees(np.arcsin(np.clip(v_radial / v_total, -1, 1)))
    
    # Main plot
    ax.plot(t, flight_path_angle, 'b-', linewidth=2.5, label='Flight Path Angle γ')
    
    # Reference lines
    ax.axhline(y=90, color='green', linestyle='--', alpha=0.5, label='Pure Vertical (90°)')
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='Horizontal (0°)')
    
    # Mark key events
    ax.scatter(t[0], flight_path_angle[0], c='green', s=150, marker='o', 
               zorder=5, edgecolors='black', label=f'Liftoff (γ={flight_path_angle[0]:.1f}°)')
    ax.scatter(t[-1], flight_path_angle[-1], c='red', s=150, marker='X', 
               zorder=5, edgecolors='black', label=f'MECO (γ={flight_path_angle[-1]:.1f}°)')
    
    # Mark max flight path angle
    max_idx = np.argmax(flight_path_angle)
    ax.scatter(t[max_idx], flight_path_angle[max_idx], c='orange', s=120, marker='D',
               zorder=5, edgecolors='black', label=f'Peak (γ={flight_path_angle[max_idx]:.1f}°)')
    
    # Annotate gravity turn region
    gt_idx = np.argmax(np.array(log.altitude) > C.GRAVITY_TURN_START_ALTITUDE/1000)
    if gt_idx > 0:
        ax.axvline(x=t[gt_idx], color='purple', linestyle=':', alpha=0.7)
        ax.annotate('Gravity Turn\nStart', xy=(t[gt_idx], flight_path_angle[gt_idx]),
                    xytext=(t[gt_idx]+15, flight_path_angle[gt_idx]-15),
                    fontsize=10, color='purple',
                    arrowprops=dict(arrowstyle='->', color='purple', alpha=0.7))
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Flight Path Angle γ (°)')
    ax.set_title('Flight Path Angle Evolution\n(γ = angle of velocity from local horizontal)')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right', fontsize=9)
    ax.set_xlim([0, max(t)])
    ax.set_ylim([-5, 100])
    
    # Add explanation text
    ax.text(0.02, 0.98, 
            'Physics Explanation:\n'
            '• At liftoff, inertial velocity is tangential (Earth rotation ~465 m/s)\n'
            '• Thrust adds vertical velocity → γ increases (climbing steeper)\n'
            '• After peak, gravity turn bends trajectory → γ decreases\n'
            '• For orbit: γ → 0° (purely horizontal velocity)',
            transform=ax.transAxes, fontsize=8, 
            verticalalignment='top', 
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    
    path = os.path.join(save_dir, '11_flight_path_angle.png')
    plt.savefig(path)
    print(f"Saved: {path}")
    plt.close()
    return path


def generate_all_plots(log, final_state, save_dir):
    """Generate all publication-quality plots."""
    os.makedirs(save_dir, exist_ok=True)
    
    print("\n" + "="*60)
    print("Generating Publication-Quality Plots")
    print("="*60)
    print(f"Output directory: {save_dir}\n")
    
    saved = []
    saved.append(plot_altitude_profile(log, save_dir))
    saved.append(plot_velocity_profile(log, save_dir))
    saved.append(plot_mass_profile(log, save_dir))
    saved.append(plot_pitch_angle(log, save_dir))
    saved.append(plot_attitude_error(log, save_dir))
    saved.append(plot_control_torque(log, save_dir))
    saved.append(plot_trajectory_local(log, save_dir))
    saved.append(plot_trajectory_3d_local(log, save_dir))
    saved.append(plot_altitude_vs_velocity(log, save_dir))
    saved.append(plot_thrust_acceleration(log, save_dir))
    saved.append(plot_flight_path_angle(log, save_dir))  # NEW
    
    print(f"\n✓ Generated {len(saved)} publication-ready plots")
    return saved


if __name__ == "__main__":
    print("Running standalone plot generation...")
    final_state, log, reason = run_simulation(verbose=True)
    generate_all_plots(log, final_state, 'plots')

