"""
RLV Phase-I Ascent Simulation - Plot Generation

Generates trajectory and telemetry visualization plots matching the reference style.
"""

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from rlv_sim.main import run_simulation
from rlv_sim import constants as C


def setup_style():
    """Configure matplotlib for professional aerospace plots."""
    plt.rcParams.update({
        'figure.figsize': (10, 7),
        'figure.dpi': 150,
        'axes.grid': True,
        'axes.axisbelow': True,
        'grid.alpha': 0.3,
        'grid.linestyle': '-',
        'font.size': 11,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'legend.fontsize': 10,
        'legend.framealpha': 0.9,
        'lines.linewidth': 2,
    })


def generate_all_plots(log, final_state, output_dir: str = "plots"):
    """Generate all trajectory and telemetry plots matching reference style."""
    os.makedirs(output_dir, exist_ok=True)
    setup_style()
    saved_files = []
    
    # Convert lists to arrays for easier manipulation
    time = np.array(log.time)
    altitude = np.array(log.altitude)  # already in km
    velocity = np.array(log.velocity)
    mass = np.array(log.mass)
    pitch_angle = np.array(log.pitch_angle)
    attitude_error = np.array(log.attitude_error)
    torque = np.array(log.torque_magnitude)
    pos_x = np.array(log.position_x)
    pos_y = np.array(log.position_y)
    pos_z = np.array(log.position_z)
    vel_x = np.array(log.velocity_x)
    vel_y = np.array(log.velocity_y)
    vel_z = np.array(log.velocity_z)
    
    # Key events
    liftoff_idx = 0
    meco_idx = len(time) - 1
    meco_alt = altitude[meco_idx]
    meco_vel = velocity[meco_idx]
    meco_time = time[meco_idx]
    
    # Compute derived quantities
    # 1. Relative Velocity (Airspeed)
    # v_rel = v_inertial - (omega_earth x r)
    omega_vec = np.array([0.0, 0.0, C.EARTH_ROTATION_RATE])
    
    # Position and Velocity arrays
    pos_array = np.column_stack((pos_x, pos_y, pos_z))
    vel_array = np.column_stack((vel_x, vel_y, vel_z))
    
    # Calculate Relative Velocity
    cross_term = np.cross(omega_vec, pos_array) 
    v_rel_array = vel_array - cross_term
    v_rel_mag = np.linalg.norm(v_rel_array, axis=1)
    
    # 2. Flight Path Angle (Relative vs Inertial)
    
    # Helper: Angle from Vertical (Zenith)
    # Zenith usually defined as Radial Outward (r / |r|)
    r_mag = np.linalg.norm(pos_array, axis=1)
    r_hat = pos_array / r_mag[:, np.newaxis]
    
    # -- Relative FPA --
    # Angle between v_rel and Local Horizontal
    # sin(gamma_rel) = (v_rel . r_hat) / |v_rel|
    v_rel_radial = np.sum(v_rel_array * r_hat, axis=1)
    sin_gamma_rel = np.clip(v_rel_radial / np.maximum(v_rel_mag, 1.0), -1.0, 1.0)
    gamma_rel = np.degrees(np.arcsin(sin_gamma_rel)) # -90 to 90 (0 is horizontal)
    
    # Angle related to Vertical (for Sideways Flight Check)
    # Theta_v_rel (angle from vertical) = 90 - gamma_rel
    # cos(theta_v) = (v_rel . r_hat) / |v_rel|
    cos_theta_rel = np.clip(v_rel_radial / np.maximum(v_rel_mag, 1.0), -1.0, 1.0)
    theta_rel_from_vertical = np.degrees(np.arccos(cos_theta_rel)) # 0 to 180 (0 is up)

    # -- Inertial FPA -- (For comparison/debugging)
    # Angle between v_inertial and Local Horizontal
    v_mag = np.linalg.norm(vel_array, axis=1)
    v_radial = np.sum(vel_array * r_hat, axis=1)
    sin_gamma_in = np.clip(v_radial / np.maximum(v_mag, 1.0), -1.0, 1.0)
    gamma_in = np.degrees(np.arcsin(sin_gamma_in))
    
    # 3. Dynamic Pressure (Q)
    q_dynamic = []
    
    # Imports for calculations
    from rlv_sim.forces import compute_atmosphere_properties
    
    for h, v_r in zip(altitude, v_rel_mag):
        h_m = h * 1000.0
        _, _, rho, _ = compute_atmosphere_properties(h_m)
        q = 0.5 * rho * v_r**2
        q_dynamic.append(q)
    
    q_dynamic = np.array(q_dynamic)
    
    # Gravity turn start logic
    pitch_diff = np.abs(np.diff(pitch_angle))
    gravity_turn_idx = np.argmax(pitch_diff > 0.1) if np.any(pitch_diff > 0.1) else 20
    gravity_turn_time = time[gravity_turn_idx]
    
    # =========================================================================
    # 01. Altitude Profile
    # =========================================================================
    fig, ax = plt.subplots()
    ax.fill_between(time, 0, altitude, alpha=0.3, color='blue')
    ax.plot(time, altitude, 'b-', linewidth=2, label='Altitude')
    ax.scatter([time[liftoff_idx]], [altitude[liftoff_idx]], 
               c='green', s=100, marker='o', zorder=5, label='Liftoff')
    ax.scatter([time[meco_idx]], [altitude[meco_idx]], 
               c='red', s=100, marker='x', zorder=5, label=f'MECO ({meco_alt:.1f} km)')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Altitude (km)')
    ax.set_title('Altitude Profile', fontweight='bold', style='italic')
    ax.legend(loc='lower right')
    ax.set_xlim(0, time[-1])
    ax.set_ylim(0, None)
    plt.tight_layout()
    path = os.path.join(output_dir, '01_altitude_profile.png')
    fig.savefig(path, bbox_inches='tight')
    saved_files.append(path)
    plt.close(fig)
    
    # =========================================================================
    # 02. Velocity Profile (Inertial vs Relative)
    # =========================================================================
    fig, ax = plt.subplots()
    
    # Plot Inertial (Red)
    ax.plot(time, velocity, 'r-', linewidth=2, label='Inertial Velocity (Earth Centered)')
    
    # Plot Relative (Green)
    ax.plot(time, v_rel_mag, 'g--', linewidth=2, label='Relative Velocity (Airspeed)')
    
    ax.scatter([time[liftoff_idx]], [velocity[liftoff_idx]], 
               c='red', s=50, marker='o', zorder=5, label=f'Liftoff Inertial ({velocity[liftoff_idx]:.0f} m/s)')
    ax.scatter([time[liftoff_idx]], [v_rel_mag[liftoff_idx]], 
               c='green', s=50, marker='o', zorder=5, label=f'Liftoff Relative ({v_rel_mag[liftoff_idx]:.0f} m/s)')
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Velocity (m/s)')
    ax.set_title('Velocity Profile: Inertial vs Relative', fontweight='bold', style='italic')
    ax.legend(loc='lower right')
    ax.set_xlim(0, time[-1])
    ax.set_ylim(0, None)
    
    # Add text note
    note = "Note: Difference is due to Earth Rotation (~465 m/s at Equator)"
    ax.text(0.02, 0.95, note, transform=ax.transAxes, fontsize=9, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    path = os.path.join(output_dir, '02_velocity_profile.png')
    fig.savefig(path, bbox_inches='tight')
    saved_files.append(path)
    plt.close(fig)
    
    # =========================================================================
    # 03. Mass Profile
    # =========================================================================
    fig, ax = plt.subplots()
    mass_tonnes = mass / 1000
    ax.fill_between(time, 0, mass_tonnes, alpha=0.3, color='green')
    ax.plot(time, mass_tonnes, 'g-', linewidth=2, label='Vehicle Mass')
    ax.axhline(y=C.DRY_MASS/1000, color='orange', linestyle='--', linewidth=1.5, 
               label=f'Dry Mass ({C.DRY_MASS/1000:.0f} t)')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Mass (tonnes)')
    ax.set_title('Mass Profile', fontweight='bold', style='italic')
    ax.legend(loc='upper right')
    ax.set_xlim(0, time[-1])
    ax.set_ylim(0, None)
    plt.tight_layout()
    path = os.path.join(output_dir, '03_mass_profile.png')
    fig.savefig(path, bbox_inches='tight')
    saved_files.append(path)
    plt.close(fig)
    
    # =========================================================================
    # 04. Pitch Angle (Gravity Turn)
    # =========================================================================
    fig, ax = plt.subplots()
    ax.fill_between(time, 0, pitch_angle, alpha=0.3, color='purple')
    ax.plot(time, pitch_angle, color='purple', linewidth=2, label='Pitch Angle (from Vertical)')
    ax.axvline(x=gravity_turn_time, color='gray', linestyle=':', linewidth=1.5,
               label='Gravity Turn Start')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Pitch Angle (deg from Vertical)')
    ax.set_title('Pitch Angle Evolution', fontweight='bold', style='italic')
    ax.legend(loc='upper left')
    ax.set_xlim(0, time[-1])
    ax.set_ylim(0, None)
    plt.tight_layout()
    path = os.path.join(output_dir, '04_pitch_angle.png')
    fig.savefig(path, bbox_inches='tight')
    saved_files.append(path)
    plt.close(fig)
    
    # =========================================================================
    # 05. Attitude Error
    # =========================================================================
    fig, ax = plt.subplots()
    ax.plot(time, attitude_error, 'c-', linewidth=1.5, label='Attitude Error')
    ax.axhline(y=1.0, color='orange', linestyle='--', label='1° Threshold')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Error (degrees)')
    ax.set_title('Attitude Control Error', fontweight='bold', style='italic')
    ax.set_xlim(0, time[-1])
    ax.legend()
    plt.tight_layout()
    path = os.path.join(output_dir, '05_attitude_error.png')
    fig.savefig(path, bbox_inches='tight')
    saved_files.append(path)
    plt.close(fig)
    
    # =========================================================================
    # 06. Control Torque
    # =========================================================================
    fig, ax = plt.subplots()
    torque_mn = torque / 1e6  # Convert to MN·m
    ax.plot(time, torque_mn, color='orange', linewidth=1.5, label='Control Torque')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Torque (MN·m)')
    ax.set_title('Control Torque Magnitude', fontweight='bold', style='italic')
    plt.tight_layout()
    path = os.path.join(output_dir, '06_control_torque.png')
    fig.savefig(path, bbox_inches='tight')
    saved_files.append(path)
    plt.close(fig)
    
    # =========================================================================
    # 07. Trajectory Local
    # =========================================================================
    fig, ax = plt.subplots()
    downrange = np.sqrt((pos_x - pos_x[0])**2 + (pos_y - pos_y[0])**2) / 1000
    ax.plot(downrange, altitude, 'b-', linewidth=2, label='Ascent Trajectory')
    ax.set_xlabel('Downrange (km)')
    ax.set_ylabel('Altitude (km)')
    ax.set_title('Trajectory (Local Frame)', fontweight='bold', style='italic')
    ax.grid(True)
    plt.tight_layout()
    path = os.path.join(output_dir, '07_trajectory_local.png')
    fig.savefig(path, bbox_inches='tight')
    saved_files.append(path)
    plt.close(fig)
    
    # =========================================================================
    # 08. 3D Trajectory
    # =========================================================================
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    east = (pos_y - pos_y[0]) / 1000
    north = (pos_x - pos_x[0]) / 1000
    ax.plot(east, north, altitude, linewidth=2)
    ax.set_xlabel('East (km)')
    ax.set_ylabel('North (km)')
    ax.set_zlabel('Altitude (km)')
    ax.set_title('3D Trajectory', fontweight='bold')
    path = os.path.join(output_dir, '08_trajectory_3d_local.png')
    fig.savefig(path, bbox_inches='tight')
    saved_files.append(path)
    plt.close(fig)
    
    # =========================================================================
    # 10. Thrust vs Gravity
    # =========================================================================
    fig, ax = plt.subplots()
    thrust_force = np.ones_like(time) * C.THRUST_MAGNITUDE / 1e6
    r_center = altitude * 1000.0 + C.R_EARTH
    gravity_force = (C.MU_EARTH * mass / (r_center**2)) / 1e6
    
    ax.plot(time, thrust_force, 'r-', label='Thrust')
    ax.plot(time, gravity_force, 'b-', label='Weight')
    ax.set_ylabel('Force (MN)')
    ax.set_xlabel('Time (s)')
    ax.legend()
    path = os.path.join(output_dir, '10_thrust_vs_gravity.png')
    fig.savefig(path, bbox_inches='tight')
    saved_files.append(path)
    plt.close(fig)

    # =========================================================================
    # 11. Flight Path Angle (Relative vs Inertial)
    # =========================================================================
    fig, ax = plt.subplots()
    
    # Plot Relative (Primary physics metric)
    ax.plot(time, gamma_rel, 'b-', linewidth=2.5, label='Relative $\gamma$ (vs Horizontal)')
    
    # Plot Inertial (Dashed)
    ax.plot(time, gamma_in, 'r--', linewidth=1.5, alpha=0.7, label='Inertial $\gamma$')
    
    ax.axhline(y=90, color='gray', linestyle=':', alpha=0.5)
    ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Flight Path Angle $\gamma$ (degrees from Horizontal)')
    ax.set_title('Flight Path Angle: Relative vs Inertial\n($\gamma$ = angle from local horizontal; 90° = vertical climb, 0° = level flight)', 
                 fontweight='bold', style='italic')
    ax.legend(loc='center right')
    ax.set_ylim(-5, 100)
    ax.set_xlim(0, time[-1])
    
    # Physics explanation box
    textstr = ('Physics Definition:\n'
               '• $\gamma$ = angle from LOCAL HORIZONTAL\n'
               '• 90° = vertical climb (straight up)\n'
               '• 0° = horizontal flight (level)\n'
               '• Gravity turn: $\gamma$ decreases toward 0°')
    props = dict(boxstyle='round', facecolor='lightyellow', alpha=0.9)
    ax.text(0.02, 0.35, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=props)
    plt.tight_layout()
    path = os.path.join(output_dir, '11_flight_path_angle.png')
    fig.savefig(path, bbox_inches='tight')
    saved_files.append(path)
    plt.close(fig)
    
    # =========================================================================
    # 12. Dynamic Pressure
    # =========================================================================
    fig, ax = plt.subplots()
    q_kpa = q_dynamic / 1000.0
    limit_kpa = C.MAX_DYNAMIC_PRESSURE / 1000.0
    
    ax.plot(time, q_kpa, 'purple', linewidth=2, label='Dynamic Pressure')
    ax.axhline(y=limit_kpa, color='red', linestyle='--', label='Limit')
    ax.set_ylabel('Q (kPa)')
    ax.set_xlabel('Time (s)')
    ax.legend()
    path = os.path.join(output_dir, '12_dynamic_pressure.png')
    fig.savefig(path, bbox_inches='tight')
    saved_files.append(path)
    plt.close(fig)

    # =========================================================================
    # 13. PHYSICS CHECK: Angle of Attack Analysis
    # =========================================================================
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    
    # Subplot 1: Pitch Angle vs Velocity Angle (From Vertical)
    # This directly answers the "Sideways Flight" contradiction.
    # If Pitch ~ 5 deg and Velocity Angle ~ 5 deg, then alpha ~ 0 (GOOD).
    # If Pitch ~ 5 deg and Velocity Angle ~ 35 deg, then alpha ~ 30 (BAD).
    
    ax1.plot(time, pitch_angle, 'purple', linewidth=2, label='Vehicle Pitch $\\theta$ (from Vertical)')
    ax1.plot(time, theta_rel_from_vertical, 'g--', linewidth=2, label='Velocity Vector (from Vertical)')
    ax1.set_ylabel('Angle from Vertical (deg)')
    ax1.set_title('Physics Check: Alignment of Attitude and Velocity', fontweight='bold')
    ax1.legend(loc='upper left')
    ax1.grid(True)
    
    # Subplot 2: Approx Angle of Attack (Alpha)
    # alpha = abs(theta - theta_v) approx
    # Actually, dot product is more accurate generally, but in 2D approx:
    alpha_est = np.abs(pitch_angle - theta_rel_from_vertical)
    
    ax2.plot(time, alpha_est, 'r-', linewidth=2, label='Estimated Angle of Attack $|\\alpha|$')
    ax2.axhline(y=0.0, color='k', linestyle='-')
    ax2.axhline(y=10.0, color='orange', linestyle='--', label='10° Awareness')
    ax2.set_ylabel('Alpha (deg)')
    ax2.set_xlabel('Time (s)')
    ax2.set_title('Angle of Attack Estimate', fontweight='bold')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    path = os.path.join(output_dir, '13_physics_check.png')
    fig.savefig(path, bbox_inches='tight')
    saved_files.append(path)
    plt.close(fig)
    
    # =========================================================================
    # 14. DIAGNOSTIC: Pitch θ vs Velocity Tilt atan(vh/vz)
    # =========================================================================
    # This plot addresses the concern about thrust/attitude coupling:
    # If pitch ~ 0-3° from vertical, but trajectory moves 70km East + 110km North,
    # there may be a mismatch in angle definitions or thrust direction.
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    
    # Compute velocity components in local frame
    # Use relative velocity for local-frame analysis
    # vh = horizontal component, vz = vertical (radial) component
    v_horizontal = np.sqrt(v_rel_mag**2 - v_rel_radial**2)  # Tangential component
    v_vertical = v_rel_radial  # Radial (upward) component
    
    # Velocity tilt angle: atan(vh / vz) - angle from vertical
    # When vz >> vh, angle is small (nearly vertical climb)
    # When vh >> vz, angle is large (nearly horizontal)
    velocity_tilt = np.degrees(np.arctan2(v_horizontal, np.maximum(v_vertical, 0.1)))
    
    # Subplot 1: Pitch vs Velocity Tilt (both from vertical)
    ax1.plot(time, pitch_angle, 'purple', linewidth=2.5, label='Pitch $\\theta$ (from Vertical)')
    ax1.plot(time, velocity_tilt, 'g--', linewidth=2, label='Velocity Tilt atan($v_h/v_z$)')
    ax1.set_ylabel('Angle from Vertical (deg)')
    ax1.set_title('Diagnostic: Pitch Angle vs Velocity Tilt Direction', fontweight='bold')
    ax1.legend(loc='upper left')
    ax1.grid(True)
    ax1.set_ylim(0, max(90, np.max(velocity_tilt) * 1.1))
    
    # Subplot 2: Velocity Components
    ax2.plot(time, v_horizontal, 'b-', linewidth=2, label='Horizontal Velocity $v_h$')
    ax2.plot(time, v_vertical, 'r-', linewidth=2, label='Vertical Velocity $v_z$ (radial)')
    ax2.set_ylabel('Velocity (m/s)')
    ax2.set_title('Velocity Components (Relative to Earth Surface)', fontweight='bold')
    ax2.legend(loc='upper left')
    ax2.grid(True)
    
    # Subplot 3: Downrange motion analysis
    downrange_east = (pos_y - pos_y[0]) / 1000  # East
    downrange_north = (pos_x - pos_x[0]) / 1000  # North
    downrange_total = np.sqrt(downrange_east**2 + downrange_north**2)
    
    ax3.plot(time, downrange_total, 'b-', linewidth=2, label='Total Downrange')
    ax3.plot(time, downrange_east, 'c--', linewidth=1.5, label='East')
    ax3.plot(time, downrange_north, 'm--', linewidth=1.5, label='North')
    ax3.set_ylabel('Distance (km)')
    ax3.set_xlabel('Time (s)')
    ax3.set_title('Downrange Motion Components', fontweight='bold')
    ax3.legend(loc='upper left')
    ax3.grid(True)
    
    # Add summary annotation
    final_east = downrange_east[-1]
    final_north = downrange_north[-1]
    summary_text = (f'MECO Summary:\n'
                    f'• East: {final_east:.1f} km\n'
                    f'• North: {final_north:.1f} km\n'
                    f'• Total Downrange: {downrange_total[-1]:.1f} km\n'
                    f'• Final Pitch: {pitch_angle[-1]:.1f}° from vertical\n'
                    f'• Final Velocity Tilt: {velocity_tilt[-1]:.1f}° from vertical')
    props = dict(boxstyle='round', facecolor='lightyellow', alpha=0.9)
    ax3.text(0.98, 0.95, summary_text, transform=ax3.transAxes, fontsize=9,
             verticalalignment='top', ha='right', bbox=props)
    
    plt.tight_layout()
    path = os.path.join(output_dir, '14_pitch_vs_velocity_tilt.png')
    fig.savefig(path, bbox_inches='tight')
    saved_files.append(path)
    plt.close(fig)
    
    # =========================================================================
    # Combined Dashboard - Ascent Profile
    # =========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('RLV Phase-I Ascent Profile', fontsize=16, fontweight='bold')
    
    # Altitude
    axes[0,0].fill_between(time, 0, altitude, alpha=0.3, color='blue')
    axes[0,0].plot(time, altitude, 'b-', linewidth=2)
    axes[0,0].scatter([time[-1]], [altitude[-1]], c='red', s=80, marker='x')
    axes[0,0].set_xlabel('Time (s)')
    axes[0,0].set_ylabel('Altitude (km)')
    axes[0,0].set_title('Altitude')
    
    # Velocity
    axes[0,1].fill_between(time, 0, velocity, alpha=0.3, color='red')
    axes[0,1].plot(time, velocity, 'r-', linewidth=2)
    axes[0,1].scatter([time[-1]], [velocity[-1]], c='red', s=80, marker='x')
    axes[0,1].set_xlabel('Time (s)')
    axes[0,1].set_ylabel('Velocity (m/s)')
    axes[0,1].set_title('Velocity')
    
    # Mass
    axes[1,0].fill_between(time, 0, mass_tonnes, alpha=0.3, color='green')
    axes[1,0].plot(time, mass_tonnes, 'g-', linewidth=2)
    axes[1,0].axhline(y=C.DRY_MASS/1000, color='orange', linestyle='--')
    axes[1,0].set_xlabel('Time (s)')
    axes[1,0].set_ylabel('Mass (tonnes)')
    axes[1,0].set_title('Mass')
    
    # Pitch
    axes[1,1].fill_between(time, 0, pitch_angle, alpha=0.3, color='purple')
    axes[1,1].plot(time, pitch_angle, color='purple', linewidth=2)
    axes[1,1].set_xlabel('Time (s)')
    axes[1,1].set_ylabel('Pitch (deg)')
    axes[1,1].set_title('Pitch Angle')
    
    plt.tight_layout()
    path = os.path.join(output_dir, 'ascent_profile.png')
    fig.savefig(path, bbox_inches='tight')
    saved_files.append(path)
    plt.close(fig)
    
    # =========================================================================
    # Combined Dashboard - Control Dynamics
    # =========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('RLV Phase-I Control Dynamics', fontsize=16, fontweight='bold')
    
    # Attitude Error
    axes[0,0].fill_between(time, 0, attitude_error, alpha=0.3, color='cyan')
    axes[0,0].plot(time, attitude_error, 'c-', linewidth=1.5)
    axes[0,0].axhline(y=1.0, color='orange', linestyle='--')
    axes[0,0].set_xlabel('Time (s)')
    axes[0,0].set_ylabel('Error (deg)')
    axes[0,0].set_title('Attitude Error')
    
    # Control Torque
    axes[0,1].fill_between(time, 0, torque_mn, alpha=0.3, color='orange')
    axes[0,1].plot(time, torque_mn, color='orange', linewidth=1.5)
    axes[0,1].set_xlabel('Time (s)')
    axes[0,1].set_ylabel('Torque (MN·m)')
    axes[0,1].set_title('Control Torque')
    
    # Pitch Angle
    axes[1,0].fill_between(time, 0, pitch_angle, alpha=0.3, color='purple')
    axes[1,0].plot(time, pitch_angle, color='purple', linewidth=2)
    axes[1,0].set_xlabel('Time (s)')
    axes[1,0].set_ylabel('Pitch (deg)')
    axes[1,0].set_title('Pitch Command')
    
    # Flight Path Angle
    axes[1,1].fill_between(time, 0, gamma_rel, alpha=0.3, color='blue')
    axes[1,1].plot(time, gamma_rel, 'b-', linewidth=2)
    axes[1,1].set_xlabel('Time (s)')
    axes[1,1].set_ylabel('γ (deg)')
    axes[1,1].set_title('Flight Path Angle')
    
    plt.tight_layout()
    path = os.path.join(output_dir, 'control_dynamics.png')
    fig.savefig(path, bbox_inches='tight')
    saved_files.append(path)
    plt.close(fig)
    
    # =========================================================================
    # Trajectory Overview
    # =========================================================================
    fig, ax = plt.subplots(figsize=(12, 8))
    scatter = ax.scatter(downrange, altitude, c=time, cmap='viridis', s=3, label='Trajectory')
    cbar = plt.colorbar(scatter, ax=ax, label='Time (s)')
    ax.scatter([downrange[0]], [altitude[0]], c='green', s=150, marker='o', zorder=5, label='Liftoff')
    ax.scatter([downrange[-1]], [altitude[-1]], c='red', s=150, marker='x', zorder=5, label='MECO')
    ax.set_xlabel('Downrange (km)')
    ax.set_ylabel('Altitude (km)')
    ax.set_title('Trajectory Overview (Color = Time)', fontweight='bold', style='italic')
    ax.legend(loc='upper left')
    ax.set_xlim(0, None)
    ax.set_ylim(0, None)
    plt.tight_layout()
    path = os.path.join(output_dir, 'trajectory.png')
    fig.savefig(path, bbox_inches='tight')
    saved_files.append(path)
    plt.close(fig)
    
    return saved_files


def main():
    """Run simulation and generate plots."""
    print("Running RLV Phase-I simulation...")
    final_state, log, reason = run_simulation(verbose=True)
    
    print("\nGenerating plots...")
    output_dir = "plots"
    saved = generate_all_plots(log, final_state, output_dir)
    
    print(f"\n✅ Generated {len(saved)} plots in '{output_dir}/':")
    for path in saved:
        print(f"   - {os.path.basename(path)}")


if __name__ == "__main__":
    main()
