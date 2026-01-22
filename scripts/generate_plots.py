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
    omega_earth = np.array([0.0, 0.0, C.EARTH_ROTATION_RATE]) # Axis of rotation (Z? No, Earth rotates about Z in ECI? 
    # WAIT. ECI frame definition in constants.py: "At equator (r = R_EARTH along X), v is along Y".
    # Rotation is about Z axis [0, 0, 1].
    
    # Vectorized computation of relative velocity
    # v_rel = v - cross(omega, r)
    omega_vec = np.array([0.0, 0.0, C.EARTH_ROTATION_RATE])
    
    # We need to compute this for every timestep.
    # pos = [pos_x, pos_y, pos_z] (N, 3)
    pos_array = np.column_stack((pos_x, pos_y, pos_z))
    vel_array = np.column_stack((vel_x, vel_y, vel_z))
    
    cross_term = np.cross(omega_vec, pos_array) # (N, 3)
    v_rel_array = vel_array - cross_term
    v_rel_mag = np.linalg.norm(v_rel_array, axis=1)
    
    # 2. Flight Path Angle (Relative to Local Horizontal)
    # gamma = angle between v_rel and local horizontal
    # sin(gamma) = v_rel_radial / |v_rel|
    # v_rel_radial = dot(v_rel, r_hat)
    r_mag = np.linalg.norm(pos_array, axis=1)
    r_hat = pos_array / r_mag[:, np.newaxis]
    
    v_rel_radial = np.sum(v_rel_array * r_hat, axis=1)
    
    # Clip for safety
    sin_gamma = np.clip(v_rel_radial / np.maximum(v_rel_mag, 1.0), -1.0, 1.0)
    gamma_relative = np.degrees(np.arcsin(sin_gamma))
    
    # 3. Dynamic Pressure (Q)
    # q = 0.5 * rho * v_rel^2
    # Be expensive to call atmosphere model for every point, but necessary for plots.
    # Approximate based on exponential for plotting speed? 
    # Or call loop.
    q_dynamic = []
    
    # Imports for calculations
    from rlv_sim.forces import compute_atmosphere_properties
    
    for h, v_mag_rel in zip(altitude, v_rel_mag):
        # h is in km in the log arrays, but compute_atmosphere_properties needs m
        h_m = h * 1000.0
        _, _, rho, _ = compute_atmosphere_properties(h_m)
        q = 0.5 * rho * v_mag_rel**2
        q_dynamic.append(q)
    
    q_dynamic = np.array(q_dynamic)
    
    # Use the corrected Relative Gamma for plotting
    gamma = gamma_relative
    
    # Gravity turn start (when pitch starts changing significantly)
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
    # 02. Velocity Profile
    # =========================================================================
    fig, ax = plt.subplots()
    ax.fill_between(time, 0, velocity, alpha=0.3, color='red')
    ax.plot(time, velocity, 'r-', linewidth=2, label='Velocity')
    ax.scatter([time[liftoff_idx]], [velocity[liftoff_idx]], 
               c='green', s=100, marker='o', zorder=5, label=f'Liftoff ({velocity[liftoff_idx]:.0f} m/s)')
    ax.scatter([time[meco_idx]], [velocity[meco_idx]], 
               c='red', s=100, marker='x', zorder=5, label=f'MECO ({meco_vel:.0f} m/s)')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Velocity (m/s)')
    ax.set_title('Velocity Profile', fontweight='bold', style='italic')
    ax.legend(loc='lower right')
    ax.set_xlim(0, time[-1])
    ax.set_ylim(0, None)
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
    ax.scatter([time[liftoff_idx]], [mass_tonnes[liftoff_idx]], 
               c='green', s=100, marker='o', zorder=5, label=f'Liftoff ({mass_tonnes[liftoff_idx]:.0f} t)')
    ax.scatter([time[meco_idx]], [mass_tonnes[meco_idx]], 
               c='red', s=100, marker='x', zorder=5, label=f'MECO ({mass_tonnes[meco_idx]:.0f} t)')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Mass (tonnes)')
    ax.set_title('Mass Profile (Propellant Consumption)', fontweight='bold', style='italic')
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
    ax.plot(time, pitch_angle, color='purple', linewidth=2, label='Pitch Angle')
    ax.axvline(x=gravity_turn_time, color='gray', linestyle=':', linewidth=1.5,
               label=f'Gravity Turn Start (t={gravity_turn_time:.0f}s)')
    peak_pitch_idx = np.argmax(pitch_angle)
    ax.scatter([time[peak_pitch_idx]], [pitch_angle[peak_pitch_idx]], 
               c='orange', s=100, marker='D', zorder=5, 
               label=f'Peak Pitch ({pitch_angle[peak_pitch_idx]:.1f}°)')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Pitch Angle (degrees)')
    ax.set_title('Pitch Angle Evolution (Gravity Turn)', fontweight='bold', style='italic')
    ax.legend(loc='upper right')
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
    ax.fill_between(time, 0, attitude_error, alpha=0.3, color='cyan')
    ax.plot(time, attitude_error, 'c-', linewidth=1.5, label='Attitude Error')
    max_error_idx = np.argmax(attitude_error)
    ax.scatter([time[max_error_idx]], [attitude_error[max_error_idx]], 
               c='red', s=80, marker='x', zorder=5, 
               label=f'Max Error ({attitude_error[max_error_idx]:.2f}°)')
    ax.axhline(y=1.0, color='orange', linestyle='--', label='1° Threshold')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Attitude Error (degrees)')
    ax.set_title('Attitude Tracking Error', fontweight='bold', style='italic')
    ax.legend(loc='upper right')
    ax.set_xlim(0, time[-1])
    ax.set_ylim(0, None)
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
    ax.fill_between(time, 0, torque_mn, alpha=0.3, color='orange')
    ax.plot(time, torque_mn, color='orange', linewidth=1.5, label='Control Torque')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Torque Magnitude (MN·m)')
    ax.set_title('Control Torque', fontweight='bold', style='italic')
    ax.legend(loc='upper right')
    ax.set_xlim(0, time[-1])
    ax.set_ylim(0, None)
    plt.tight_layout()
    path = os.path.join(output_dir, '06_control_torque.png')
    fig.savefig(path, bbox_inches='tight')
    saved_files.append(path)
    plt.close(fig)
    
    # =========================================================================
    # 07. Trajectory Local (Downrange vs Altitude)
    # =========================================================================
    fig, ax = plt.subplots()
    # Compute downrange distance (horizontal distance from launch)
    downrange = np.sqrt((pos_x - pos_x[0])**2 + (pos_y - pos_y[0])**2) / 1000  # km
    ax.fill_between(downrange, 0, altitude, alpha=0.3, color='blue')
    ax.plot(downrange, altitude, 'b-', linewidth=2, label='Trajectory')
    ax.scatter([downrange[liftoff_idx]], [altitude[liftoff_idx]], 
               c='green', s=100, marker='o', zorder=5, label='Liftoff')
    ax.scatter([downrange[meco_idx]], [altitude[meco_idx]], 
               c='red', s=100, marker='x', zorder=5, 
               label=f'MECO (DR={downrange[meco_idx]:.1f} km)')
    ax.set_xlabel('Downrange Distance (km)')
    ax.set_ylabel('Altitude (km)')
    ax.set_title('Trajectory (Local Frame)', fontweight='bold', style='italic')
    ax.legend(loc='upper left')
    ax.set_xlim(0, None)
    ax.set_ylim(0, None)
    ax.set_aspect('equal', adjustable='datalim')
    plt.tight_layout()
    path = os.path.join(output_dir, '07_trajectory_local.png')
    fig.savefig(path, bbox_inches='tight')
    saved_files.append(path)
    plt.close(fig)
    
    # =========================================================================
    # 08. 3D Trajectory (Local)
    # =========================================================================
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    # Use local ENU (East, North, Up)
    east = (pos_y - pos_y[0]) / 1000  # km
    north = (pos_x - pos_x[0]) / 1000  # km  
    up = altitude
    
    # Color by time
    points = np.array([east, north, up]).T.reshape(-1, 1, 3)
    ax.plot(east, north, up, 'b-', linewidth=2, label='Trajectory')
    ax.scatter([east[0]], [north[0]], [up[0]], c='green', s=150, marker='o', label='Liftoff')
    ax.scatter([east[-1]], [north[-1]], [up[-1]], c='red', s=150, marker='x', label='MECO')
    ax.set_xlabel('East (km)')
    ax.set_ylabel('North (km)')
    ax.set_zlabel('Altitude (km)')
    ax.set_title('3D Trajectory (Local ENU Frame)', fontweight='bold', style='italic')
    ax.legend()
    plt.tight_layout()
    path = os.path.join(output_dir, '08_trajectory_3d_local.png')
    fig.savefig(path, bbox_inches='tight')
    saved_files.append(path)
    plt.close(fig)
    
    # =========================================================================
    # 09. Altitude vs Velocity
    # =========================================================================
    fig, ax = plt.subplots()
    ax.fill_between(velocity, 0, altitude, alpha=0.3, color='magenta')
    ax.plot(velocity, altitude, color='magenta', linewidth=2, label='Trajectory')
    ax.scatter([velocity[liftoff_idx]], [altitude[liftoff_idx]], 
               c='green', s=100, marker='o', zorder=5, label='Liftoff')
    ax.scatter([velocity[meco_idx]], [altitude[meco_idx]], 
               c='red', s=100, marker='x', zorder=5, label='MECO')
    ax.set_xlabel('Velocity (m/s)')
    ax.set_ylabel('Altitude (km)')
    ax.set_title('Altitude vs Velocity', fontweight='bold', style='italic')
    ax.legend(loc='lower right')
    ax.set_xlim(0, None)
    ax.set_ylim(0, None)
    plt.tight_layout()
    path = os.path.join(output_dir, '09_altitude_vs_velocity.png')
    fig.savefig(path, bbox_inches='tight')
    saved_files.append(path)
    plt.close(fig)
    
    # =========================================================================
    # 10. Thrust vs Gravity (Force Balance)
    # =========================================================================
    fig, ax = plt.subplots()
    # Approximate thrust and gravity forces
    g = C.G0  # m/s²
    thrust_force = np.ones_like(time) * C.THRUST_MAGNITUDE / 1e6  # MN (simplified)
    
    # Correct Inverse-Square Gravity Calculation for Plots
    # F_g = mu * m / r^2
    r_center = altitude * 1000.0 + C.R_EARTH
    gravity_force = (C.MU_EARTH * mass / (r_center**2)) / 1e6 # MN
    
    ax.plot(time, thrust_force, 'r-', linewidth=2, label='Thrust')
    ax.fill_between(time, 0, thrust_force, alpha=0.2, color='red')
    ax.plot(time, gravity_force, 'b-', linewidth=2, label='Weight (mg)')
    ax.fill_between(time, 0, gravity_force, alpha=0.2, color='blue')
    
    # TWR = 1 line
    ax.axhline(y=thrust_force[0], color='red', linestyle=':', alpha=0.5)
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Force (MN)')
    ax.set_title('Thrust vs Weight', fontweight='bold', style='italic')
    ax.legend(loc='upper right')
    ax.set_xlim(0, time[-1])
    ax.set_ylim(0, None)
    
    # Add TWR annotation
    twr_initial = thrust_force[0] / gravity_force[0]
    twr_final = thrust_force[-1] / gravity_force[-1]
    ax.annotate(f'TWR₀ = {twr_initial:.2f}', xy=(5, thrust_force[0]*0.9), fontsize=10)
    ax.annotate(f'TWR_MECO = {twr_final:.2f}', xy=(time[-1]*0.7, thrust_force[-1]*1.1), fontsize=10)
    
    plt.tight_layout()
    path = os.path.join(output_dir, '10_thrust_vs_gravity.png')
    fig.savefig(path, bbox_inches='tight')
    saved_files.append(path)
    plt.close(fig)
    
    # =========================================================================
    # 11. Flight Path Angle (Gamma) - RELATIVE
    # =========================================================================
    fig, ax = plt.subplots()
    ax.fill_between(time, 0, gamma, alpha=0.3, color='blue')
    ax.plot(time, gamma, 'b-', linewidth=2, label='Relative FPA γ')
    
    # Reference lines
    ax.axhline(y=90, color='green', linestyle='--', alpha=0.5, label='Vertical (90°)')
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='Horizontal (0°)')
    ax.axvline(x=gravity_turn_time, color='gray', linestyle=':', alpha=0.7)
    
    # Markers
    ax.scatter([time[liftoff_idx]], [gamma[liftoff_idx]], 
               c='green', s=100, marker='o', zorder=5, label=f'Liftoff (γ={gamma[liftoff_idx]:.1f}°)')
    gamma_min_idx = np.argmin(gamma) # For relative, it starts at 90 and drops.
    ax.scatter([time[meco_idx]], [gamma[meco_idx]], 
               c='red', s=100, marker='x', zorder=5, label=f'MECO (γ={gamma[meco_idx]:.1f}°)')
    
    # Physics explanation box
    textstr = 'Corrected Physics:\n• Plot shows Air-Relative Flight Path Angle\n• Starts at 90° (Vertical climb w.r.t Air)\n• Gravity turn reduces γ towards horizontal (0°)'
    props = dict(boxstyle='round', facecolor='lightyellow', alpha=0.9)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=8,
            verticalalignment='top', bbox=props)
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Relative Flight Path Angle γ (°)')
    ax.set_title('Relative Flight Path Angle\n(γ = angle of relative velocity from local horizontal)', 
                 fontweight='bold', style='italic')
    ax.legend(loc='lower left', fontsize=9)
    ax.set_xlim(0, time[-1])
    ax.set_ylim(-10, 100)
    plt.tight_layout()
    path = os.path.join(output_dir, '11_flight_path_angle.png')
    fig.savefig(path, bbox_inches='tight')
    saved_files.append(path)
    plt.close(fig)

    # =========================================================================
    # 12. Dynamic Pressure (Max-Q)
    # =========================================================================
    fig, ax = plt.subplots()
    q_kpa = q_dynamic / 1000.0
    limit_kpa = C.MAX_DYNAMIC_PRESSURE / 1000.0
    
    ax.fill_between(time, 0, q_kpa, alpha=0.3, color='purple')
    ax.plot(time, q_kpa, 'purple', linewidth=2, label='Dynamic Pressure (Q)')
    ax.axhline(y=limit_kpa, color='red', linestyle='--', linewidth=2, label=f'Max-Q Limit ({limit_kpa:.1f} kPa)')
    
    # Find Max Q
    max_q_idx = np.argmax(q_kpa)
    max_q_val = q_kpa[max_q_idx]
    max_q_time = time[max_q_idx]
    
    ax.scatter([max_q_time], [max_q_val], c='red', s=100, marker='D', zorder=5, 
               label=f'Peak Q ({max_q_val:.1f} kPa)')
               
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Dynamic Pressure (kPa)')
    ax.set_title('Dynamic Pressure Evolution', fontweight='bold', style='italic')
    ax.legend(loc='upper right')
    ax.set_xlim(0, time[-1])
    ax.set_ylim(0, None)
    plt.tight_layout()
    path = os.path.join(output_dir, '12_dynamic_pressure.png')
    fig.savefig(path, bbox_inches='tight')
    saved_files.append(path)
    plt.close(fig)
    ax.axvline(x=gravity_turn_time, color='gray', linestyle=':', alpha=0.7)
    
    # Markers
    ax.scatter([time[liftoff_idx]], [gamma[liftoff_idx]], 
               c='green', s=100, marker='o', zorder=5, label=f'Liftoff (γ={gamma[liftoff_idx]:.1f}°)')
    gamma_min_idx = np.argmin(gamma)
    ax.scatter([time[gamma_min_idx]], [gamma[gamma_min_idx]], 
               c='orange', s=100, marker='D', zorder=5, label=f'Min (γ={gamma[gamma_min_idx]:.1f}°)')
    ax.scatter([time[meco_idx]], [gamma[meco_idx]], 
               c='red', s=100, marker='x', zorder=5, label=f'MECO (γ={gamma[meco_idx]:.1f}°)')
    
    # Physics explanation box
    textstr = 'Physics Explanation:\n• At liftoff, velocity is horizontal (Earth rotation) → γ=90°\n• Thrust adds vertical velocity → γ decreases towards 0° (vertical)\n• Gravity turn bends trajectory → γ increases towards 90°'
    props = dict(boxstyle='round', facecolor='lightyellow', alpha=0.9)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=8,
            verticalalignment='top', bbox=props)
    
    # Annotation for gravity turn
    ax.annotate('Gravity Turn\nStart', xy=(gravity_turn_time, gamma[gravity_turn_idx]),
                xytext=(gravity_turn_time + 10, gamma[gravity_turn_idx] + 15),
                fontsize=9, style='italic', color='gray',
                arrowprops=dict(arrowstyle='->', color='gray', alpha=0.7))
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Flight Path Angle γ (°)')
    ax.set_title('Flight Path Angle Evolution\n(γ = angle of velocity from local vertical)', 
                 fontweight='bold', style='italic')
    ax.legend(loc='lower left', fontsize=9)
    ax.set_xlim(0, time[-1])
    ax.set_ylim(-10, 100)
    plt.tight_layout()
    path = os.path.join(output_dir, '11_flight_path_angle.png')
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
    axes[1,1].fill_between(time, 0, gamma, alpha=0.3, color='blue')
    axes[1,1].plot(time, gamma, 'b-', linewidth=2)
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
