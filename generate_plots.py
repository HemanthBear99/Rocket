"""
RLV Phase-I Ascent Simulation - Generate and Save All Plots

This script runs the simulation and generates comprehensive visualizations.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from rlv_sim import run_simulation
from rlv_sim import constants as C


def plot_altitude_velocity(log, save_dir):
    """Plot altitude and velocity profiles."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Altitude and Velocity Profiles', fontsize=14, fontweight='bold')
    
    # Altitude vs Time
    ax1.plot(log.time, log.altitude, 'b-', linewidth=2)
    ax1.fill_between(log.time, 0, log.altitude, alpha=0.3)
    ax1.set_xlabel('Time (s)', fontsize=12)
    ax1.set_ylabel('Altitude (km)', fontsize=12)
    ax1.set_title('Altitude Profile')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, max(log.time)])
    ax1.set_ylim([0, None])
    
    # Velocity vs Time
    ax2.plot(log.time, log.velocity, 'r-', linewidth=2)
    ax2.fill_between(log.time, 0, log.velocity, alpha=0.3, color='red')
    ax2.set_xlabel('Time (s)', fontsize=12)
    ax2.set_ylabel('Velocity (m/s)', fontsize=12)
    ax2.set_title('Velocity Profile')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, max(log.time)])
    ax2.set_ylim([0, None])
    
    plt.tight_layout()
    path = os.path.join(save_dir, 'altitude_velocity.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"Saved: {path}")
    plt.close()
    return path


def plot_mass_propellant(log, save_dir):
    """Plot mass and propellant consumption."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Mass and Propellant Profiles', fontsize=14, fontweight='bold')
    
    # Mass vs Time
    ax1.plot(log.time, log.mass, 'g-', linewidth=2)
    ax1.axhline(y=C.DRY_MASS, color='r', linestyle='--', label=f'Dry Mass ({C.DRY_MASS} kg)')
    ax1.fill_between(log.time, C.DRY_MASS, log.mass, alpha=0.3, color='green')
    ax1.set_xlabel('Time (s)', fontsize=12)
    ax1.set_ylabel('Mass (kg)', fontsize=12)
    ax1.set_title('Total Mass')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_xlim([0, max(log.time)])
    
    # Propellant fraction vs Time
    propellant_fraction = [(m - C.DRY_MASS) / C.PROPELLANT_MASS * 100 for m in log.mass]
    ax2.plot(log.time, propellant_fraction, 'm-', linewidth=2)
    ax2.fill_between(log.time, 0, propellant_fraction, alpha=0.3, color='magenta')
    ax2.set_xlabel('Time (s)', fontsize=12)
    ax2.set_ylabel('Propellant Remaining (%)', fontsize=12)
    ax2.set_title('Propellant Consumption')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, max(log.time)])
    ax2.set_ylim([0, 100])
    
    plt.tight_layout()
    path = os.path.join(save_dir, 'mass_propellant.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"Saved: {path}")
    plt.close()
    return path


def plot_guidance_control(log, save_dir):
    """Plot guidance and control parameters."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Guidance and Control', fontsize=14, fontweight='bold')
    
    # Pitch Angle vs Time
    ax = axes[0, 0]
    ax.plot(log.time, log.pitch_angle, 'b-', linewidth=2)
    ax.axhline(y=45, color='r', linestyle='--', alpha=0.5, label='Target (45°)')
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Pitch Angle (deg)', fontsize=12)
    ax.set_title('Guidance Pitch Angle')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_xlim([0, max(log.time)])
    
    # Attitude Error vs Time
    ax = axes[0, 1]
    ax.plot(log.time, log.attitude_error, 'c-', linewidth=1)
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Attitude Error (deg)', fontsize=12)
    ax.set_title('Attitude Tracking Error')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, max(log.time)])
    
    # Control Torque vs Time
    ax = axes[1, 0]
    ax.plot(log.time, log.torque_magnitude, 'orange', linewidth=1)
    ax.axhline(y=C.MAX_TORQUE, color='r', linestyle='--', alpha=0.5, label=f'Saturation ({C.MAX_TORQUE} Nm)')
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Torque Magnitude (Nm)', fontsize=12)
    ax.set_title('Control Torque')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_xlim([0, max(log.time)])
    
    # Quaternion Norm vs Time
    ax = axes[1, 1]
    ax.plot(log.time, log.quaternion_norm, 'k-', linewidth=1)
    ax.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='Unit norm')
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Quaternion Norm', fontsize=12)
    ax.set_title('Quaternion Norm (Validation)')
    ax.set_ylim([0.9999, 1.0001])
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_xlim([0, max(log.time)])
    
    plt.tight_layout()
    path = os.path.join(save_dir, 'guidance_control.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"Saved: {path}")
    plt.close()
    return path


def plot_trajectory_2d(log, save_dir):
    """Plot 2D trajectory projections."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('2D Trajectory Projections', fontsize=14, fontweight='bold')
    
    # Convert to km
    x = np.array(log.position_x) / 1000
    y = np.array(log.position_y) / 1000
    z = np.array(log.position_z) / 1000
    
    # X-Y projection
    ax = axes[0]
    ax.plot(x, y, 'b-', linewidth=1.5)
    ax.scatter(x[0], y[0], c='g', s=100, marker='o', zorder=5, label='Start')
    ax.scatter(x[-1], y[-1], c='r', s=100, marker='x', zorder=5, label='MECO')
    # Earth circle
    theta = np.linspace(0, 2*np.pi, 100)
    R = C.R_EARTH / 1000
    ax.plot(R*np.cos(theta), R*np.sin(theta), 'b--', alpha=0.3, label='Earth surface')
    ax.set_xlabel('X (km)', fontsize=12)
    ax.set_ylabel('Y (km)', fontsize=12)
    ax.set_title('X-Y Plane')
    ax.axis('equal')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # X-Z projection
    ax = axes[1]
    ax.plot(x, z, 'b-', linewidth=1.5)
    ax.scatter(x[0], z[0], c='g', s=100, marker='o', zorder=5, label='Start')
    ax.scatter(x[-1], z[-1], c='r', s=100, marker='x', zorder=5, label='MECO')
    ax.plot(R*np.cos(theta), R*np.sin(theta), 'b--', alpha=0.3)
    ax.set_xlabel('X (km)', fontsize=12)
    ax.set_ylabel('Z (km)', fontsize=12)
    ax.set_title('X-Z Plane')
    ax.axis('equal')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Y-Z projection
    ax = axes[2]
    ax.plot(y, z, 'b-', linewidth=1.5)
    ax.scatter(y[0], z[0], c='g', s=100, marker='o', zorder=5, label='Start')
    ax.scatter(y[-1], z[-1], c='r', s=100, marker='x', zorder=5, label='MECO')
    ax.plot(R*np.cos(theta), R*np.sin(theta), 'b--', alpha=0.3)
    ax.set_xlabel('Y (km)', fontsize=12)
    ax.set_ylabel('Z (km)', fontsize=12)
    ax.set_title('Y-Z Plane')
    ax.axis('equal')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    path = os.path.join(save_dir, 'trajectory_2d.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"Saved: {path}")
    plt.close()
    return path


def plot_trajectory_3d(log, save_dir):
    """Plot 3D trajectory with Earth."""
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Convert to km
    x = np.array(log.position_x) / 1000
    y = np.array(log.position_y) / 1000
    z = np.array(log.position_z) / 1000
    
    # Color trajectory by time
    colors = plt.cm.viridis(np.linspace(0, 1, len(x)))
    for i in range(len(x)-1):
        ax.plot(x[i:i+2], y[i:i+2], z[i:i+2], c=colors[i], linewidth=2)
    
    # Start and end markers
    ax.scatter(x[0], y[0], z[0], c='lime', s=200, marker='o', label='Liftoff', edgecolors='black')
    ax.scatter(x[-1], y[-1], z[-1], c='red', s=200, marker='X', label='MECO', edgecolors='black')
    
    # Plot Earth sphere
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    R = C.R_EARTH / 1000
    xe = R * np.outer(np.cos(u), np.sin(v))
    ye = R * np.outer(np.sin(u), np.sin(v))
    ze = R * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(xe, ye, ze, alpha=0.2, color='blue', edgecolor='none')
    
    ax.set_xlabel('X (km)', fontsize=12)
    ax.set_ylabel('Y (km)', fontsize=12)
    ax.set_zlabel('Z (km)', fontsize=12)
    ax.set_title('3D Trajectory (Color = Time)', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left')
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=0, vmax=max(log.time)))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.5, aspect=20, pad=0.1)
    cbar.set_label('Time (s)', fontsize=12)
    
    plt.tight_layout()
    path = os.path.join(save_dir, 'trajectory_3d.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"Saved: {path}")
    plt.close()
    return path


def plot_flight_summary(log, final_state, save_dir):
    """Create a summary dashboard plot."""
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle('RLV Phase-I Ascent Simulation - Flight Summary', fontsize=16, fontweight='bold')
    
    # Create grid
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Altitude
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(log.time, log.altitude, 'b-', linewidth=2)
    ax1.fill_between(log.time, 0, log.altitude, alpha=0.3)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Altitude (km)')
    ax1.set_title('Altitude')
    ax1.grid(True, alpha=0.3)
    
    # Velocity
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(log.time, log.velocity, 'r-', linewidth=2)
    ax2.fill_between(log.time, 0, log.velocity, alpha=0.3, color='red')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Velocity (m/s)')
    ax2.set_title('Velocity')
    ax2.grid(True, alpha=0.3)
    
    # Mass
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(log.time, log.mass, 'g-', linewidth=2)
    ax3.axhline(y=C.DRY_MASS, color='r', linestyle='--', alpha=0.5)
    ax3.fill_between(log.time, C.DRY_MASS, log.mass, alpha=0.3, color='green')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Mass (kg)')
    ax3.set_title('Mass')
    ax3.grid(True, alpha=0.3)
    
    # Pitch angle
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(log.time, log.pitch_angle, 'purple', linewidth=2)
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Pitch (deg)')
    ax4.set_title('Guidance Pitch')
    ax4.grid(True, alpha=0.3)
    
    # Attitude error
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.plot(log.time, log.attitude_error, 'c-', linewidth=1)
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('Error (deg)')
    ax5.set_title('Attitude Error')
    ax5.grid(True, alpha=0.3)
    
    # Torque
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.plot(log.time, log.torque_magnitude, 'orange', linewidth=1)
    ax6.set_xlabel('Time (s)')
    ax6.set_ylabel('Torque (Nm)')
    ax6.set_title('Control Torque')
    ax6.grid(True, alpha=0.3)
    
    # 3D trajectory (spanning bottom row)
    ax7 = fig.add_subplot(gs[2, :], projection='3d')
    x = np.array(log.position_x) / 1000
    y = np.array(log.position_y) / 1000
    z = np.array(log.position_z) / 1000
    ax7.plot(x, y, z, 'b-', linewidth=1.5)
    ax7.scatter(x[0], y[0], z[0], c='g', s=100, marker='o')
    ax7.scatter(x[-1], y[-1], z[-1], c='r', s=100, marker='x')
    
    # Earth
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 30)
    R = C.R_EARTH / 1000
    xe = R * np.outer(np.cos(u), np.sin(v))
    ye = R * np.outer(np.sin(u), np.sin(v))
    ze = R * np.outer(np.ones(np.size(u)), np.cos(v))
    ax7.plot_surface(xe, ye, ze, alpha=0.15, color='blue')
    
    ax7.set_xlabel('X (km)')
    ax7.set_ylabel('Y (km)')
    ax7.set_zlabel('Z (km)')
    ax7.set_title('3D Trajectory')
    
    plt.tight_layout()
    path = os.path.join(save_dir, 'flight_summary.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"Saved: {path}")
    plt.close()
    return path


def plot_altitude_vs_velocity(log, save_dir):
    """Plot altitude vs velocity (phase portrait style)."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Color by time
    scatter = ax.scatter(log.velocity, log.altitude, c=log.time, cmap='viridis', s=2)
    ax.plot(log.velocity, log.altitude, 'b-', alpha=0.3, linewidth=0.5)
    
    # Mark key points
    ax.scatter(log.velocity[0], log.altitude[0], c='lime', s=200, marker='o', 
               edgecolors='black', zorder=5, label='Liftoff')
    ax.scatter(log.velocity[-1], log.altitude[-1], c='red', s=200, marker='X', 
               edgecolors='black', zorder=5, label='MECO')
    
    ax.set_xlabel('Velocity (m/s)', fontsize=12)
    ax.set_ylabel('Altitude (km)', fontsize=12)
    ax.set_title('Altitude vs Velocity', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label('Time (s)', fontsize=12)
    
    plt.tight_layout()
    path = os.path.join(save_dir, 'altitude_vs_velocity.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"Saved: {path}")
    plt.close()
    return path


def generate_all_plots(log, final_state, save_dir):
    """Generate all standard plots from simulation log."""
    print("\n" + "="*70)
    print("Generating Standard Plots...")
    print("="*70)
    
    os.makedirs(save_dir, exist_ok=True)
    print(f"Saving to: {save_dir}\n")
    
    saved_plots = []
    
    saved_plots.append(plot_altitude_velocity(log, save_dir))
    saved_plots.append(plot_mass_propellant(log, save_dir))
    saved_plots.append(plot_guidance_control(log, save_dir))
    saved_plots.append(plot_trajectory_2d(log, save_dir))
    saved_plots.append(plot_trajectory_3d(log, save_dir))
    saved_plots.append(plot_flight_summary(log, final_state, save_dir))
    saved_plots.append(plot_altitude_vs_velocity(log, save_dir))
    
    print(f"\nStandard plots generated: {len(saved_plots)}")
    return saved_plots


if __name__ == "__main__":
    # Test execution
    print("Running standalone test...")
    # Add parent paths if running directly
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    final_state, log, reason = run_simulation(verbose=True)
    generate_all_plots(log, final_state, 'plots/standard')
