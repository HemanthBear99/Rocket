#!/usr/bin/env python3
"""
Diagnostic script to analyze gravity turn physics.

Checks:
1. Pitch command vs velocity tilt alignment
2. Gamma definition (from horizontal vs vertical)
3. Thrust direction coupling to attitude
4. Frame transformations (ECI vs local)
"""

import numpy as np
import matplotlib.pyplot as plt
from rlv_sim.state import create_initial_state
from rlv_sim.guidance import compute_guidance_output
from rlv_sim.control import compute_control_output
from rlv_sim.integrators import integrate
from rlv_sim.utils import compute_relative_velocity
from rlv_sim import constants as C

def run_diagnostic():
    """Run gravity turn diagnostic simulation."""
    
    # Initialize state
    state = create_initial_state()
    r, v, q, omega, m = state
    
    # Storage for diagnostics
    diagnostics = {
        't': [],
        'altitude': [],
        'v_rel_mag': [],
        'pitch_cmd': [],
        'gamma_cmd': [],
        'velocity_tilt': [],  # atan2(v_horiz / v_vert)
        'pitch_from_pos': [],  # Downrange position tilt
        'thrust_dir_z': [],     # Thrust vertical component
        'v_x': [],  # East velocity
        'v_y': [],  # North velocity
        'v_z': [],  # Vertical velocity
        'pos_x': [], # East position
        'pos_y': [], # North position
    }
    
    dt = 0.01
    
    print("Collecting 150 seconds of diagnostic data...")
    print("=" * 100)
    print("T(s)    Alt(km) Vrel(m/s) Pitch(°) Gamma(°) VelTilt(°) ThrustZ  Vx(m/s)  Vy(m/s)  Vz(m/s) DownrageGeometry(°)")
    print("=" * 100)
    
    for step_num in range(15000):  # 150 seconds
        t = step_num * dt
        
        # Compute guidance
        guidance = compute_guidance_output(r, v, q, omega, m, t)
        thrust_dir = guidance['thrust_direction']
        pitch_cmd = guidance['pitch_angle']
        gamma_cmd = guidance['gamma_angle']
        throttle = guidance['throttle']
        
        # Compute current state metrics
        altitude = float(np.linalg.norm(r) - C.R_EARTH)
        v_rel = compute_relative_velocity(r, v)
        v_rel_mag = float(np.linalg.norm(v_rel))
        
        # Local frame: r_hat = radial, horizontal perpendicular to r
        r_norm = float(np.linalg.norm(r))
        r_hat = r / r_norm
        
        # Horizontal velocity component (perpendicular to radial)
        v_horiz = v_rel - np.dot(v_rel, r_hat) * r_hat
        v_horiz_mag = float(np.linalg.norm(v_horiz))
        # Vertical velocity component (along radial)
        v_vert = float(np.dot(v_rel, r_hat))
        
        # Flight path angle from ACTUAL velocity
        if v_rel_mag > 1.0:
            velocity_tilt = np.arctan2(v_horiz_mag, v_vert)
        else:
            velocity_tilt = np.pi / 2
        
        # Downrange geometry: atan2(downrange / altitude)
        # Position: r = [x_east, y_north, z_vertical]
        x_east = float(r[0])
        y_north = float(r[1])
        z_vertical = float(r[2])
        
        # Distance from launch point (launch is at Earth surface)
        launch_r = C.R_EARTH
        current_r_mag = float(np.linalg.norm(r))
        
        # Horizontal position displacement (on Earth surface)
        # Using simple spherical: angle = displacement / R_earth
        horiz_displacement = current_r_mag * np.arctan2(np.sqrt(x_east**2 + y_north**2), z_vertical)
        
        # Downrange tilt = atan2(horizontal_distance / altitude)
        if altitude > 100:
            pitch_from_geometry = np.arctan2(horiz_displacement, altitude)
        else:
            pitch_from_geometry = 0.0
        
        # Store diagnostics
        diagnostics['t'].append(t)
        diagnostics['altitude'].append(altitude / 1000)  # km
        diagnostics['v_rel_mag'].append(v_rel_mag)
        diagnostics['pitch_cmd'].append(np.degrees(pitch_cmd))
        diagnostics['gamma_cmd'].append(np.degrees(gamma_cmd))
        diagnostics['velocity_tilt'].append(np.degrees(velocity_tilt))
        diagnostics['pitch_from_pos'].append(np.degrees(pitch_from_geometry))
        diagnostics['thrust_dir_z'].append(thrust_dir[2])
        diagnostics['v_x'].append(float(v_rel[0]))
        diagnostics['v_y'].append(float(v_rel[1]))
        diagnostics['v_z'].append(float(v_rel[2]))
        diagnostics['pos_x'].append(x_east / 1000)
        diagnostics['pos_y'].append(y_north / 1000)
        
        # Print progress
        if step_num % 500 == 0 and step_num > 0:
            print(f"{t:6.2f}  {altitude/1000:6.2f}  {v_rel_mag:9.1f} "
                  f"{np.degrees(pitch_cmd):7.2f}  {np.degrees(gamma_cmd):7.2f}  "
                  f"{np.degrees(velocity_tilt):8.2f}  {thrust_dir[2]:7.4f}  "
                  f"{v_rel[0]:8.1f}  {v_rel[1]:8.1f}  {v_rel[2]:8.1f}  "
                  f"{np.degrees(pitch_from_geometry):8.2f}")
        
        # Compute control
        control = compute_control_output(r, v, q, omega, guidance)
        
        # Integrate one step
        state = integrate(state, guidance, control, throttle, dt)
        r, v, q, omega, m = state
    
    print("=" * 100)
    print("\nDIAGNOSTIC ANALYSIS:")
    print("-" * 100)
    
    # Check 1: Pitch command vs velocity tilt
    pitch_diff = np.array(diagnostics['pitch_cmd']) - np.array(diagnostics['velocity_tilt'])
    pitch_diff_mean = np.nanmean(np.abs(pitch_diff[~np.isnan(pitch_diff)]))
    print(f"\n✓ Pitch Command vs Velocity Tilt:")
    print(f"  Mean difference: {pitch_diff_mean:.2f}°")
    print(f"  Pitch range: {min(diagnostics['pitch_cmd']):.2f}° → {max(diagnostics['pitch_cmd']):.2f}°")
    print(f"  VelTilt range: {min(diagnostics['velocity_tilt']):.2f}° → {max(diagnostics['velocity_tilt']):.2f}°")
    if pitch_diff_mean > 5.0:
        print(f"  ⚠️  MISMATCH: Pitch command not aligned with velocity tilt!")
    else:
        print(f"  ✓ Aligned")
    
    # Check 2: Gamma definition
    gamma_diff = np.array(diagnostics['gamma_cmd']) - np.array(diagnostics['velocity_tilt'])
    gamma_diff_mean = np.nanmean(np.abs(gamma_diff[~np.isnan(gamma_diff)]))
    print(f"\n✓ Gamma Command vs Velocity Tilt:")
    print(f"  Mean difference: {gamma_diff_mean:.2f}°")
    print(f"  Gamma range: {min(diagnostics['gamma_cmd']):.2f}° → {max(diagnostics['gamma_cmd']):.2f}°")
    if gamma_diff_mean > 10.0:
        print(f"  ⚠️  MISMATCH: Gamma not controlling velocity tilt!")
    else:
        print(f"  ✓ Aligned")
    
    # Check 3: Pitch from position geometry
    print(f"\n✓ Position Geometry:")
    print(f"  Final altitude: {diagnostics['altitude'][-1]:.2f} km")
    print(f"  Final East displacement: {diagnostics['pos_x'][-1]:.2f} km")
    print(f"  Final North displacement: {diagnostics['pos_y'][-1]:.2f} km")
    print(f"  Implied pitch from geometry: {diagnostics['pitch_from_pos'][-1]:.2f}°")
    print(f"  Pitch command at end: {diagnostics['pitch_cmd'][-1]:.2f}°")
    if abs(diagnostics['pitch_cmd'][-1] - diagnostics['pitch_from_pos'][-1]) > 10.0:
        print(f"  ⚠️  MISMATCH: Pitch command doesn't create the observed trajectory!")
    else:
        print(f"  ✓ Consistent")
    
    # Check 4: Velocity components
    print(f"\n✓ Velocity Components at MECO:")
    print(f"  Vx (East): {diagnostics['v_x'][-1]:.1f} m/s")
    print(f"  Vy (North): {diagnostics['v_y'][-1]:.1f} m/s")
    print(f"  Vz (Vertical): {diagnostics['v_z'][-1]:.1f} m/s")
    v_final = np.sqrt(diagnostics['v_x'][-1]**2 + diagnostics['v_y'][-1]**2 + diagnostics['v_z'][-1]**2)
    print(f"  Total Vrel: {v_final:.1f} m/s")
    
    # Plot diagnostics
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    
    # Plot 1: Pitch command vs velocity tilt
    ax = axes[0, 0]
    ax.plot(diagnostics['t'], diagnostics['pitch_cmd'], 'b-', linewidth=2, label='Pitch Command')
    ax.plot(diagnostics['t'], diagnostics['velocity_tilt'], 'r--', linewidth=2, label='Velocity Tilt')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Angle (°)')
    ax.set_title('Pitch Command vs Velocity Tilt')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Plot 2: Gamma command vs velocity tilt
    ax = axes[0, 1]
    ax.plot(diagnostics['t'], diagnostics['gamma_cmd'], 'g-', linewidth=2, label='Gamma Command')
    ax.plot(diagnostics['t'], diagnostics['velocity_tilt'], 'r--', linewidth=2, label='Velocity Tilt')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Angle (°)')
    ax.set_title('Gamma Command vs Velocity Tilt')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Plot 3: Thrust vector vertical component
    ax = axes[1, 0]
    ax.plot(diagnostics['t'], diagnostics['thrust_dir_z'], 'k-', linewidth=2)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Thrust Z Component')
    ax.set_title('Thrust Direction Z Component (should decrease from 1 to ~0.2)')
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Velocity components
    ax = axes[1, 1]
    ax.plot(diagnostics['t'], diagnostics['v_z'], 'b-', linewidth=2, label='Vz (Vertical)')
    ax.plot(diagnostics['t'], diagnostics['v_x'], 'r-', linewidth=2, label='Vx (East)')
    ax.plot(diagnostics['t'], diagnostics['v_y'], 'g-', linewidth=2, label='Vy (North)')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Velocity (m/s)')
    ax.set_title('Velocity Components')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Plot 5: Altitude
    ax = axes[2, 0]
    ax.plot(diagnostics['t'], diagnostics['altitude'], 'b-', linewidth=2)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Altitude (km)')
    ax.set_title('Altitude Profile')
    ax.grid(True, alpha=0.3)
    
    # Plot 6: Position downrange
    ax = axes[2, 1]
    downrange = np.sqrt(np.array(diagnostics['pos_x'])**2 + np.array(diagnostics['pos_y'])**2)
    ax.plot(diagnostics['t'], downrange, 'b-', linewidth=2, label='Downrange')
    ax.plot(diagnostics['t'], diagnostics['altitude'], 'r--', linewidth=2, label='Altitude')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Distance (km)')
    ax.set_title('Downrange vs Altitude')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('plots/gravity_turn_diagnostic.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved diagnostic plot to plots/gravity_turn_diagnostic.png")

if __name__ == '__main__':
    run_diagnostic()
