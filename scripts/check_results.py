
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path.cwd()))

from rlv_sim.main import run_simulation
from rlv_sim import constants as C
from rlv_sim.frames import rotate_vector_by_quaternion
from rlv_sim.guidance import compute_local_vertical

def check():
    print("Running sim...")
    state, log, reason = run_simulation(verbose=False)
    
    # Extract data
    time = np.array(log.time)
    pos_x = np.array(log.position_x)
    pos_y = np.array(log.position_y)
    pos_z = np.array(log.position_z)
    vel_x = np.array(log.velocity_x)
    vel_y = np.array(log.velocity_y)
    vel_z = np.array(log.velocity_z)
    pitch_cmd = np.array(log.pitch_angle)
    actual_pitch = np.array(log.actual_pitch_angle)
    
    # Calculate derived metrics
    # Relative Velocity
    omega_vec = np.array([0.0, 0.0, C.EARTH_ROTATION_RATE])
    pos_array = np.column_stack((pos_x, pos_y, pos_z))
    vel_array = np.column_stack((vel_x, vel_y, vel_z))
    cross_term = np.cross(omega_vec, pos_array) 
    v_rel_array = vel_array - cross_term
    v_rel_mag = np.linalg.norm(v_rel_array, axis=1)
    
    # Local Vertical/Horizontal components
    r_mag = np.linalg.norm(pos_array, axis=1)
    r_hat = pos_array / r_mag[:, np.newaxis]
    v_rel_radial = np.sum(v_rel_array * r_hat, axis=1)
    
    v_horizontal = np.sqrt(v_rel_mag**2 - v_rel_radial**2)
    v_vertical = v_rel_radial
    
    # Velocity Tilt (from vertical)
    velocity_tilt = np.degrees(np.arctan2(v_horizontal, np.maximum(v_vertical, 0.1)))
    
    # Flight Path Angle (from horizontal)
    gamma_rel = np.degrees(np.arctan2(v_vertical, v_horizontal))
    
    # Downrange (Calculated as Slant Range in Plotting Script)
    downrange_slant = np.sqrt((pos_x - pos_x[0])**2 + (pos_y - pos_y[0])**2) / 1000.0
    
    # Downrange (Projected on Surface)
    # Calculate angle between current r and initial r
    r_norms = np.linalg.norm(pos_array, axis=1)
    # Dot product of normalized vectors
    # r0 is along X axis (1, 0, 0)
    # dot = x / |r|
    cos_theta = pos_x / r_norms
    theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    true_downrange = C.R_EARTH * theta / 1000.0 # km
    
    # MECO values
    idx = -1
    print("\n--- MECO RESULTS ---")
    print(f"Time: {time[idx]:.1f} s")
    print(f"Altitude: {log.altitude[idx]:.1f} km")
    print(f"Slant Range (Total Displacement): {downrange_slant[idx]:.1f} km")
    print(f"True Downrange (Ground Track):    {true_downrange[idx]:.1f} km")
    print(f"Commanded Pitch (from Vertical): {pitch_cmd[idx]:.2f} deg")
    print(f"Actual Pitch (from Vertical):    {actual_pitch[idx]:.2f} deg")
    print(f"Velocity Tilt (from Vertical):   {velocity_tilt[idx]:.2f} deg")
    print(f"Flight Path Angle (Gamma):       {gamma_rel[idx]:.2f} deg")
    # Assuming gamma_check_deg is defined elsewhere or will be added by the user
    # print(f"Gamma (Check): {gamma_check_deg[-1]:.1f}°")
    print(f"Gamma + Tilt (should be 90):     {gamma_rel[idx] + velocity_tilt[idx]:.2f} deg")
    
    print("\n--- RAW VELOCITY DATA ---")
    print(f"Vel X (Radial/Vertical): {vel_x[idx]:.1f} m/s")
    print(f"Vel Y (Tangential/East): {vel_y[idx]:.1f} m/s")
    print(f"Vel Z (North):           {vel_z[idx]:.1f} m/s")
    
    # Expected distance from const velocity
    exp_dist = 464.6 * time[idx] / 1000.0
    print(f"Expected Dist if V_tan constant: {exp_dist:.1f} km")
    
    print("\n--- CONSISTENCY CHECK ---")
    
    # check 1: Pitch vs Velocity Tilt
    # In gravity turn, they should be close, with Pitch > Tilt (slightly) or Pitch < Tilt depending on formulation
    print(f"Difference (Pitch - Tilt): {actual_pitch[idx] - velocity_tilt[idx]:.2f} deg")
    
    if abs(actual_pitch[idx] - velocity_tilt[idx]) > 10.0:
        print("FAIL: Massive disagreement between Attitude and Velocity Vector.")
        print("      This confirms the User's suspicion of physics/frame inconsistency.")
    else:
        print("PASS: Attitude follows Velocity Vector.")
        
    # check 2: Gamma vs Downrange
    # If Gamma > 85 deg (vertical) but Downrange (SLANT) > 100 km, that's consistent with altitude gain
    # But True Downrange should be small
    if true_downrange[idx] > 50.0 and gamma_rel[idx] > 80.0:
         print("FAIL: Vertical Flight but High Ground Track. Inconsistent.")
    else:
         print("PASS: Ground Track consistent with Vertical Flight.")


if __name__ == "__main__":
    check()
