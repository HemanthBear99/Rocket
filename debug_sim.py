
import numpy as np
from rlv_sim.main import run_simulation
from rlv_sim import constants as C

def analyze_crash():
    print("Running Debug Simulation...")
    final_state, log, reason = run_simulation(verbose=False)
    
    # Convert to arrays
    time = np.array(log.time)
    alt = np.array(log.altitude)
    pitch = np.array(log.pitch_angle)
    # v_x = np.array(log.velocity_x)
    
    # Find crash index
    print(f"Simulation ended at t={time[-1]:.2f}s, Reason: {reason}")
    print(f"Final Altitude: {alt[-1]:.2f} km")
    
    # Check max pitch
    max_pitch_idx = np.argmax(np.abs(pitch))
    print(f"Max Pitch (Guidance): {pitch[max_pitch_idx]:.2f} deg at t={time[max_pitch_idx]:.2f}s")
    
    # Check Attitude Error
    att_error = np.array(log.attitude_error)
    max_err_idx = np.argmax(np.abs(att_error))
    print(f"Max Attitude Error: {att_error[max_err_idx]:.2f} deg at t={time[max_err_idx]:.2f}s")
    
    if max_err_idx > 0 and att_error[max_err_idx] > 10.0:
        print("!!! LOSS OF ATTITUDE CONTROL (>10 deg error) !!!")
        
    # Check Throttling
    # We didn't log throttle explicitly in the list, but it's in guidance output?
    # log is a named tuple, check fields
    # It has 'torque_magnitude'.
    
    # Check Torque Saturation
    torques = np.array(log.torque_magnitude)
    sat_limit = C.MAX_TORQUE
    saturated = torques >= (sat_limit * 0.99)
    if np.any(saturated):
        sat_idx = np.where(saturated)[0][0]
        print(f"!!! TORQUE SATURATED at t={time[sat_idx]:.2f}s !!!")
        print(f"Saturation persisted for {np.sum(saturated)} steps")

if __name__ == "__main__":
    analyze_crash()
