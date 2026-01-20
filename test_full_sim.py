"""Full simulation test."""
import sys
sys.path.insert(0, 'd:/T-client')

from rlv_sim import run_simulation

# Run the full simulation
final_state, log, reason = run_simulation(verbose=True)

print("\n" + "="*60)
print("FINAL SUMMARY")
print("="*60)
print(f"Termination: {reason}")
print(f"Time: {final_state.t:.2f} s")
print(f"Altitude: {final_state.altitude/1000:.2f} km")
print(f"Velocity: {final_state.speed:.2f} m/s")
print(f"Mass: {final_state.m:.2f} kg")
print(f"Data points logged: {len(log.time)}")
