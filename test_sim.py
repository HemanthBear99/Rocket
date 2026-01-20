"""Quick test of the RLV simulation."""
import sys
sys.path.insert(0, 'd:/T-client')

from rlv_sim.state import create_initial_state
from rlv_sim.main import simulation_step
import rlv_sim.constants as C

# Initial state
state = create_initial_state()
print(f"Initial: {state}")

# Single step
state, g, c = simulation_step(state, C.DT)
print(f"After 1 step: {state}")
print(f"Guidance phase: {g['phase']}")
print(f"Torque: {c['torque_magnitude']:.2f} Nm")

# Run 1000 steps
for i in range(999):
    state, g, c = simulation_step(state, C.DT)

print(f"After 1000 steps (t={state.t:.2f}s): alt={state.altitude/1000:.2f}km, v={state.speed:.1f}m/s")
print("Test passed!")
