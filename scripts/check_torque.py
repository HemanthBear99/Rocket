"""Check torque and attitude error."""
from rlv_sim import constants as C
from rlv_sim.state import create_initial_state
from rlv_sim.guidance import compute_guidance_output
from rlv_sim.control import compute_control_output
from rlv_sim.integrators import rk4_step
import numpy as np

state = create_initial_state()
max_torque = 0
max_error = 0
torque_samples = []
error_samples = []

for i in range(10000):
    g = compute_guidance_output(state.r, state.v, state.t, state.m)
    c = compute_control_output(state.q, state.omega, g['thrust_direction'])
    t_mag = np.linalg.norm(c['torque'])
    
    if t_mag > max_torque:
        max_torque = t_mag
    if c['error_degrees'] > max_error:
        max_error = c['error_degrees']
    
    if i % 500 == 0:
        torque_samples.append(t_mag)
        error_samples.append(c['error_degrees'])
    
    if state.m <= C.DRY_MASS:
        break
    state = rk4_step(state, c['torque'], C.DT, thrust_on=g['thrust_on'])

print("ISSUE 1: TORQUE PHYSICS")
print(f"Max torque: {max_torque:.0f} N*m")
print(f"Sample torques: {[f'{t:.0f}' for t in torque_samples[:10]]}")
print(f"VERDICT: {'PASS (non-zero)' if max_torque > 10 else 'FAIL (zero)'}")
print()
print("ISSUE 4: ATTITUDE ERROR")
print(f"Max error: {max_error:.2f} deg")
print(f"Sample errors: {[f'{e:.2f}' for e in error_samples[:10]]}")
print(f"VERDICT: {'PASS <2' if max_error < 2 else 'WARN 2-8' if max_error < 8 else 'FAIL >8'}")
