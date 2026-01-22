"""Detailed check of what's happening during simulation."""
from rlv_sim import constants as C
from rlv_sim.state import create_initial_state
from rlv_sim.guidance import compute_guidance_output
from rlv_sim.control import compute_control_output
from rlv_sim.integrators import rk4_step
import numpy as np

state = create_initial_state()
print("Step | Time | Alt(km) | Error(deg) | Torque(N*m) | Phase")
print("-"*70)

max_error = 0
for i in range(5000):
    g = compute_guidance_output(state.r, state.v, state.t, state.m)
    c = compute_control_output(state.q, state.omega, g['thrust_direction'])
    
    if c['error_degrees'] > max_error:
        max_error = c['error_degrees']
    
    # Print every 500 steps
    if i % 500 == 0:
        alt_km = state.altitude / 1000
        print(f"{i:5d} | {state.t:5.1f}s | {alt_km:7.2f} | {c['error_degrees']:8.2f} | {np.linalg.norm(c['torque']):12.0f} | {g['phase']}")
    
    if state.m <= C.DRY_MASS:
        break
    state = rk4_step(state, c['torque'], C.DT, thrust_on=g['thrust_on'])

print("-"*70)
print(f"Max attitude error: {max_error:.2f} deg")
print(f"Kp = {C.KP_ATTITUDE:.2e}, Kd = {C.KD_ATTITUDE:.2e}")
print(f"VERDICT: {'PASS <2' if max_error < 2 else 'WARN 2-8' if max_error < 8 else 'FAIL >8'}")
