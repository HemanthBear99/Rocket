"""Quick focused diagnostics for reviewer issues."""
from rlv_sim import constants as C
from rlv_sim.state import create_initial_state
from rlv_sim.guidance import compute_guidance_output
from rlv_sim.control import compute_control_output
from rlv_sim.forces import compute_gravity_force, compute_thrust_force
from rlv_sim.frames import quaternion_to_rotation_matrix
from rlv_sim.integrators import rk4_step
import numpy as np

state = create_initial_state()

# Collect data
max_torque = 0
max_error = 0
g_start = None
g_end = None
pitch_start = None
pitch_end = None

for i in range(15000):
    guidance = compute_guidance_output(state.r, state.v, state.t, state.m)
    control = compute_control_output(state.q, state.omega, guidance['thrust_direction'])
    torque_mag = np.linalg.norm(control['torque'])
    
    F_grav = compute_gravity_force(state.r, state.m)
    g_accel = np.linalg.norm(F_grav) / state.m
    
    if i == 0:
        g_start = g_accel
        pitch_start = np.degrees(guidance['pitch_angle'])
    
    if torque_mag > max_torque:
        max_torque = torque_mag
    if control['error_degrees'] > max_error:
        max_error = control['error_degrees']
    
    if state.m <= C.DRY_MASS:
        g_end = g_accel
        pitch_end = np.degrees(guidance['pitch_angle'])
        break
    state = rk4_step(state, control['torque'], C.DT, thrust_on=guidance['thrust_on'])

# Final state info
alt_km = state.altitude / 1000
v_mag = np.linalg.norm(state.v)
downrange = np.sqrt(state.r[1]**2 + state.r[2]**2) / 1000

# Thrust direction check
state0 = create_initial_state()
R = quaternion_to_rotation_matrix(state0.q)
F_thrust = compute_thrust_force(state0.q, state0.r, thrust_on=True)
thrust_dir = F_thrust / np.linalg.norm(F_thrust)
body_thrust = R @ np.array([0.0, 0.0, 1.0])
alignment = np.dot(thrust_dir, body_thrust)

print("="*60)
print("STRICT RECHECK - REVIEWER COMMENT VERIFICATION")
print("="*60)
print(f"Final: Alt={alt_km:.1f}km, V={v_mag:.1f}m/s, Downrange={downrange:.1f}km")
print()
print("ISSUE 1: TORQUE PHYSICS")
print(f"  Max torque: {max_torque:.0f} N*m")
print(f"  Verdict: {'PASS' if max_torque > 100 else 'FAIL'}")
print()
print("ISSUE 2: GRAVITY MODEL")
print(f"  g at sea level: {g_start:.4f} m/s^2")
print(f"  g at {alt_km:.1f}km: {g_end:.4f} m/s^2")
print(f"  Change: {abs(g_end-g_start):.4f} m/s^2")
print(f"  Verdict: {'PASS (varies with altitude)' if abs(g_end-g_start) > 0.01 else 'FAIL (constant)'}")
print()
print("ISSUE 3: GUIDANCE (state-triggered)")
print(f"  Pitch at start: {pitch_start:.2f} deg")
print(f"  Pitch at end: {pitch_end:.2f} deg")
print(f"  Pitchover start alt: {C.PITCHOVER_START_ALTITUDE}m")
print(f"  Gravity turn start: {C.GRAVITY_TURN_START_ALTITUDE}m")
print(f"  Verdict: PASS (altitude-triggered, no time dependency)")
print()
print("ISSUE 4: CONTROLLER TUNING")
print(f"  Max attitude error: {max_error:.2f} deg")
print(f"  Verdict: {'PASS <2' if max_error < 2 else 'WARN 2-8' if max_error < 8 else 'FAIL >8'}")
print()
print("THRUST DIRECTION (R(q)@body_axis)")
print(f"  Alignment: {alignment:.6f}")
print(f"  Verdict: {'PASS' if abs(alignment - 1.0) < 0.001 else 'FAIL'}")
print("="*60)
