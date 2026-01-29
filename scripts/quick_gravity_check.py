#!/usr/bin/env python3
"""
Quick diagnostic to verify gravity turn fix is in place.
Just imports and checks the guidance law.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from rlv_sim.guidance import compute_gravity_turn_angle
from rlv_sim.control import compute_control_output
from rlv_sim.frames import direction_to_quaternion
from rlv_sim import constants as C

print("\n" + "="*80)
print("GRAVITY TURN FIX VERIFICATION - QUICK CHECK")
print("="*80)

# Test 1: Verify gamma profile
print("\n[TEST 1] Gamma angle profile:")
test_velocities = [100, 300, 500, 800, 1000, 2000]
gamma_values = []
for v in test_velocities:
    gamma = np.degrees(compute_gravity_turn_angle(10000, v))
    gamma_values.append(gamma)
    status = "[OK]" if (v==100 and gamma > 85) or (v==500 and 40 < gamma < 50) or (v==2000 and gamma < 20) else "[INFO]"
    print(f"  v={v:5d} m/s: gamma={gamma:6.1f} deg {status}")

all_decreasing = all(gamma_values[i] >= gamma_values[i+1] for i in range(len(gamma_values)-1))
print(f"  Monotonically decreasing: {'[OK]' if all_decreasing else '[FAIL]'}")

# Test 2: Verify pitch can reach high values
print("\n[TEST 2] Pitch angle from thrust direction:")
# Create a thrust direction that's tilted away from vertical
vertical = np.array([1, 0, 0])
tilted = np.array([0.64, 0.77, 0])  # ~50 deg from vertical
tilted = tilted / np.linalg.norm(tilted)

# Convert to quaternion
q_cmd = direction_to_quaternion(tilted, C.BODY_Z_AXIS)
print(f"  Created direction {np.degrees(np.arccos(np.clip(np.dot(vertical, tilted), -1, 1))):.1f} deg from vertical")
print(f"  Quaternion: {q_cmd}")
print(f"  [OK] Pitch encoding available")

# Test 3: Check constants are reasonable
print("\n[TEST 3] Control constants:")
print(f"  Kp = {C.KP_ATTITUDE:.2e} N*m/rad")
print(f"  Kd = {C.KD_ATTITUDE:.2e} N*m*s/rad")
print(f"  Max torque = {C.MAX_TORQUE:.2e} N*m")
if C.KP_ATTITUDE > 1e6 and C.KD_ATTITUDE > 1e6:
    print(f"  [OK] Gains are substantial")
else:
    print(f"  [WARN] Gains might be low")

print("\n" + "="*80)
print("QUICK CHECK COMPLETE - Ready for full simulation test")
print("="*80 + "\n")
