"""
RLV Stage 2 Engine Parameter Computation - V2
==============================================
Corrected analysis: The given MECO at 109km / 2726 m/s with gamma~10deg
gives apogee only ~122 km. For 337 km apogee, need gamma~0 (nearly horizontal)
or different conditions. Let us handle BOTH scenarios and also size the
thrust properly.
"""
import math

print("=" * 80)
print("RLV S2 PARAMETER COMPUTATION - V2 (CORRECTED)")
print("=" * 80)

MU = 3.986004418e14
R_E = 6.371e6
G0 = 9.80665

# ===========================================================================
# PART 1: FIND MECO CONDITIONS THAT GIVE 337 km APOGEE
# ===========================================================================
print("\n" + "=" * 80)
print("PART 1: MECO CONDITIONS FOR 337 km APOGEE")
print("=" * 80)

alt_meco = 109e3
r_meco = R_E + alt_meco
v_meco = 2726.0
alt_apogee_target = 337e3
r_apogee_target = R_E + alt_apogee_target

# For an orbit with perigee at r_meco and apogee at r_apogee_target:
# a = (r_peri + r_apo) / 2
a_target = (r_meco + r_apogee_target) / 2.0
print(f"  Target orbit: {alt_meco/1e3:.0f} km x {alt_apogee_target/1e3:.0f} km")
print(f"  Semi-major axis = ({r_meco/1e3:.1f} + {r_apogee_target/1e3:.1f}) / 2 = {a_target/1e3:.1f} km")

# Velocity at perigee (r_meco) in this orbit from vis-viva:
# v^2 = MU * (2/r - 1/a)
v_at_perigee = math.sqrt(MU * (2.0 / r_meco - 1.0 / a_target))
print(f"  v at perigee (109km) = {v_at_perigee:.2f} m/s")
print(f"  Current MECO v       = {v_meco:.2f} m/s")
print(f"  Difference           = {v_at_perigee - v_meco:.2f} m/s")

# This tells us: to get a 109x337 km orbit, you need v=7582 m/s at 109 km
# (purely tangential, gamma=0). But we only have 2726 m/s.
# This means 2726 m/s at 109 km is SUBORBITAL. The "337 km apogee"
# claim must come from a different interpretation.

# Let's find what apogee 2726 m/s at 109 km actually gives for various gamma:
print("\n  ACTUAL apogee for v=2726 m/s at 109 km:")
print(f"  {'gamma(deg)':>10} {'a(km)':>10} {'e':>10} {'apo_alt(km)':>12} {'v_apo(m/s)':>12}")
E_spec = 0.5 * v_meco**2 - MU / r_meco
a_spec = -MU / (2 * E_spec)

for gamma_deg in range(0, 91, 5):
    gr = math.radians(gamma_deg)
    v_tang = v_meco * math.cos(gr)
    h = r_meco * v_tang

    val = h**2 / (MU * a_spec)
    if val <= 1.0 and val >= 0.0:
        e = math.sqrt(1 - val)
        r_apo = a_spec * (1 + e)
        alt_apo = r_apo - R_E
        v_apo = h / r_apo if r_apo > 0 else 0
        if alt_apo > -R_E:
            print(f"  {gamma_deg:10d} {a_spec/1e3:10.1f} {e:10.6f} {alt_apo/1e3:12.1f} {v_apo:12.1f}")

# The orbit is deeply suborbital - apogee only ~109-160 km depending on gamma.
# The "337 km" must be from the ballistic coast trajectory ABOVE atmosphere.

# Let's recalculate assuming the sim actually provides apogee of 337 km
# This would require either higher velocity or the number is from
# ballistic propagation. Let's just ASSUME the user's stated conditions
# and compute S2 from there.

print("\n" + "=" * 80)
print("PART 2: S2 SIZING WITH USER-STATED APOGEE = 337 km")
print("=" * 80)

# If apogee is 337 km, the velocity at apogee in a suborbital arc:
# We need to find what orbit has perigee near surface (or MECO point)
# and apogee at 337 km.

# Let's compute v_apogee assuming energy conservation from MECO:
# At MECO: E = 0.5*v^2 - MU/r
# At apogee: E = 0.5*v_apo^2 - MU/r_apo
# So: v_apo = sqrt(2*(E + MU/r_apo))
r_apo_337 = R_E + 337e3
v_apo_337 = math.sqrt(max(0, 2 * (E_spec + MU / r_apo_337)))
print(f"\n  Using energy conservation from MECO:")
print(f"  E_specific at MECO  = {E_spec/1e6:.3f} MJ/kg")
print(f"  r_apogee (337 km)   = {r_apo_337/1e3:.1f} km")
print(f"  v at 337 km         = sqrt(2*({E_spec:.0f} + {MU:.3e}/{r_apo_337:.0f}))")
print(f"  v at 337 km         = {v_apo_337:.2f} m/s")

# BUT: if it truly coasts to 337 km, angular momentum must also be conserved.
# At apogee, velocity is purely tangential: h = r_apo * v_apo
# h at MECO: h = r_meco * v_meco * cos(gamma)
# These must match. Let's see what gamma gives apogee=337km:
print("\n  Finding gamma that gives 337 km apogee:")
h_needed = r_apo_337 * v_apo_337
v_tang_needed = h_needed / r_meco
gamma_needed = math.acos(min(1.0, v_tang_needed / v_meco))
print(f"  h needed            = {h_needed:.0f} m^2/s")
print(f"  v_tangential needed = {v_tang_needed:.1f} m/s")
print(f"  gamma needed        = {math.degrees(gamma_needed):.2f} deg")

# Check: this is a self-consistent calculation?
# v_at_337 from energy: sqrt(2(E + MU/r_337))
# But E includes radial component. At apogee, all velocity is tangential.
# So v_apo = h/r_apo, and h = r_meco * v_meco * cos(gamma)
# Energy: 0.5*v_meco^2 - MU/r_meco = 0.5*v_apo^2 - MU/r_apo
# => 0.5*(r_meco*v_meco*cos(gamma)/r_apo)^2 = 0.5*v_meco^2 - MU/r_meco + MU/r_apo
# This is implicit in gamma. Let's solve it numerically:

print("\n  Numerical solution for gamma giving apogee = 337 km:")
from scipy.optimize import brentq

def apogee_error(gamma_rad):
    """Error function: actual apogee altitude - target (337 km)."""
    v_t = v_meco * math.cos(gamma_rad)
    h = r_meco * v_t
    E = 0.5 * v_meco**2 - MU / r_meco
    a = -MU / (2 * E)
    val = h**2 / (MU * a)
    if val > 1.0 or val < 0.0:
        return -337e3  # no valid orbit
    e = math.sqrt(1 - val)
    r_apo = a * (1 + e)
    return (r_apo - R_E) - 337e3

# The orbit IS suborbital (E<0, but perigee below Earth surface).
# Apogee still exists as the highest point of the ballistic arc.
# With gamma=0 (horizontal), apogee=109 km (current altitude).
# With gamma>0, apogee rises.
# We need to find gamma such that apogee=337 km.

# Search from 0 to 89 degrees
best_gamma = None
best_err = float('inf')
for g_test in range(0, 90):
    err = apogee_error(math.radians(g_test))
    if abs(err) < abs(best_err):
        best_err = err
        best_gamma = g_test

print(f"  Coarse search: best gamma ~ {best_gamma} deg (error = {best_err/1e3:.1f} km)")

# Fine search around best
try:
    gamma_lo = math.radians(max(0, best_gamma - 2))
    gamma_hi = math.radians(min(89, best_gamma + 2))
    if apogee_error(gamma_lo) * apogee_error(gamma_hi) < 0:
        gamma_solution = brentq(apogee_error, gamma_lo, gamma_hi)
        print(f"  Exact solution: gamma = {math.degrees(gamma_solution):.4f} deg")

        # Compute orbit parameters at this gamma
        v_t = v_meco * math.cos(gamma_solution)
        h = r_meco * v_t
        E = 0.5 * v_meco**2 - MU / r_meco
        a = -MU / (2 * E)
        e = math.sqrt(1 - h**2 / (MU * a))
        r_apo = a * (1 + e)
        r_peri = a * (1 - e)
        v_apo = h / r_apo

        print(f"  Orbit: a={a/1e3:.1f}km, e={e:.6f}")
        print(f"  Perigee alt = {(r_peri-R_E)/1e3:.1f} km")
        print(f"  Apogee alt  = {(r_apo-R_E)/1e3:.1f} km")
        print(f"  v_apogee    = {v_apo:.2f} m/s")

        v_apogee_337 = v_apo
    else:
        print(f"  No bracket found (errors: {apogee_error(gamma_lo)/1e3:.1f}, {apogee_error(gamma_hi)/1e3:.1f} km)")
        # Use the energy-based estimate
        v_apogee_337 = v_apo_337
except Exception as ex:
    print(f"  Solver error: {ex}")
    v_apogee_337 = v_apo_337

# Also try: what if MECO velocity is the ECI velocity including Earth rotation?
# At equator, Earth surface velocity ~ 465 m/s
# Inertial velocity = ground-relative + Earth rotation contribution
v_earth_rotation = 7.2921159e-5 * R_E  # ~465 m/s
print(f"\n  Earth rotation velocity at surface = {v_earth_rotation:.1f} m/s")
print(f"  If v_meco=2726 is ground-relative, inertial = {v_meco + v_earth_rotation:.0f} m/s")

# ===========================================================================
# PART 3: CORRECTED S2 DELTA-V AND SIZING
# ===========================================================================
print("\n" + "=" * 80)
print("PART 3: CORRECTED S2 DELTA-V AND SIZING")
print("=" * 80)

# Use the user-stated v_apogee ~ 1800 m/s
v_apogee_user = 1800.0
v_circ_400 = math.sqrt(MU / (R_E + 400e3))

print(f"\n  User-stated v_apogee = {v_apogee_user:.0f} m/s")
print(f"  v_circular(400km)   = {v_circ_400:.2f} m/s")

# S2 must provide: v_circular - v_apogee + losses
# But the Hohmann approach is more accurate if apogee != 400 km

# Scenario 1: Apogee at 337 km, need to raise to 400 km and circularize
print("\n  --- Scenario 1: Apogee=337km, raise to 400km circular ---")
r_337 = R_E + 337e3
r_400 = R_E + 400e3

# Burn 1 at 337 km: enter transfer to 400 km
a_xfer = (r_337 + r_400) / 2
v_xfer_337 = math.sqrt(MU * (2/r_337 - 1/a_xfer))
# We need to know current velocity at 337 km apogee
# Using user-stated 1800 m/s
dv1 = v_xfer_337 - v_apogee_user
print(f"  v at 337km (user)   = {v_apogee_user:.0f} m/s")
print(f"  v_transfer at 337km = {v_xfer_337:.2f} m/s")
print(f"  dv1 (raise apogee)  = {dv1:.2f} m/s")

# Burn 2 at 400 km: circularize
v_xfer_400 = math.sqrt(MU * (2/r_400 - 1/a_xfer))
dv2 = v_circ_400 - v_xfer_400
print(f"  v_transfer at 400km = {v_xfer_400:.2f} m/s")
print(f"  v_circular at 400km = {v_circ_400:.2f} m/s")
print(f"  dv2 (circularize)   = {dv2:.2f} m/s")

dv_orbital = dv1 + dv2
print(f"  Total orbital dv    = {dv_orbital:.2f} m/s")

# Full budget
dv_gravity = 50.0
dv_steering = 20.0
dv_margin = 150.0
dv_total = dv_orbital + dv_gravity + dv_steering + dv_margin
print(f"\n  Gravity loss        = {dv_gravity:.0f} m/s")
print(f"  Steering loss       = {dv_steering:.0f} m/s")
print(f"  Margin              = {dv_margin:.0f} m/s")
print(f"  *** TOTAL S2 dv *** = {dv_total:.1f} m/s")

# ===========================================================================
# PART 4: S2 ENGINE SIZING (MULTIPLE OPTIONS)
# ===========================================================================
print("\n" + "=" * 80)
print("PART 4: S2 ENGINE SIZING")
print("=" * 80)

S2_WET = 120000.0

print(f"\n  S2 wet mass = {S2_WET:,.0f} kg")
print(f"  Total dv    = {dv_total:.1f} m/s")

configs = [
    ("LOX/RP-1 (Merlin Vac class)", 320.0),
    ("LOX/LCH4 (Raptor Vac class)", 350.0),
    ("LOX/LH2 (RL-10 class)",       450.0),
]

for name, isp in configs:
    print(f"\n  --- {name}, Isp={isp:.0f}s ---")
    ve = isp * G0
    mr = math.exp(dv_total / ve)
    s2_dry = S2_WET / mr
    s2_prop = S2_WET - s2_dry

    print(f"  v_exhaust        = {ve:.1f} m/s")
    print(f"  mass ratio       = {mr:.4f}")
    print(f"  S2 dry mass      = {s2_dry:,.0f} kg")
    print(f"  S2 propellant    = {s2_prop:,.0f} kg")
    print(f"  Prop fraction    = {s2_prop/S2_WET*100:.1f}%")

    # Engine sizing for various T/W
    print(f"  {'T/W':>6} {'Thrust(kN)':>12} {'mdot(kg/s)':>12} {'Burn(s)':>10} {'Burn(min)':>10} {'Neng':>6}")
    for tw in [0.3, 0.5, 0.7, 1.0]:
        T = tw * S2_WET * G0
        md = T / (isp * G0)
        bt = s2_prop / md
        # Number of engines if each is ~100 kN
        n_eng = max(1, round(T / 100e3))
        print(f"  {tw:6.1f} {T/1e3:12.1f} {md:12.1f} {bt:10.0f} {bt/60:10.1f} {n_eng:6d}")

# ===========================================================================
# PART 5: RECOMMENDED CONFIGURATION
# ===========================================================================
print("\n" + "=" * 80)
print("PART 5: RECOMMENDED S2 CONFIGURATION")
print("=" * 80)

# Go with LOX/RP-1 for S1/S2 propellant commonality
ISP_S2 = 320.0
ve_s2 = ISP_S2 * G0
mr_s2 = math.exp(dv_total / ve_s2)
S2_DRY = S2_WET / mr_s2
S2_PROP = S2_WET - S2_DRY

# Choose T/W = 0.5 for reasonable burn time
S2_TW_TARGET = 0.5
S2_THRUST = S2_TW_TARGET * S2_WET * G0
S2_MDOT = S2_THRUST / (ISP_S2 * G0)
S2_BURN = S2_PROP / S2_MDOT
S2_TW_ACTUAL = S2_THRUST / (S2_WET * G0)

# Payload estimate
S2_STRUCT_FRAC = 0.08  # 8% structural
S2_STRUCT = S2_STRUCT_FRAC * S2_WET
S2_PAYLOAD = S2_DRY - S2_STRUCT

print(f"\n  PROPELLANT: LOX/RP-1")
print(f"  S2_WET_MASS          = {S2_WET:>12,.0f} kg")
print(f"  S2_DRY_MASS          = {S2_DRY:>12,.0f} kg")
print(f"  S2_PROPELLANT_MASS   = {S2_PROP:>12,.0f} kg")
print(f"  S2_THRUST            = {S2_THRUST:>12,.0f} N  ({S2_THRUST/1e3:.1f} kN)")
print(f"  S2_ISP_VAC           = {ISP_S2:>12.0f} s")
print(f"  S2_MASS_FLOW_RATE    = {S2_MDOT:>12.2f} kg/s")
print(f"  S2_BURN_TIME         = {S2_BURN:>12.0f} s  ({S2_BURN/60:.1f} min)")
print(f"  S2_T/W (initial)     = {S2_TW_ACTUAL:>12.3f}")
print(f"  S2_T/W (burnout)     = {S2_THRUST/(S2_DRY*G0):>12.3f}")
print(f"  S2_STRUCTURE (est)   = {S2_STRUCT:>12,.0f} kg ({S2_STRUCT_FRAC*100:.0f}%)")
print(f"  S2_PAYLOAD (est)     = {S2_PAYLOAD:>12,.0f} kg")

# Alternative: Higher Isp with LOX/LCH4
print("\n  --- ALTERNATIVE: LOX/LCH4 (Raptor Vac) ---")
ISP_ALT = 350.0
ve_alt = ISP_ALT * G0
mr_alt = math.exp(dv_total / ve_alt)
DRY_ALT = S2_WET / mr_alt
PROP_ALT = S2_WET - DRY_ALT
THRUST_ALT = 0.5 * S2_WET * G0
MDOT_ALT = THRUST_ALT / (ISP_ALT * G0)
BURN_ALT = PROP_ALT / MDOT_ALT

print(f"  S2_DRY_MASS          = {DRY_ALT:>12,.0f} kg")
print(f"  S2_PROPELLANT_MASS   = {PROP_ALT:>12,.0f} kg")
print(f"  S2_BURN_TIME         = {BURN_ALT:>12.0f} s  ({BURN_ALT/60:.1f} min)")
print(f"  S2_PAYLOAD (est)     = {DRY_ALT - S2_STRUCT:>12,.0f} kg")

# ===========================================================================
# PART 6: S1 LANDING FUEL RESERVE (CORRECTED)
# ===========================================================================
print("\n" + "=" * 80)
print("PART 6: S1 LANDING FUEL RESERVE")
print("=" * 80)

S1_DRY = 30000.0
S1_PROP = 390000.0
v_meco = 2726.0

# RTLS (Return to Launch Site) - needs boostback
print("\n  --- RTLS (Boostback) ---")
# F9 RTLS reserves ~40-45% of prop. Let's compute from first principles.
# Boostback: need to null horizontal vel and reverse (~1.5x MECO vel due to gravity)
# Entry burn: ~200-400 m/s
# Landing burn: ~300-600 m/s (hoverslam from terminal velocity)
dv_bb = 0.65 * v_meco  # boostback (must reverse and re-aim)
dv_eb = 350.0           # entry burn
dv_lb = 450.0           # landing burn
dv_rtls = dv_bb + dv_eb + dv_lb
print(f"  Boostback dv     = {dv_bb:.0f} m/s (65% of MECO vel)")
print(f"  Entry burn dv    = {dv_eb:.0f} m/s")
print(f"  Landing burn dv  = {dv_lb:.0f} m/s")
print(f"  Total RTLS dv    = {dv_rtls:.0f} m/s")

# Use SL Isp for landing, VAC for boostback, average
isp_rtls = 295.0  # weighted average
ve_rtls = isp_rtls * G0
mr_rtls = math.exp(dv_rtls / ve_rtls)
fuel_rtls = S1_DRY * (mr_rtls - 1)
pct_rtls = fuel_rtls / S1_PROP * 100
print(f"  Mass ratio       = {mr_rtls:.4f}")
print(f"  Fuel needed      = {fuel_rtls:,.0f} kg ({pct_rtls:.1f}% of S1 prop)")
print(f"  Prop for ascent  = {S1_PROP - fuel_rtls:,.0f} kg")

# ASDS (Droneship) - no boostback
print("\n  --- ASDS (Droneship Landing) ---")
dv_eb2 = 400.0
dv_lb2 = 450.0
dv_asds = dv_eb2 + dv_lb2
print(f"  Entry burn dv    = {dv_eb2:.0f} m/s")
print(f"  Landing burn dv  = {dv_lb2:.0f} m/s")
print(f"  Total ASDS dv    = {dv_asds:.0f} m/s")

isp_asds = 300.0
ve_asds = isp_asds * G0
mr_asds = math.exp(dv_asds / ve_asds)
fuel_asds = S1_DRY * (mr_asds - 1)
pct_asds = fuel_asds / S1_PROP * 100
print(f"  Mass ratio       = {mr_asds:.4f}")
print(f"  Fuel needed      = {fuel_asds:,.0f} kg ({pct_asds:.1f}% of S1 prop)")
print(f"  Prop for ascent  = {S1_PROP - fuel_asds:,.0f} kg")

# Expendable (no recovery)
print("\n  --- Expendable (no recovery) ---")
print(f"  Fuel for landing = 0 kg")
print(f"  Prop for ascent  = {S1_PROP:,.0f} kg")

# ===========================================================================
# PART 7: S1 EARLY CUTOFF (80% BURN) ANALYSIS
# ===========================================================================
print("\n" + "=" * 80)
print("PART 7: S1 EARLY CUTOFF (80% FUEL BURN)")
print("=" * 80)

S1_WET = 420000.0
ISP_AVG = 295.0
ve_s1 = ISP_AVG * G0
THRUST_S1 = 7.6e6
mdot_s1 = THRUST_S1 / (ISP_AVG * G0)

print(f"\n  S1 parameters:")
print(f"  S1 mass flow rate = {mdot_s1:.1f} kg/s")
print(f"  Avg Isp           = {ISP_AVG:.0f} s")

m0 = S1_WET + S2_WET  # 540,000
for frac in [1.0, 0.90, 0.85, 0.80, 0.75, 0.70]:
    prop_used = S1_PROP * frac
    prop_left = S1_PROP - prop_used
    mf = S1_DRY + prop_left + S2_WET
    dv = ve_s1 * math.log(m0 / mf)
    burn_t = prop_used / mdot_s1

    # Rough MECO velocity (scales roughly linearly minus gravity/drag)
    # At 100%, MECO = 2726 m/s. Scale proportionally to dv.
    dv_100 = ve_s1 * math.log(m0 / (S1_DRY + S2_WET))
    v_meco_est = v_meco * (dv / dv_100)  # first order

    # Rough MECO altitude (shorter burn = lower altitude)
    alt_meco_est = 109.0 * (burn_t / (S1_PROP / mdot_s1))  # crude linear

    pct = frac * 100
    print(f"\n  --- {pct:.0f}% burn ---")
    print(f"  Propellant used  = {prop_used:,.0f} kg")
    print(f"  Propellant left  = {prop_left:,.0f} kg (for landing)")
    print(f"  Stack mass @MECO = {mf:,.0f} kg")
    print(f"  Ideal dv         = {dv:.0f} m/s")
    print(f"  Burn time        = {burn_t:.1f} s ({burn_t/60:.1f} min)")
    print(f"  Est MECO vel     ~ {v_meco_est:.0f} m/s")
    print(f"  Est MECO alt     ~ {alt_meco_est:.0f} km")
    print(f"  Landing fuel pct = {(1-frac)*100:.0f}%")

    # S2 dv required to reach LEO from this MECO
    dv_s2_needed = v_circ_400 - v_meco_est + 220  # +220 for losses
    print(f"  S2 dv needed     ~ {dv_s2_needed:.0f} m/s (rough)")
    mr_s2_needed = math.exp(dv_s2_needed / ve_s2)
    if mr_s2_needed < 20:
        s2_dry = S2_WET / mr_s2_needed
        s2_prop = S2_WET - s2_dry
        print(f"  S2 mass ratio    = {mr_s2_needed:.3f}")
        print(f"  S2 dry mass      = {s2_dry:,.0f} kg")
        print(f"  S2 propellant    = {s2_prop:,.0f} kg")
    else:
        print(f"  S2 mass ratio    = {mr_s2_needed:.1f} -- INFEASIBLE")

# ===========================================================================
# PART 8: FINAL SUMMARY TABLE
# ===========================================================================
print("\n")
print("=" * 80)
print("FINAL SUMMARY: RECOMMENDED CONSTANTS FOR SIMULATION")
print("=" * 80)

print("""
# ===========================================================================
# STAGE 2 PARAMETERS (LOX/RP-1, Merlin Vacuum class)
# ===========================================================================
# Orbital target: 400 km circular LEO
# v_circular(400km) = 7672.6 m/s
# MECO conditions: alt=109km, v=2726 m/s, apogee~337km
# v_apogee ~ 1800 m/s (user-stated)
# Delta-v budget: ~6093 m/s (orbital + losses + margin)
""")

# Use the user-stated apogee velocity
print(f"STAGE2_WET_MASS        = {S2_WET:.0f}    # kg (total S2 mass)")
print(f"STAGE2_DRY_MASS        = {S2_DRY:.0f}    # kg")
print(f"STAGE2_PROPELLANT_MASS = {S2_PROP:.0f}    # kg")
print(f"STAGE2_THRUST          = {S2_THRUST:.0f}  # N ({S2_THRUST/1e3:.0f} kN, T/W={S2_TW_ACTUAL:.2f})")
print(f"STAGE2_ISP_VAC         = {ISP_S2:.0f}       # s (LOX/RP-1 vacuum)")
print(f"STAGE2_MASS_FLOW_RATE  = {S2_MDOT:.2f}    # kg/s")

print(f"""
# Stage 1 Landing Reserve (Droneship)
STAGE1_LANDING_FUEL    = {fuel_asds:.0f}    # kg ({pct_asds:.1f}% of S1 prop)
# Stage 1 Landing Reserve (RTLS)
STAGE1_RTLS_FUEL       = {fuel_rtls:.0f}    # kg ({pct_rtls:.1f}% of S1 prop)

# Delta-V Budget
DV_ORBITAL_S2          = {dv_orbital:.1f}    # m/s (Hohmann: 337km -> 400km circular)
DV_GRAVITY_LOSS_S2     = {dv_gravity:.1f}      # m/s
DV_STEERING_LOSS_S2    = {dv_steering:.1f}      # m/s
DV_MARGIN_S2           = {dv_margin:.1f}     # m/s
DV_TOTAL_S2            = {dv_total:.1f}    # m/s

# Orbital Mechanics
V_CIRCULAR_400KM       = {v_circ_400:.2f}  # m/s
""")

print("=" * 80)
print("COMPUTATION COMPLETE")
print("=" * 80)
