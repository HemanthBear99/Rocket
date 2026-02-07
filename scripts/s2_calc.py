"""
RLV Stage 2 Engine Parameter Computation
=========================================
Computes realistic S2 parameters for circularization at 400 km LEO.
"""
import math

print("=" * 80)
print("RLV STAGE 2 ENGINE PARAMETER COMPUTATION")
print("=" * 80)

# ===========================================================================
# SECTION 1: FUNDAMENTAL CONSTANTS
# ===========================================================================
print("\n--- SECTION 1: FUNDAMENTAL CONSTANTS ---")
MU = 3.986004418e14   # m^3/s^2 (Earth gravitational parameter)
R_E = 6.371e6          # m (Earth mean radius)
G0 = 9.80665           # m/s^2 (standard gravity)

print(f"MU_EARTH    = {MU:.6e} m^3/s^2")
print(f"R_EARTH     = {R_E:.3e} m")
print(f"G0          = {G0:.5f} m/s^2")

# ===========================================================================
# SECTION 2: CURRENT S1 PARAMETERS (from constants.py)
# ===========================================================================
print("\n--- SECTION 2: CURRENT S1 PARAMETERS ---")
S1_DRY   = 30000.0     # kg
S1_PROP  = 390000.0     # kg
S1_WET   = 420000.0     # kg
S2_TOTAL = 120000.0     # kg (currently inert)
STACK_TOTAL = S1_WET + S2_TOTAL
THRUST_S1 = 7.6e6       # N
ISP_SL    = 282.0       # s (sea level)
ISP_VAC   = 311.0       # s (vacuum)

print(f"S1 dry mass       = {S1_DRY:,.0f} kg")
print(f"S1 propellant     = {S1_PROP:,.0f} kg")
print(f"S1 wet mass       = {S1_WET:,.0f} kg")
print(f"S2 total (inert)  = {S2_TOTAL:,.0f} kg")
print(f"Stack total       = {STACK_TOTAL:,.0f} kg")
print(f"S1 thrust         = {THRUST_S1/1e6:.1f} MN")
print(f"S1 Isp (SL)       = {ISP_SL:.0f} s")
print(f"S1 Isp (vac)      = {ISP_VAC:.0f} s")
print(f"S1 mass flow      = {THRUST_S1/(ISP_SL*G0):.1f} kg/s")
print(f"S1 T/W at launch  = {THRUST_S1/(STACK_TOTAL*G0):.2f}")

# ===========================================================================
# SECTION 3: ORBITAL MECHANICS - TARGET ORBIT
# ===========================================================================
print("\n--- SECTION 3: ORBITAL MECHANICS - TARGET ORBIT ---")
h_target = 400e3  # m (400 km LEO)
r_target = R_E + h_target
v_circular = math.sqrt(MU / r_target)

print(f"Target altitude   = {h_target/1e3:.0f} km")
print(f"Target radius     = {r_target/1e3:.1f} km")
print(f"v_circular(400km) = sqrt({MU:.3e} / {r_target:.3e})")
print(f"v_circular(400km) = {v_circular:.2f} m/s")

# Reference circular velocities
for h_km in [200, 300, 400, 500]:
    r = R_E + h_km * 1e3
    vc = math.sqrt(MU / r)
    print(f"  v_circ({h_km}km) = {vc:.1f} m/s")

# ===========================================================================
# SECTION 4: CURRENT MECO CONDITIONS (from sim)
# ===========================================================================
print("\n--- SECTION 4: CURRENT MECO CONDITIONS ---")
alt_meco = 109e3   # m
v_meco   = 2726.0  # m/s (inertial)
gamma_meco = 10.0  # degrees (approximate flight path angle at MECO)

r_meco = R_E + alt_meco
print(f"MECO altitude     = {alt_meco/1e3:.0f} km")
print(f"MECO radius       = {r_meco/1e3:.1f} km")
print(f"MECO velocity     = {v_meco:.0f} m/s (inertial)")

# Compute orbital energy at MECO
E_meco = 0.5 * v_meco**2 - MU / r_meco
print(f"Specific energy   = {E_meco/1e6:.3f} MJ/kg")
print(f"Orbit bound?      = {'YES' if E_meco < 0 else 'NO (ESCAPE!)'}")

# Semi-major axis from vis-viva: E = -MU/(2a) => a = -MU/(2E)
a_meco = -MU / (2 * E_meco)
print(f"Semi-major axis   = {a_meco/1e3:.1f} km")

# Account for flight path angle
gamma_rad = math.radians(gamma_meco)
v_horiz = v_meco * math.cos(gamma_rad)
v_vert  = v_meco * math.sin(gamma_rad)
h_ang = r_meco * v_horiz  # specific angular momentum = r * v_tangential

print(f"v_horizontal      = {v_horiz:.1f} m/s")
print(f"v_vertical        = {v_vert:.1f} m/s")
print(f"h (ang. momentum) = {h_ang:.1f} m^2/s")

# Eccentricity and apsides
e_meco = math.sqrt(1 - h_ang**2 / (MU * a_meco))
r_apogee = a_meco * (1 + e_meco)
r_perigee = a_meco * (1 - e_meco)
alt_apogee = r_apogee - R_E
alt_perigee = r_perigee - R_E

print(f"Eccentricity      = {e_meco:.6f}")
print(f"Apogee radius     = {r_apogee/1e3:.1f} km")
print(f"Apogee altitude   = {alt_apogee/1e3:.1f} km")
print(f"Perigee radius    = {r_perigee/1e3:.1f} km")
print(f"Perigee altitude  = {alt_perigee/1e3:.1f} km")

# Velocity at apogee (angular momentum conservation: v_apo = h/r_apo)
v_apogee = h_ang / r_apogee
print(f"v_apogee          = {v_apogee:.2f} m/s (tangential)")

# Sensitivity to gamma_meco
print("\n  Sensitivity to gamma_meco:")
for gamma_deg in [0, 5, 10, 15, 20]:
    gr = math.radians(gamma_deg)
    vh = v_meco * math.cos(gr)
    h = r_meco * vh
    E = 0.5 * v_meco**2 - MU / r_meco
    a = -MU / (2 * E)
    if h**2 / (MU * a) <= 1.0:
        e = math.sqrt(1 - h**2 / (MU * a))
        r_apo = a * (1 + e)
        v_apo = h / r_apo
        print(f"  gamma={gamma_deg:2d}deg: apogee={(r_apo - R_E)/1e3:.1f}km, v_apo={v_apo:.1f}m/s")
    else:
        print(f"  gamma={gamma_deg:2d}deg: CIRCULAR OR INVALID")

# ===========================================================================
# SECTION 5: DELTA-V BUDGET FOR S2
# ===========================================================================
print("\n--- SECTION 5: DELTA-V BUDGET FOR S2 ---")
print("Using gamma_meco = 10 deg as baseline:")

r_apo_current = r_apogee
r_target_400 = R_E + 400e3

# OPTION A: Hohmann transfer from apogee to 400 km circular
print("\n  OPTION A: Hohmann transfer from ~337km apogee to 400km circular")
a_transfer = (r_apo_current + r_target_400) / 2
v_transfer_perigee = math.sqrt(MU * (2 / r_apo_current - 1 / a_transfer))
dv1 = v_transfer_perigee - v_apogee
print(f"  Current apogee       = {alt_apogee/1e3:.1f} km")
print(f"  v at current apogee  = {v_apogee:.2f} m/s")
print(f"  Transfer a           = {a_transfer/1e3:.1f} km")
print(f"  v_transfer at peri   = {v_transfer_perigee:.2f} m/s")
print(f"  dv1 (raise apogee)   = {dv1:.2f} m/s")

v_transfer_apogee = math.sqrt(MU * (2 / r_target_400 - 1 / a_transfer))
v_circ_400 = math.sqrt(MU / r_target_400)
dv2 = v_circ_400 - v_transfer_apogee
print(f"  v_transfer at 400km  = {v_transfer_apogee:.2f} m/s")
print(f"  v_circular at 400km  = {v_circ_400:.2f} m/s")
print(f"  dv2 (circularize)    = {dv2:.2f} m/s")

dv_total_A = dv1 + dv2
print(f"  TOTAL delta-v (A)    = {dv_total_A:.2f} m/s")

# OPTION B: Hypothetical 109x400 orbit
print("\n  OPTION B: If orbit were 109x400km, single circularization burn:")
a_meco_to_400 = (r_meco + r_target_400) / 2
v_at_400_hypothetical = math.sqrt(MU * (2 / r_target_400 - 1 / a_meco_to_400))
dv_circ_hypothetical = v_circ_400 - v_at_400_hypothetical
print(f"  If orbit were 109x400km:")
print(f"    v at 400km apogee  = {v_at_400_hypothetical:.2f} m/s")
print(f"    dv to circularize  = {dv_circ_hypothetical:.2f} m/s")

# OPTION C: Direct
print("\n  OPTION C: Direct delta-v from apogee velocity to v_circular:")
dv_direct = v_circ_400 - v_apogee
print(f"  dv_direct = v_circ - v_apo = {v_circ_400:.2f} - {v_apogee:.2f} = {dv_direct:.2f} m/s")

# REALISTIC BUDGET
print("\n  REALISTIC TOTAL DELTA-V BUDGET:")
dv_orbital = dv_total_A
dv_gravity_loss = 50.0
dv_steering_loss = 20.0
dv_margin = 100.0
dv_total = dv_orbital + dv_gravity_loss + dv_steering_loss + dv_margin
print(f"  Orbital maneuvers    = {dv_orbital:.1f} m/s")
print(f"  Gravity loss         = {dv_gravity_loss:.1f} m/s")
print(f"  Steering loss        = {dv_steering_loss:.1f} m/s")
print(f"  Margin               = {dv_margin:.1f} m/s")
print(f"  *** TOTAL S2 dv ***  = {dv_total:.1f} m/s")

# ===========================================================================
# SECTION 6: S2 MASS BREAKDOWN USING TSIOLKOVSKY
# ===========================================================================
print("\n--- SECTION 6: S2 MASS BREAKDOWN (Tsiolkovsky) ---")

S2_TOTAL_MASS = 120000.0  # kg

for engine_name, isp_s2 in [("LOX/LH2 (Vac)", 340.0), ("LOX/RP-1 (Vac)", 320.0), ("LOX/LCH4 (Vac)", 350.0)]:
    print(f"\n  Engine type: {engine_name}, Isp = {isp_s2:.0f} s")
    v_e = isp_s2 * G0
    print(f"  Exhaust velocity     = {v_e:.1f} m/s")

    mass_ratio = math.exp(dv_total / v_e)
    print(f"  Mass ratio (m0/mf)   = exp({dv_total:.1f}/{v_e:.1f}) = {mass_ratio:.4f}")

    m0 = S2_TOTAL_MASS
    mf = m0 / mass_ratio
    S2_dry = mf
    S2_prop = m0 - mf

    structural_fraction = S2_dry / S2_TOTAL_MASS
    propellant_fraction = S2_prop / S2_TOTAL_MASS

    print(f"  S2 wet mass (m0)     = {m0:,.0f} kg")
    print(f"  S2 dry mass (mf)     = {S2_dry:,.0f} kg")
    print(f"  S2 propellant        = {S2_prop:,.0f} kg")
    print(f"  Structural fraction  = {structural_fraction:.4f} ({structural_fraction*100:.1f}%)")
    print(f"  Propellant fraction  = {propellant_fraction:.4f} ({propellant_fraction*100:.1f}%)")

    S2_structure = 0.10 * S2_TOTAL_MASS
    S2_payload = S2_dry - S2_structure
    if S2_payload > 0:
        print(f"  --- If S2 structure = 10% of wet ({S2_structure:,.0f} kg):")
        print(f"      Payload to orbit = {S2_payload:,.0f} kg")
    else:
        print(f"  WARNING: S2 dry mass < structural estimate!")

    print(f"  --- Engine Sizing ---")
    for tw_ratio in [0.5, 0.7, 1.0]:
        thrust = tw_ratio * m0 * G0
        mdot = thrust / (isp_s2 * G0)
        burn_time = S2_prop / mdot
        print(f"    T/W={tw_ratio}: Thrust={thrust/1e3:.1f}kN, mdot={mdot:.1f}kg/s, burn={burn_time:.0f}s")

# ===========================================================================
# SECTION 7: RECOMMENDED S2 CONFIGURATION
# ===========================================================================
print("\n--- SECTION 7: RECOMMENDED S2 CONFIGURATION ---")

ISP_S2 = 320.0  # s (LOX/RP-1 vacuum)
v_e_s2 = ISP_S2 * G0

mass_ratio = math.exp(dv_total / v_e_s2)
S2_WET = S2_TOTAL_MASS
S2_DRY = S2_WET / mass_ratio
S2_PROPELLANT = S2_WET - S2_DRY

S2_THRUST = 100e3  # N (100 kN)
S2_MDOT = S2_THRUST / (ISP_S2 * G0)
S2_BURN_TIME = S2_PROPELLANT / S2_MDOT
S2_TW = S2_THRUST / (S2_WET * G0)

print(f"  ISP_S2              = {ISP_S2:.0f} s")
print(f"  Exhaust velocity    = {v_e_s2:.1f} m/s")
print(f"  Delta-v required    = {dv_total:.1f} m/s")
print(f"  Mass ratio          = {mass_ratio:.4f}")
print(f"")
print(f"  S2_WET_MASS         = {S2_WET:,.0f} kg")
print(f"  S2_DRY_MASS         = {S2_DRY:,.0f} kg")
print(f"  S2_PROPELLANT_MASS  = {S2_PROPELLANT:,.0f} kg")
print(f"  S2_THRUST           = {S2_THRUST/1e3:.0f} kN")
print(f"  S2_ISP              = {ISP_S2:.0f} s")
print(f"  S2_MASS_FLOW_RATE   = {S2_MDOT:.2f} kg/s")
print(f"  S2_BURN_TIME        = {S2_BURN_TIME:.0f} s ({S2_BURN_TIME/60:.1f} min)")
print(f"  S2_T/W (initial)    = {S2_TW:.3f}")

S2_STRUCTURE = 0.10 * S2_WET
S2_PAYLOAD = S2_DRY - S2_STRUCTURE
print(f"")
print(f"  S2 structure (est)  = {S2_STRUCTURE:,.0f} kg (10% of wet)")
print(f"  Payload to LEO      = {S2_PAYLOAD:,.0f} kg")
if S2_PAYLOAD < 0:
    print(f"  NOTE: Negative payload means S2 is undersized or needs higher Isp!")
    S2_STRUCTURE_REAL = S2_DRY
    print(f"  (All S2 dry mass is structure: {S2_STRUCTURE_REAL:,.0f} kg)")

# ===========================================================================
# SECTION 8: S1 FUEL RESERVE FOR BOOSTER LANDING
# ===========================================================================
print("\n--- SECTION 8: S1 FUEL RESERVE FOR BOOSTER LANDING ---")
print("  Falcon-9 style boostback + landing burn analysis")

print("\n  --- Boostback (RTLS) Scenario ---")
dv_boostback = v_meco * 0.6
dv_entry = 300.0
dv_landing = 500.0
dv_total_landing = dv_boostback + dv_entry + dv_landing

print(f"  Boostback dv        = {dv_boostback:.0f} m/s")
print(f"  Entry burn dv       = {dv_entry:.0f} m/s")
print(f"  Landing burn dv     = {dv_landing:.0f} m/s")
print(f"  Total landing dv    = {dv_total_landing:.0f} m/s")

ISP_EFFECTIVE_LANDING = 300.0
v_e_landing = ISP_EFFECTIVE_LANDING * G0

mass_ratio_landing = math.exp(dv_total_landing / v_e_landing)
fuel_for_landing = S1_DRY * (mass_ratio_landing - 1)
fuel_fraction = fuel_for_landing / S1_PROP

print(f"  Mass ratio (landing) = {mass_ratio_landing:.4f}")
print(f"  S1 dry mass         = {S1_DRY:,.0f} kg")
print(f"  Fuel for landing    = {fuel_for_landing:,.0f} kg")
print(f"  As % of S1 prop     = {fuel_fraction*100:.1f}%")
print(f"  S1 prop available   = {S1_PROP - fuel_for_landing:,.0f} kg for ascent")

print("\n  --- Downrange Landing (Droneship/ASDS) Scenario ---")
dv_entry_dr = 400.0
dv_landing_dr = 500.0
dv_total_dr = dv_entry_dr + dv_landing_dr

mass_ratio_dr = math.exp(dv_total_dr / v_e_landing)
fuel_for_landing_dr = S1_DRY * (mass_ratio_dr - 1)
fuel_fraction_dr = fuel_for_landing_dr / S1_PROP

print(f"  Entry burn dv       = {dv_entry_dr:.0f} m/s")
print(f"  Landing burn dv     = {dv_landing_dr:.0f} m/s")
print(f"  Total landing dv    = {dv_total_dr:.0f} m/s")
print(f"  Mass ratio          = {mass_ratio_dr:.4f}")
print(f"  Fuel for landing    = {fuel_for_landing_dr:,.0f} kg")
print(f"  As % of S1 prop     = {fuel_fraction_dr*100:.1f}%")
print(f"  S1 prop available   = {S1_PROP - fuel_for_landing_dr:,.0f} kg for ascent")

# ===========================================================================
# SECTION 9: S1 EARLY CUTOFF (80% FUEL BURN) IMPACT
# ===========================================================================
print("\n--- SECTION 9: S1 EARLY CUTOFF (80% FUEL BURN) ---")

fuel_burn_fraction = 0.80
S1_PROP_USED = S1_PROP * fuel_burn_fraction
S1_PROP_RESERVED = S1_PROP - S1_PROP_USED
S1_MASS_AT_CUTOFF = S1_DRY + S1_PROP_RESERVED + S2_TOTAL

print(f"  Fuel burn fraction  = {fuel_burn_fraction*100:.0f}%")
print(f"  S1 prop used        = {S1_PROP_USED:,.0f} kg")
print(f"  S1 prop reserved    = {S1_PROP_RESERVED:,.0f} kg")
print(f"  Stack mass at MECO  = {S1_MASS_AT_CUTOFF:,.0f} kg")

ISP_AVG_S1 = 295.0  # average of SL and VAC for trajectory
v_e_s1 = ISP_AVG_S1 * G0

m0_stack = STACK_TOTAL
mf_stack_100 = S1_DRY + S2_TOTAL
mf_stack_80 = S1_DRY + S1_PROP_RESERVED + S2_TOTAL

dv_100_percent = v_e_s1 * math.log(m0_stack / mf_stack_100)
dv_80_percent = v_e_s1 * math.log(m0_stack / mf_stack_80)
dv_lost = dv_100_percent - dv_80_percent

print(f"")
print(f"  S1 dv (100% burn)   = {ISP_AVG_S1}s * {G0} * ln({m0_stack}/{mf_stack_100})")
print(f"                       = {dv_100_percent:.1f} m/s")
print(f"  S1 dv (80% burn)    = {ISP_AVG_S1}s * {G0} * ln({m0_stack}/{mf_stack_80})")
print(f"                       = {dv_80_percent:.1f} m/s")
print(f"  Delta-v LOST         = {dv_lost:.1f} m/s")
print(f"  Fractional loss      = {dv_lost/dv_100_percent*100:.1f}%")

v_meco_80_approx = v_meco - dv_lost * 0.7

burn_time_100 = S1_PROP / (THRUST_S1 / (ISP_AVG_S1 * G0))
burn_time_80 = S1_PROP_USED / (THRUST_S1 / (ISP_AVG_S1 * G0))

print(f"")
print(f"  Burn time (100%)     = {burn_time_100:.1f} s")
print(f"  Burn time (80%)      = {burn_time_80:.1f} s")
print(f"  Time reduction       = {burn_time_100 - burn_time_80:.1f} s")
print(f"")
print(f"  Estimated MECO velocity (80%) = {v_meco_80_approx:.0f} m/s (rough)")
print(f"  Estimated MECO altitude (80%) ~ 70-85 km (lower due to shorter burn)")

# Impact on S2
dv_deficit = v_meco - v_meco_80_approx
dv_s2_with_early_cutoff = dv_total + dv_deficit

print(f"")
print(f"  S2 dv deficit from early S1 cutoff = {dv_deficit:.0f} m/s")
print(f"  S2 total dv required (with 80%)    = {dv_s2_with_early_cutoff:.0f} m/s")

mass_ratio_new = math.exp(dv_s2_with_early_cutoff / v_e_s2)
S2_DRY_NEW = S2_WET / mass_ratio_new
S2_PROP_NEW = S2_WET - S2_DRY_NEW

print(f"  S2 mass ratio (new) = {mass_ratio_new:.4f}")
print(f"  S2 dry mass (new)   = {S2_DRY_NEW:,.0f} kg")
print(f"  S2 propellant (new) = {S2_PROP_NEW:,.0f} kg")
print(f"  S2 burn time (new)  = {S2_PROP_NEW / S2_MDOT:.0f} s")

# ===========================================================================
# SECTION 10: FINAL RECOMMENDED PARAMETERS
# ===========================================================================
print("\n")
print("=" * 80)
print("SECTION 10: FINAL RECOMMENDED PARAMETERS FOR constants.py")
print("=" * 80)

FUEL_RESERVE = fuel_for_landing_dr
S1_PROP_ASCENT = S1_PROP - FUEL_RESERVE
FUEL_RESERVE_PCT = FUEL_RESERVE / S1_PROP * 100

print(f"")
print(f"# Stage 1 (with landing reserve for droneship)")
print(f"STAGE1_DRY_MASS        = {S1_DRY:,.0f}     # kg")
print(f"STAGE1_PROPELLANT_MASS = {S1_PROP:,.0f}   # kg (total)")
print(f"STAGE1_PROP_ASCENT     = {S1_PROP_ASCENT:,.0f}   # kg (available for ascent)")
print(f"STAGE1_LANDING_RESERVE = {FUEL_RESERVE:,.0f}    # kg ({FUEL_RESERVE_PCT:.1f}% of total)")
print(f"STAGE1_WET_MASS        = {S1_WET:,.0f}   # kg")
print(f"STAGE1_THRUST          = {THRUST_S1/1e6:.1f}e6      # N")
print(f"STAGE1_ISP_SL          = {ISP_SL:.0f}         # s")
print(f"STAGE1_ISP_VAC         = {ISP_VAC:.0f}         # s")

print(f"")
print(f"# Stage 2")
print(f"STAGE2_WET_MASS        = {S2_WET:,.0f}   # kg")
print(f"STAGE2_DRY_MASS        = {S2_DRY:,.0f}   # kg")
print(f"STAGE2_PROPELLANT_MASS = {S2_PROPELLANT:,.0f}   # kg")
print(f"STAGE2_THRUST          = {S2_THRUST:,.0f}   # N ({S2_THRUST/1e3:.0f} kN)")
print(f"STAGE2_ISP             = {ISP_S2:.0f}         # s (vacuum, LOX/RP-1)")
print(f"STAGE2_MASS_FLOW_RATE  = {S2_MDOT:.2f}      # kg/s")
print(f"STAGE2_BURN_TIME       = {S2_BURN_TIME:.0f}        # s ({S2_BURN_TIME/60:.1f} min)")
print(f"STAGE2_T_W_INITIAL     = {S2_TW:.3f}       # dimensionless")

S2_ENGINE_MASS = 500.0
S2_AVIONICS = 200.0
S2_PAYLOAD_FAIRING = 2000.0
S2_TANKS = 0.05 * S2_WET
S2_PAYLOAD_NET = S2_DRY - S2_ENGINE_MASS - S2_AVIONICS - S2_PAYLOAD_FAIRING - S2_TANKS

print(f"")
print(f"# Stage 2 Mass Breakdown (estimate)")
print(f"S2_ENGINE_MASS         = {S2_ENGINE_MASS:,.0f}       # kg")
print(f"S2_AVIONICS            = {S2_AVIONICS:,.0f}       # kg")
print(f"S2_TANKS_STRUCTURE     = {S2_TANKS:,.0f}     # kg (5% wet mass)")
print(f"S2_PAYLOAD_FAIRING     = {S2_PAYLOAD_FAIRING:,.0f}     # kg")
print(f"S2_NET_PAYLOAD_TO_LEO  = {S2_PAYLOAD_NET:,.0f}   # kg")

print(f"")
print(f"# Orbital Parameters")
print(f"V_CIRCULAR_400KM       = {v_circ_400:.2f}    # m/s")
print(f"V_APOGEE_TRANSFER      = {v_apogee:.2f}   # m/s")
print(f"DV_TOTAL_S2            = {dv_total:.1f}      # m/s")
print(f"DV_RAISE_APOGEE        = {dv1:.2f}      # m/s")
print(f"DV_CIRCULARIZE         = {dv2:.2f}      # m/s")

# ===========================================================================
# SECTION 11: VERIFICATION
# ===========================================================================
print(f"")
print("=" * 80)
print("SECTION 11: VERIFICATION & SANITY CHECKS")
print("=" * 80)

total_system_dv_ideal = v_e_s1 * math.log(STACK_TOTAL / mf_stack_100) + v_e_s2 * math.log(S2_WET / S2_DRY)
print(f"")
print(f"  Total ideal dv (S1+S2) = {total_system_dv_ideal:.0f} m/s")
print(f"  v_circular(400km)      = {v_circ_400:.0f} m/s")
print(f"  v_circular + losses    ~ {v_circ_400 + 1500:.0f} m/s (typical ~1.5 km/s losses)")
print(f"  Margin                 = {total_system_dv_ideal - (v_circ_400 + 1500):.0f} m/s")

print(f"")
print(f"  S1 T/W at liftoff     = {THRUST_S1/(STACK_TOTAL*G0):.2f} (should be 1.2-1.8)")
print(f"  S2 T/W at ignition    = {S2_TW:.3f} (should be 0.3-1.0 for upper stage)")
print(f"  S2 T/W at burnout     = {S2_THRUST/(S2_DRY*G0):.3f}")

print(f"")
S1_burn_time = S1_PROP / (THRUST_S1 / (ISP_AVG_S1 * G0))
print(f"  S1 burn time          = {S1_burn_time:.0f} s ({S1_burn_time/60:.1f} min)")
print(f"  S2 burn time          = {S2_BURN_TIME:.0f} s ({S2_BURN_TIME/60:.1f} min)")

print(f"")
print(f"  S1 prop fraction      = {S1_PROP/S1_WET*100:.1f}%")
print(f"  S2 prop fraction      = {S2_PROPELLANT/S2_WET*100:.1f}%")
print(f"  (Typical: 85-95% for S1, 80-92% for S2)")

v_staging = v_meco
v_LEO = v_circ_400
print(f"")
print(f"  Staging velocity      = {v_staging:.0f} m/s ({v_staging/v_LEO*100:.1f}% of LEO vel)")
print(f"  (Typical staging: 30-45% of LEO velocity)")

print(f"")
print("=" * 80)
print("COMPUTATION COMPLETE")
print("=" * 80)
