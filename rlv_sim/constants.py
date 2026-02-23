"""
RLV Phase-I Ascent Simulation - Physical Constants and Vehicle Parameters

This module defines all physical constants, Earth parameters, vehicle specifications,
and control system parameters used throughout the simulation.

VALUES FROM: Developer_Implementation.pdf - Appendix A
"""

import numpy as np

# =============================================================================
# EARTH PARAMETERS (Table 2)
# =============================================================================

# Gravitational parameter (m^3/s^2)
MU_EARTH = 3.986004418e14

# Earth mean radius (m)
R_EARTH = 6.371e6

# Standard gravitational acceleration at sea level (m/s^2)
G0 = 9.80665

# Atmospheric parameters (exponential model)
RHO_0 = 1.225  # Sea level density (kg/m^3)
H_SCALE = 8500.0  # Scale height (m)

# =============================================================================
# VEHICLE PARAMETERS - STACKED CONFIGURATION (Tables 3, A.11)
# Stage 1 + Stage 2 combined for Phase I
# =============================================================================

# Stage 1 Mass Properties (Table 3)
STAGE1_DRY_MASS = 30000.0  # kg (includes landing fuel margin)
STAGE1_PROPELLANT_MASS = 390000.0  # kg
STAGE1_WET_MASS = 420000.0  # kg

# Stage 2 Mass Properties (Table 11, doc page 49)
STAGE2_MASS = 120000.0  # kg (total S2 wet mass, constant during Phase I)
STAGE2_DRY_MASS = 8000.0      # kg (S2 structure, doc Table 11: mdry,2 = 8,000 kg)
STAGE2_PROPELLANT_MASS = 112000.0  # kg (LOX/RP-1, doc Table 11: mprop,2 = 112,000 kg)
STAGE2_WET_MASS = STAGE2_DRY_MASS + STAGE2_PROPELLANT_MASS  # 120,000 kg (verification)

# Stacked Vehicle Mass (A.13)
DRY_MASS = STAGE1_DRY_MASS + STAGE2_MASS  # 150,000 kg at S1 MECO
PROPELLANT_MASS = STAGE1_PROPELLANT_MASS  # 390,000 kg (S1 only)
INITIAL_MASS = STAGE1_WET_MASS + STAGE2_MASS  # 540,000 kg total

# S1 Fuel Reserve for Booster Recovery (RTLS)
# Budget breakdown for aggressive gravity-turn MECO (~50° from vertical):
#   boostback ~1100 m/s (reverse ~1500 m/s horizontal velocity)
#   entry     ~350 m/s  (aerodynamic braking supplement)
#   landing   ~800 m/s  (suicide burn from terminal velocity)
# Total ~2250 m/s.  At Isp=282s from 75t -> 30t:
#   dv_avail = 282*9.81*ln(75/30) = 2530 m/s — adequate with margin.
# 45,000 kg = 11.5% of S1 prop — within Falcon 9 RTLS range (~10-15%).
STAGE1_LANDING_FUEL_RESERVE = 45000.0  # kg reserved for booster landing
STAGE1_ASCENT_PROPELLANT = STAGE1_PROPELLANT_MASS - STAGE1_LANDING_FUEL_RESERVE  # 345,000 kg burned during ascent

# Mass flow rate: mdot = T / (Isp * g0)
# Computed after propulsion parameters are defined

# =============================================================================
# PROPULSION PARAMETERS (Table 5)
# =============================================================================

# Engine parameters
THRUST_MAGNITUDE = 7.6e6  # Maximum thrust (N) - sea level
ISP = 282.0  # Specific impulse at sea level (s)
# Note: PDF also specifies Isp_vac = 311 s for vacuum, but using SL for Phase I

# Mass flow rate: mdot = T / (Isp * g0)
MASS_FLOW_RATE = THRUST_MAGNITUDE / (ISP * G0)

# =============================================================================
# AERODYNAMIC PARAMETERS (Table 4)
# =============================================================================

DRAG_COEFFICIENT = 0.42  # Cd (constant during Phase I)
REFERENCE_AREA = 10.75  # Reference cross-sectional area (m^2)
REFERENCE_DIAMETER = 3.7  # m

# Normal force coefficient slope (per radian) for slender body
C_N_ALPHA = 4.0  # Typical value for rocket cylinder+nose cone

# =============================================================================
# INERTIA & GEOMETRY - VARIABLE MODEL
# =============================================================================

# Vehicle Dimensions (from Base)
H_STAGE1 = 40.0         # m (Height of Stage 1)
H_STACK = 60.0          # m (Total Stack Height)

# Center of Gravity Locations (Height from Base Z=0)
# Stage 1 Propellant is bottom-heavy (0-30m)
# Stage 2 Payload is top-heavy (at 50m)
H_CG_FULL = 20.0        # m (Low CG due to massive fuel load)
H_CG_EMPTY = 35.0       # m (High CG due to heavy upper stage and empty tanks)

# Aerodynamic Center of Pressure (Unstable: CP ahead of CG)
H_CP = 42.0             # m (Original Unstable Configuration - CP ahead of CG)

# Inertia Tensor - FULL (Launch Mass 540t)
# Ixx/Iyy massive due to fuel distribution
IXX_FULL = 5.36e7       # kg·m²
IZZ_FULL = 2.0e6        # kg·m²

# Inertia Tensor - EMPTY (MECO Mass 150t)
# Ixx/Iyy reduced significantly
IXX_EMPTY = 1.2e7       # kg·m²
IZZ_EMPTY = 1.0e6       # kg·m²

# Precomputed Tensors for interpolation
INERTIA_TENSOR_FULL = np.diag([IXX_FULL, IXX_FULL, IZZ_FULL])
INERTIA_TENSOR_EMPTY = np.diag([IXX_EMPTY, IXX_EMPTY, IZZ_EMPTY])

# Stage-1 booster recovery rigid-body model (post-separation, 40 m core only)
# Mass range: 60,000 kg (dry + reserve) -> 30,000 kg (dry)
STAGE1_RECOVERY_CG_FULL = 18.0    # m from booster base at 60 t
STAGE1_RECOVERY_CG_EMPTY = 22.0   # m from booster base at 30 t
STAGE1_RECOVERY_CP = 16.0         # m from booster base (tail-fin stabilized recovery)
STAGE1_RECOVERY_IXX_FULL = 8.0e6  # kg*m^2 at 60 t
STAGE1_RECOVERY_IXX_EMPTY = 4.0e6 # kg*m^2 at 30 t
STAGE1_RECOVERY_IZZ_FULL = 1.03e5 # kg*m^2 at 60 t
STAGE1_RECOVERY_IZZ_EMPTY = 5.15e4 # kg*m^2 at 30 t

# Structural Limits
MAX_DYNAMIC_PRESSURE = 35000.0  # Pa (Max-Q Limit)

# =============================================================================
# CONTROL PARAMETERS (Table 7)
# =============================================================================

# Guidance & Control Gains
# -----------------------------------------------------------------------------
# PD Controller Design per Document Section 17.6:
#
#   τ_cmd = -Kp · q_ev - Kd · ω_e
#
# where q_ev is the vector part of the error quaternion.
# For small angles: q_ev ≈ sin(θ/2)·axis ≈ (θ/2)·axis
# So the linearized torque is: τ ≈ -Kp·(θ/2)·axis - Kd·ω
# The linearized dynamics: I·θ̈ + Kd·θ̇ + (Kp/2)·θ = 0
#
# Design basis (second-order with q_ev error signal):
#   Natural frequency:  ωn = sqrt(Kp / (2·I))
#   Damping ratio:      ζ  = Kd / (2 · sqrt(Kp·I/2))
#
# For I_full = 5.36e7 kg·m² (full propellant, worst case):
#   Kp = 1.2e8  →  ωn = sqrt(1.2e8 / (2*5.36e7)) = 1.058 rad/s  (~1s response)
#   Kd = 2·ζ·sqrt(Kp·I/2) = 2·0.7·sqrt(1.2e8*5.36e7/2) = 7.94e7
#
# For I_empty = 1.2e7 kg·m² (MECO, dry vehicle):
#   ωn = sqrt(1.2e8 / (2*1.2e7)) = 2.236 rad/s  (faster response — good)
#   ζ  = 7.94e7 / (2·sqrt(1.2e8*1.2e7/2)) = 1.48  (overdamped — stable)
#
# System is critically-damped at launch and overdamped at MECO.

KP_ATTITUDE = 1.2e8   # Proportional gain (N·m) — acts on q_ev (Document §17.6)
KD_ATTITUDE = 7.94e7   # Derivative gain (N·m·s/rad) — ζ=0.7 at full mass

# Maximum control torque (N·m)
# Physical basis: 7.6 MN thrust × 3.7m diameter × ~0.7 gimbal leverage ≈ 2e7 N·m
# Increased to accommodate higher Kd without excessive saturation
MAX_TORQUE = 3.0e7

# Pitchover Maneuver Parameters
PITCHOVER_START_ALTITUDE = 0.0   # Start pitchover immediately at liftoff (m)
PITCHOVER_END_ALTITUDE = 2000.0    # Altitude to end pitch kick (m)
PITCHOVER_ANGLE = 10.0 * np.pi / 180.0  # 10 degrees kick (more aggressive pitchover)
PITCHOVER_AZIMUTH = 90.0 * np.pi / 180.0  # East (inertial +Y)
PITCHOVER_RAMP_DISTANCE = 100.0  # Altitude range for smooth ramp-in (m)

# Target pitch angle at end of Phase I (rad from vertical)
TARGET_PITCH_ANGLE = np.radians(45.0)

# Guidance Logic Constants
GRAVITY_TURN_START_ALTITUDE = 500.0  # m - start gravity turn earlier
GRAVITY_TURN_TRANSITION_RANGE = 20000.0  # m (shorter transition to make turn more aggressive)
MIN_VELOCITY_FOR_TURN = 50.0  # m/s

# =============================================================================
# SIMULATION PARAMETERS (Table 1)
# =============================================================================

# Time step (s) - Reduced for better energy conservation
DT = 0.005

# Maximum simulation time (s)
# Must be long enough for: S1 ascent (~132s) + S2 burn (~350s)
# + coast to transfer apogee (~2500s) + circularisation (~50s)
# = ~3032s.  Allow margin for longer coasts.
MAX_TIME = 4000.0

# Target MECO criteria (Phase I)
TARGET_ALTITUDE = 110000.0  # m
TARGET_SPEED = 2500.0       # m/s

# =============================================================================
# INITIAL CONDITIONS (A.8)
# =============================================================================

# Launch site position (inertial frame, m)
# Positioned at Earth's surface along +X axis
INITIAL_POSITION = np.array([R_EARTH, 0.0, 0.0])

# Earth Rotation Rate (rad/s)
EARTH_ROTATION_RATE = 7.2921159e-5

# Initial velocity (inertial frame, m/s)
# Includes tangential velocity due to Earth rotation: v = omega x r
# At equator (r = R_EARTH along X), v is along Y
INITIAL_VELOCITY = np.array([0.0, EARTH_ROTATION_RATE * R_EARTH, 0.0])

# Initial quaternion - must align body +Z with radial direction
# Vehicle at [R_earth, 0, 0], so radial is +X axis
# Need to rotate body +Z (inertial Z) to point along inertial +X
# This corresponds to a +90-degree rotation about +Y axis
INITIAL_QUATERNION = np.array([np.sqrt(2)/2, 0.0, np.sqrt(2)/2, 0.0])

# Initial angular velocity (body frame, rad/s)
INITIAL_OMEGA = np.array([0.0, 0.0, 0.0])

# Body frame reference vectors
BODY_Z_AXIS = np.array([0.0, 0.0, 1.0])  # Thrust direction in body frame

# =============================================================================
# ATMOSPHERE MODEL CONSTANTS (US Standard Atmosphere 1976)
# =============================================================================

# Sea level reference conditions
ATM_T0 = 288.15            # Sea level temperature (K)
ATM_P0 = 101325.0          # Sea level pressure (Pa)
ATM_RHO0 = RHO_0               # Alias for compatibility

# Lapse rates and layer boundaries
ATM_LAPSE_RATE = 0.0065    # Temperature lapse rate (K/m)
ATM_TROPOPAUSE = 11000.0   # Troposphere height (m)
ATM_T_STRATOSPHERE = 216.65  # Stratosphere temperature (K)

# Fallback values
ATM_SPEED_OF_SOUND_FALLBACK = 340.0  # Fallback speed of sound (m/s)

# =============================================================================
# PHYSICS CONSTANTS FOR HIGH FIDELITY MODEL
# =============================================================================

# Thermodynamics
GAMMA = 1.4      # Adiabatic index for air
R_GAS = 287.05   # Specific gas constant (J/(kg·K))

# Aerodynamics - Drag Coefficient vs Mach Number
# Simple look-up table for Cd(Mach)
MACH_BREAKPOINTS = np.array([0.0, 0.8, 1.05, 1.3, 2.0, 5.0, 10.0, 25.0])
CD_VALUES = np.array([0.42, 0.42, 0.75, 0.65, 0.50, 0.35, 0.25, 0.20])

# Lift slope vs Mach (per rad)
CL_ALPHA_VALUES = np.array([2.0, 2.0, 1.8, 1.6, 1.2, 0.8, 0.6, 0.4])

# Propulsion - Vacuum Isp (Stage 1)
ISP_VAC = 311.0  # S1 Vacuum specific impulse (s)
# ISP (Sea Level) is defined above as 282.0

# =============================================================================
# STAGE 2 PROPULSION PARAMETERS
# =============================================================================
# LOX/RP-1 vacuum-optimized upper stage engine (Merlin Vacuum class)
#
# Delta-V budget (doc Table 11, Table 12):
#   v_circular(400km) = sqrt(mu/(R_E+400km)) = 7672 m/s
#   v_at_separation ~= 2200 m/s (from S1 ascent gravity turn)
#   Required dv ~= 5472 m/s + ~900 m/s gravity losses = 6372 m/s
#   Available dv = 348 * 9.81 * ln(120000/8000) = 9253 m/s
#   Margin: ~2881 m/s (45%) — large margin allows for non-ideal steering losses
#
# Engine specs (doc Table 12):
#   Thrust: 950 kN (vacuum)
#   Isp: 348 s (vacuum, LOX/RP-1 with high-expansion nozzle)
#   Mass flow: 950000 / (348 * 9.81) = 278.2 kg/s
#   Burn time: 112000 / 278.2 = 402.6 s
#
STAGE2_THRUST = 950000.0       # N (950 kN vacuum thrust, doc Table 12)
STAGE2_ISP_VAC = 348.0         # s (LOX/RP-1 vacuum Isp, Merlin Vacuum class)
STAGE2_MASS_FLOW_RATE = STAGE2_THRUST / (STAGE2_ISP_VAC * G0)  # ~287.3 kg/s
STAGE2_BURN_TIME = STAGE2_PROPELLANT_MASS / STAGE2_MASS_FLOW_RATE  # ~358.6 s

# Target orbit
TARGET_ORBIT_ALTITUDE = 400000.0  # m (400 km LEO)

# =============================================================================
# NUMERICAL TOLERANCES
# =============================================================================

# Validation tolerances
QUATERNION_NORM_TOL = 1e-6  # Allowable deviation from unit norm
ENERGY_TOLERANCE = 1e-3     # Relative energy conservation tolerance

# General numerical tolerances
ZERO_TOLERANCE = 1e-10      # Near-zero check for divisions/normalizations
SMALL_VELOCITY_TOL = 1e-5   # Small velocity threshold (m/s)
DENSITY_FLOOR = 1e-12       # Minimum density before treating as vacuum (kg/m³)
CRASH_ALTITUDE_TOLERANCE = -1000.0  # Altitude below which we consider it a crash (m)
# Above this altitude the atmosphere is effectively vacuum (rho << DENSITY_FLOOR).
# Aerodynamic forces are zeroed here to avoid spurious near-zero drag/lift
# computations and for clarity — matches the top of the US76 + extension model.
AERO_DISABLE_ALTITUDE = 120000.0    # m (120 km)

# Winds (simplified log/power profile)
WIND_REF_ALT = 10000.0   # m
WIND_REF_SPEED = 30.0    # m/s at 10 km
WIND_EXPONENT = 0.2
WIND_DIRECTION_AZIMUTH = np.radians(270.0)  # from West to East

# =============================================================================
# PARAMETER SUMMARY (for logging)
# =============================================================================

# Control Limits [FIX #2, FIX #4] - Phase I Safety Constraints
MAX_PITCH_ANGLE = np.radians(75.0)  # Increased from 60° to allow more aggressive pitch maneuvers
MAX_GIMBAL_RATE = np.radians(25.0)   # Increased from 20°/s for better tracking response

# =============================================================================

def print_config():
    """Print configuration summary."""
    print("="*60)
    print("RLV Phase-I Configuration (PDF-Compliant)")
    print("="*60)
    print(f"Initial mass: {INITIAL_MASS:,.0f} kg")
    print(f"Dry mass (MECO): {DRY_MASS:,.0f} kg")
    print(f"Propellant: {PROPELLANT_MASS:,.0f} kg")
    print(f"Thrust: {THRUST_MAGNITUDE/1e6:.1f} MN")
    print(f"Isp: {ISP:.0f} s")
    print(f"Inertia Ixx/Iyy: {IXX_FULL:.2e} kg·m²")
    print(f"Inertia Izz: {IZZ_FULL:.2e} kg·m²")
    print(f"Kp: {KP_ATTITUDE:.1e}, Kd: {KD_ATTITUDE:.1e}")
    print(f"Max torque: {MAX_TORQUE:.1e} N·m")
    print(f"Max pitch angle: {np.degrees(MAX_PITCH_ANGLE):.1f} degrees")
    print(f"Max gimbal rate: {np.degrees(MAX_GIMBAL_RATE):.1f} degrees/second")
    print("="*60)

