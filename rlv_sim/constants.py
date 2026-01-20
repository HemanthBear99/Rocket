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

# Stage 2 Mass Properties (Table in A.11)
STAGE2_MASS = 120000.0  # kg (constant during Phase I)

# Stacked Vehicle Mass (A.13)
DRY_MASS = STAGE1_DRY_MASS + STAGE2_MASS  # 150,000 kg at MECO
PROPELLANT_MASS = STAGE1_PROPELLANT_MASS  # 390,000 kg
INITIAL_MASS = STAGE1_WET_MASS + STAGE2_MASS  # 540,000 kg total

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

# =============================================================================
# INERTIA TENSOR - STACKED CONFIGURATION (A.14)
# Computed from Stage 1 + Stage 2 using parallel axis theorem
# =============================================================================

# Stage 1 Inertia Tensor (A.3)
I1_xx = 1.20e7  # kg·m²
I1_yy = 1.20e7  # kg·m²
I1_zz = 2.00e6  # kg·m²

# Stage 2 Inertia Tensor (A.12)
I2_xx = 2.5e6  # kg·m²
I2_yy = 2.5e6  # kg·m²
I2_zz = 4.0e5  # kg·m²

# Stage 2 CG offset from Stage 1 CG (A.12)
STAGE2_CG_OFFSET = 18.0  # m along +Z

# Stacked Inertia using parallel axis theorem (A.14)
# I_total = I1 + I2 + m2 * (d^2 * I3 - d*d^T)
# where d is the offset vector [0, 0, 18]
Ixx = I1_xx + I2_xx + STAGE2_MASS * (STAGE2_CG_OFFSET**2)  # Parallel axis for xx
Iyy = I1_yy + I2_yy + STAGE2_MASS * (STAGE2_CG_OFFSET**2)  # Parallel axis for yy
Izz = I1_zz + I2_zz  # No offset contribution for zz (offset along z)

# Inertia tensor (diagonal for symmetric vehicle)
INERTIA_TENSOR = np.array([
    [Ixx, 0.0, 0.0],
    [0.0, Iyy, 0.0],
    [0.0, 0.0, Izz]
])

# Inverse inertia tensor (precomputed for efficiency)
INERTIA_TENSOR_INV = np.linalg.inv(INERTIA_TENSOR)

# =============================================================================
# CONTROL PARAMETERS (Table 7)
# =============================================================================

# Guidance & Control Gains
# -----------------------------------------------------------------------------
KP_ATTITUDE = 1.2e4  # Proportional gain
KD_ATTITUDE = 3.5e3  # Derivative gain

# Maximum control torque (N·m)
MAX_TORQUE = 2.5e6

# Guidance Logic Constants
GRAVITY_TURN_START_ALTITUDE = 1500.0  # m
GRAVITY_TURN_TRANSITION_RANGE = 4000.0  # m
MIN_VELOCITY_FOR_TURN = 50.0  # m/s

# Pitchover Maneuver (Deterministic Azimuth)
PITCHOVER_START_ALTITUDE = 100.0   # Altitude to start pitch kick (m)
PITCHOVER_END_ALTITUDE = 1000.0    # Altitude to end pitch kick (m)
PITCHOVER_ANGLE = 2.0 * np.pi / 180.0  # 2 degrees kick
PITCHOVER_AZIMUTH = 90.0 * np.pi / 180.0 # East (inertial +Y)

# Target pitch angle at end of Phase I (rad from vertical)
TARGET_PITCH_ANGLE = np.radians(45.0)

# =============================================================================
# SIMULATION PARAMETERS (Table 1)
# =============================================================================

# Time step (s)
DT = 0.01

# Maximum simulation time (s)
MAX_TIME = 300.0

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
# Computed from rotation matrix: R @ [0,0,1] = [1,0,0]
# This corresponds to a 90-degree rotation about +Y axis
INITIAL_QUATERNION = np.array([np.sqrt(2)/2, 0.0, np.sqrt(2)/2, 0.0])

# Initial angular velocity (body frame, rad/s)
INITIAL_OMEGA = np.array([0.0, 0.0, 0.0])

# =============================================================================
# ATMOSPHERE MODEL CONSTANTS (US Standard Atmosphere 1976)
# =============================================================================

# Sea level reference conditions
ATM_T0 = 288.15            # Sea level temperature (K)
ATM_P0 = 101325.0          # Sea level pressure (Pa)
ATM_RHO0 = 1.225           # Sea level density (kg/m³) - alias for RHO_0

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

# Propulsion - Vacuum Isp
ISP_VAC = 311.0  # Vacuum specific impulse (s)
# ISP (Sea Level) is defined above as 282.0

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

# =============================================================================
# PARAMETER SUMMARY (for logging)
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
    print(f"Inertia Ixx/Iyy: {Ixx:.2e} kg·m²")
    print(f"Inertia Izz: {Izz:.2e} kg·m²")
    print(f"Kp: {KP_ATTITUDE:.1e}, Kd: {KD_ATTITUDE:.1e}")
    print(f"Max torque: {MAX_TORQUE:.1e} N·m")
    print("="*60)

