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

# Attitude control PD gains
KP_ATTITUDE = 1.2e4  # Proportional gain
KD_ATTITUDE = 3.5e3  # Derivative gain

# Maximum control torque (N·m)
MAX_TORQUE = 2.5e6

# =============================================================================
# GUIDANCE PARAMETERS (Table 6) - Altitude-based gravity turn
# =============================================================================

# Gravity turn parameters (altitude-based per PDF)
GRAVITY_TURN_START_ALTITUDE = 1500.0  # m - when to start pitching over
GRAVITY_TURN_TRANSITION_RANGE = 25000.0  # m - altitude range for transition
MIN_VELOCITY_FOR_TURN = 150.0  # m/s - minimum velocity before starting turn

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

# Initial velocity (inertial frame, m/s)
# Starting at rest relative to Earth's surface
INITIAL_VELOCITY = np.array([0.0, 0.0, 0.0])

# Initial quaternion - must align body +Z with radial direction
# Vehicle at [R_earth, 0, 0], so radial is +X axis
# Need to rotate body +Z (inertial Z) to point along inertial +X
# Computed from rotation matrix: R @ [0,0,1] = [1,0,0]
# This corresponds to a 90-degree rotation about +Y axis
INITIAL_QUATERNION = np.array([np.sqrt(2)/2, 0.0, np.sqrt(2)/2, 0.0])

# Initial angular velocity (body frame, rad/s)
INITIAL_OMEGA = np.array([0.0, 0.0, 0.0])

# =============================================================================
# VALIDATION TOLERANCES
# =============================================================================

QUATERNION_NORM_TOL = 1e-6  # Allowable deviation from unit norm
ENERGY_TOLERANCE = 1e-3  # Relative energy conservation tolerance

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
