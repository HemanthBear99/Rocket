"""
RLV Phase-I Ascent Simulation Package

A physically correct, numerically stable Python simulation of a
Reusable Launch Vehicle (RLV) Phase-I ascent.

Modules:
    - constants: Physical constants and vehicle parameters
    - state: Global state dataclass
    - frames: Quaternion operations and frame transformations
    - forces: Force computations (gravity, thrust, drag)
    - mass: Mass flow calculations
    - dynamics: Rotational and translational dynamics
    - guidance: Ascent guidance law
    - control: Attitude control system
    - integrators: RK4 numerical integration
    - validation: Physics validation checks
    - main: Simulation entry point
"""

from .state import State, create_initial_state
from .main import run_simulation, run_full_mission, FullMissionResult, SimulationLog
from .guidance import GuidanceState, create_guidance_state
from .config import SimulationConfig, create_default_config, create_test_config

__version__ = "1.1.0"
__author__ = "RLV Simulation Team"

__all__ = [
    'State',
    'create_initial_state',
    'run_simulation',
    'run_full_mission',
    'FullMissionResult',
    'SimulationLog',
    'GuidanceState',
    'create_guidance_state',
    'SimulationConfig',
    'create_default_config',
    'create_test_config',
]
