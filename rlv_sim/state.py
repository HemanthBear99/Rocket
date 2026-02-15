"""
RLV Phase-I Ascent Simulation - Global State Vector

This module defines the single global state dataclass that contains all
simulation state variables. No duplicated state is allowed anywhere.
"""

from dataclasses import dataclass, field
import numpy as np

from . import constants as C


@dataclass
class State:
    """
    Global state vector for the RLV simulation.
    
    All state variables are stored in a single location to ensure consistency
    and prevent duplication.
    
    Attributes:
        r: Position vector in inertial frame (m) [3]
        v: Velocity vector in inertial frame (m/s) [3]
        q: Orientation quaternion [w, x, y, z] (unit quaternion)
        omega: Angular velocity in body frame (rad/s) [3]
        m: Total vehicle mass (kg)
        t: Simulation time (s)
    """
    
    # Position in inertial frame (m)
    r: np.ndarray = field(default_factory=lambda: np.zeros(3))
    
    # Velocity in inertial frame (m/s)
    v: np.ndarray = field(default_factory=lambda: np.zeros(3))
    
    # Orientation quaternion [w, x, y, z]
    q: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0]))
    
    # Angular velocity in body frame (rad/s)
    omega: np.ndarray = field(default_factory=lambda: np.zeros(3))
    
    # Total mass (kg)
    m: float = 0.0
    
    # Simulation time (s)
    t: float = 0.0
    
    def __post_init__(self):
        """Ensure arrays are numpy arrays with correct dtype."""
        for attr in ['r', 'v', 'q', 'omega']:
            setattr(self, attr, np.asarray(getattr(self, attr), dtype=np.float64))
    
    def copy(self) -> 'State':
        """Create a deep copy of the state."""
        return State(
            r=self.r.copy(),
            v=self.v.copy(),
            q=self.q.copy(),
            omega=self.omega.copy(),
            m=self.m,
            t=self.t
        )
    
    def to_vector(self) -> np.ndarray:
        """Convert state to a flat numpy array check [r, v, q, omega, m]."""
        return np.concatenate([
            self.r, self.v, self.q, self.omega, [self.m]
        ])
    
    @classmethod
    def from_vector(cls, vec: np.ndarray, t: float) -> 'State':
        """
        Create a State from a flat numpy array.
        
        Args:
            vec: State vector [r(3), v(3), q(4), omega(3), m(1)]
            t: Current simulation time
        """
        return cls(
            r=vec[0:3].copy(),
            v=vec[3:6].copy(),
            q=vec[6:10].copy(),
            omega=vec[10:13].copy(),
            m=vec[13],
            t=t
        )
    
    @property
    def altitude(self) -> float:
        """Altitude above Earth's surface (m)."""
        return np.linalg.norm(self.r) - C.R_EARTH
    
    @property
    def speed(self) -> float:
        """Magnitude of velocity (m/s)."""
        return np.linalg.norm(self.v)
    
    @property
    def propellant_remaining(self) -> float:
        """Remaining propellant mass (kg)."""
        return max(0.0, self.m - C.DRY_MASS)
    
    def __str__(self) -> str:
        """Human-readable state summary."""
        return (
            f"State(t={self.t:.2f}s, "
            f"alt={self.altitude/1000:.2f}km, "
            f"v={self.speed:.1f}m/s, "
            f"m={self.m:.1f}kg)"
        )


def create_initial_state() -> State:
    """
    Create the initial state for the simulation.
    
    Returns:
        State object initialized with launch conditions.
    """
    return State(
        r=C.INITIAL_POSITION.copy(),
        v=C.INITIAL_VELOCITY.copy(),
        q=C.INITIAL_QUATERNION.copy(),
        omega=C.INITIAL_OMEGA.copy(),
        m=C.INITIAL_MASS,
        t=0.0
    )
