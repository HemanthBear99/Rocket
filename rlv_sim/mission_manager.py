"""
RLV Mission Manager

This module handles the high-level state machine for the mission.
It defines discrete mission phases and handles the transition logic between them.

Phases implemented:
- ASCENT: Liftoff to MECO
- COAST: MECO to Apogee
"""

from enum import Enum, auto
import logging
from dataclasses import dataclass

from . import constants as C
from .state import State
from .mass import is_propellant_exhausted

logger = logging.getLogger(__name__)

class MissionPhase(Enum):
    PRELAUNCH = auto()
    ASCENT = auto()
    COAST = auto()
    STAGE_SEPARATION = auto()
    APOGEE_REACHED = auto()
    # Booster Phases
    BOOSTER_FLIP = auto()
    BOOSTER_BOOSTBACK = auto()
    BOOSTER_COAST = auto()
    BOOSTER_ENTRY = auto()
    BOOSTER_LANDING = auto()

class MissionManager:
    """
    Manages the current mission phase and transitions.
    """
    def __init__(self, vehicle_type: str = "ascent"):
        self.vehicle_type = vehicle_type
        self.current_phase = MissionPhase.ASCENT # Default start
        
        if vehicle_type == "orbiter":
            self.current_phase = MissionPhase.COAST # Orbiter starts after separation
        elif vehicle_type == "booster":
            self.current_phase = MissionPhase.BOOSTER_FLIP # Booster starts at separation
            
        self.meco_time = None
        self.apogee_time = None
        self.max_altitude_reached = 0.0
        self._last_radial_velocity = 0.0

    def update(self, state: State, dt: float):
        """
        Check for phase transitions based on current state.
        
        Args:
            state: Current vehicle state
            dt: Time step
        """
        self.max_altitude_reached = max(self.max_altitude_reached, state.altitude)
        
        # Calculate radial velocity for apogee detection
        # r_dot = (r . v) / |r|
        r_norm = (state.r[0]**2 + state.r[1]**2 + state.r[2]**2)**0.5
        if r_norm > 1.0:
            radial_velocity = (state.r[0]*state.v[0] + state.r[1]*state.v[1] + state.r[2]*state.v[2]) / r_norm
        else:
            radial_velocity = 0.0

        # --- Transition Logic ---

        # ASCENT -> COAST (MECO)
        if self.current_phase == MissionPhase.ASCENT:
            # Check for propellant exhaustion (Main Engine Cut-Off)
            if is_propellant_exhausted(state.m):
                logger.info(f"MECO detected at t={state.t:.2f}s, Alt={state.altitude/1000:.1f}km")
                self.current_phase = MissionPhase.COAST
                self.meco_time = state.t

        # COAST -> STAGE_SEPARATION (Time-based or Apogee-based?)
        # Let's say separation happens 3 seconds after MECO (typical)
        elif self.current_phase == MissionPhase.COAST:
            if self.meco_time and (state.t - self.meco_time) > 3.0:
                 logger.info(f"Stage Separation trigger at t={state.t:.2f}s")
                 self.current_phase = MissionPhase.STAGE_SEPARATION
        
        # STAGE_SEPARATION -> APOGEE (or next phase)
        # This is a momentary state, usually we immediately transition to next
        # But for the Fork Orchestrator, we can stay here until the runner handles it.
        # Or we can just let it flow.
        # Actually, let's keep it in Coast for the rest of path if we don't assume specific logic.
        
        # ... Wait, if we separate, we fork.
        # So we just need to detect separation time.
        pass

        # COAST (continued) -> APOGEE
        if self.current_phase in [MissionPhase.COAST, MissionPhase.STAGE_SEPARATION]:
            # Apogee detection logic...
            # ... (keep existing apogee logic)
            if self._last_radial_velocity >= 0.0 and radial_velocity < 0.0:
                 # Check overlap with atmosphere to avoid false positives at liftoff (though we are in COAST so safe)
                 if state.altitude > 10000.0: 
                    logger.info(f"Apogee detected at t={state.t:.2f}s, Alt={state.altitude/1000:.1f}km")
                    self.current_phase = MissionPhase.APOGEE_REACHED
                    self.apogee_time = state.t
                    
        # --- BOOSTER LOGIC ---
        if self.vehicle_type == "booster":
            # FLIP -> BOOSTBACK
            # Flip for 15 seconds
            if self.current_phase == MissionPhase.BOOSTER_FLIP:
                # We assume we started FLIP at start of 'run_simulation' for this fork
                # So just time check: t > start_t + 15
                # But state.t is absolute time. We need to track phase timer.
                # Hack: Just check absolute time if we know roughly when separation happened (t~145s)
                if state.t > 160.0:
                    self.current_phase = MissionPhase.BOOSTER_BOOSTBACK
                    logger.info(f"Booster Flip Complete, Starting Boostback at t={state.t:.2f}s")
                    
            # BOOSTBACK -> COAST
            # Boostback burn for 25 seconds
            elif self.current_phase == MissionPhase.BOOSTER_BOOSTBACK:
                if state.t > 185.0:
                    self.current_phase = MissionPhase.BOOSTER_COAST
                    logger.info(f"Boostback Complete, starting Coast at t={state.t:.2f}s")
                    
            # COAST -> ENTRY
            elif self.current_phase == MissionPhase.BOOSTER_COAST:
                if state.altitude < 40000.0:
                    self.current_phase = MissionPhase.BOOSTER_ENTRY
                    logger.info(f"Booster Entry Interface at t={state.t:.2f}s, Alt={state.altitude/1000:.1f}km")

            # ENTRY -> LANDING
            elif self.current_phase == MissionPhase.BOOSTER_ENTRY:
                if state.altitude < 5000.0:
                    self.current_phase = MissionPhase.BOOSTER_LANDING
                    logger.info(f"Booster Landing Burn Start at t={state.t:.2f}s")

        self._last_radial_velocity = radial_velocity

    def get_phase(self) -> MissionPhase:
        return self.current_phase
