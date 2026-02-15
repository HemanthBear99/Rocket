import pytest
from rlv_sim import main
from rlv_sim.mission_manager import MissionManager

def test_run_simulation_completes():
    # Run a very short simulation for test speed
    state, log, reason = main.run_simulation(dt=0.1, max_time=1.0, verbose=False)
    # Allow small floating-point tolerance on time accumulation
    assert state.t <= 1.0 + 0.15  # One extra timestep tolerance
    assert hasattr(log, 'time')
    assert isinstance(reason, str)
    assert len(log.time) > 0

def test_check_termination_conditions():
    import numpy as np
    from rlv_sim.state import State
    mission_mgr = MissionManager(vehicle_type="ascent")
    # Give an upward radial velocity so apogee check doesn't trigger
    # (v_vert > 0 means still ascending, so max_time check fires first)
    v_ascending = np.array([500.0, 465.0, 0.0])  # positive radial + tangential
    s2 = State(m=main.C.INITIAL_MASS, r=main.C.INITIAL_POSITION.copy(),
               v=v_ascending, q=main.C.INITIAL_QUATERNION.copy(),
               omega=main.C.INITIAL_OMEGA.copy(), t=200.0)
    term, reason = main.check_termination(s2, max_time=100.0, mission_mgr=mission_mgr)
    assert term is True
    assert 'Maximum simulation time' in reason

def test_simulation_step_pipeline():
    from rlv_sim.state import State
    from rlv_sim.actuator import ActuatorState
    mission_mgr = MissionManager(vehicle_type="ascent")
    s = State(r=main.C.INITIAL_POSITION.copy(), v=main.C.INITIAL_VELOCITY.copy(), q=main.C.INITIAL_QUATERNION.copy(), omega=main.C.INITIAL_OMEGA.copy(), m=main.C.INITIAL_MASS, t=0.0)
    a = ActuatorState(thrust_dir=main.C.INITIAL_POSITION.copy()/main.C.R_EARTH)
    s2, guidance, control, a2, gs = main.simulation_step(s, a, mission_mgr, dt=0.1)
    assert hasattr(s2, 'r')
    assert isinstance(guidance, dict)
    assert isinstance(control, dict)
    assert hasattr(a2, 'thrust_dir')
    from rlv_sim.guidance import GuidanceState
    assert isinstance(gs, GuidanceState)
