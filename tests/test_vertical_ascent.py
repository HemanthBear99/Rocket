
import numpy as np

from rlv_sim import constants as C
from rlv_sim.integrators import integrate
from rlv_sim.state import create_initial_state

def test_vertical_ascent_tsiolkovsky():
    """
    PRIORITY 4: VALIDATE VERTICAL ASCENT
    Tests if the rocket simulation matches the ideal Tsiolkovsky rocket equation
    in a vacuum/gravity-only vertical ascent scenario.
    """
    
    # Setup
    state = create_initial_state()
    # Position: High altitude to minimize drag (vacuum approx)
    state.r = np.array([C.R_EARTH + 300000.0, 0.0, 0.0]) # 300km (basically vacuum)
    state.v = np.array([0.0, 0.0, 0.0])
    state.m = C.INITIAL_MASS
    
    dt = 0.1
    duration = 20.0 # Short burn for test to avoid large gravity turn deviations
    
    # Run Simulation loop
    steps = int(duration / dt)
    
    # We use a simplified integration loop that doesn't use the full 'simulation_step'
    # because that would induce guidance/turn. We just want to test THRUST vs MASS.
    
    # Actually, we should test the INTEGRATOR + FORCES, not the guidance.
    # The integrate function handles forces.
    
    for _ in range(steps):
        # Apply pure vertical torque (none)
        torque = np.zeros(3)
        
        # Manually integrate
        # We assume guidance asks for vertical thrust (which it naturally does at v=0 if we don't pitch)
        # But wait, at 300km, density is 0, so drag is 0.
        
        # Helper to force vertical orientation
        # If we start upright (identity quaternion usually means Body aligned with ECI or similar)
        # We need to ensure logic holds.
        pass
        
        # We use the actual integrate function
        # Force throttle=1.0, thrust_on=True
        state = integrate(state, torque, dt, thrust_on=True, throttle=1.0)

    # Theoretical Delta-V (Tsiolkovsky)
    # dV = Isp * g0 * ln(m_initial / m_final)
    # However, this is in gravity field.
    # v_actual ≈ dV - g_avg * t
    
    # Mass consumed
    m_final = state.m
    m_consumed = C.INITIAL_MASS - m_final
    
    # We should verify mass consumption rate too
    # m_dot = F_vac / (Isp * g0)
    # Expected m_final
    
    # Use constant mass flow rate defined in constants
    expected_mass_loss = C.MASS_FLOW_RATE * duration
    print(f"Mass Consumed: {m_consumed:.2f} kg")
    print(f"Expected Loss: {expected_mass_loss:.2f} kg")
    # Allow small error due to discrete steps
    assert abs(m_consumed - expected_mass_loss) < (expected_mass_loss * 0.01), f"Mass error: {m_consumed} vs {expected_mass_loss}"
    
    # Velocity Check
    v_exhaust = C.ISP_VAC * C.G0
    delta_v_ideal = v_exhaust * np.log(C.INITIAL_MASS / state.m)
    
    # Gravity loss
    # g at 300km
    r_avg = C.R_EARTH + 300000.0
    g_avg = C.MU_EARTH / (r_avg**2)
    v_loss_gravity = g_avg * duration
    
    v_expected = delta_v_ideal - v_loss_gravity
    v_actual = state.speed
    
    print(f"Expected V: {v_expected:.2f} m/s")
    print(f"Actual V:   {v_actual:.2f} m/s")
    
    # 5% tolerance is generous but appropriate for numerical integration vs analytical approx
    error_pct = abs(v_actual - v_expected) / v_expected
    assert error_pct < 0.05, f"Velocity error {error_pct*100:.2f}% exceeds 5%"

if __name__ == "__main__":
    test_vertical_ascent_tsiolkovsky()
