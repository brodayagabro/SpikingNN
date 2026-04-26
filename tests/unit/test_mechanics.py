"""
Unit tests for neuromechanical components.
"""

import numpy as np
import pytest
from spikingnn_core.mechanics import (
    SimpleAdaptedMuscle,
    Afferents, Simple_Afferents,
    Pendulum, OneDOFLimb, OneDOFLimb_withGR
)
from spikingnn_core.system import Afferented_Limb, Var_Limb


class TestSimpleAdaptedMuscle:
    """Tests for SimpleAdaptedMuscle model."""
    
    def test_initialization(self):
        """Test default parameter initialization."""
        muscle = SimpleAdaptedMuscle()
        assert muscle.w == 0.5
        assert muscle.A == 0.0074
        assert muscle.N == 10
        assert muscle.tau_c == 1/71  # Stored as inverse
        assert muscle.tau_1 == 1/130
        
    def test_set_params(self):
        """Test parameter updates."""
        muscle = SimpleAdaptedMuscle()
        muscle.set_params(w=1.0, A=0.01, tau_c=50)
        assert muscle.w == 1.0
        assert muscle.A == 0.01
        assert muscle.tau_c == 1/50  # Converted to inverse
        
    def test_step_activation(self):
        """Test activation dynamics with constant input."""
        muscle = SimpleAdaptedMuscle(w=1.0, tau_c=100)  # tau_c=100ms
        muscle.set_init_conditions()
        
        # Apply constant input
        for _ in range(100):
            muscle.step(dt=0.1, u=10.0)
        
        # Activation should approach steady state: Cn = w*u*tau_c
        expected_Cn = 1.0 * 10.0 * 100  # w * u * tau_c
        assert abs(muscle.Cn - expected_Cn) < 1.0  # Allow for integration error
        
    def test_force_saturation(self):
        """Test Hill-type activation curve saturation."""
        muscle = SimpleAdaptedMuscle(m=2.5, k=0.75)
        
        # Low activation -> low force
        muscle.Cn = 0.1
        x_low = muscle.Cn**muscle.m / (muscle.Cn**muscle.m + muscle.k**muscle.m)
        assert x_low < 0.1
        
        # High activation -> saturated force
        muscle.Cn = 10.0
        x_high = muscle.Cn**muscle.m / (muscle.Cn**muscle.m + muscle.k**muscle.m)
        assert x_high > 0.99


class TestAfferents:
    """Tests for afferent feedback models."""
    
    def test_ia_velocity_sensitivity(self):
        """Test Ia afferent velocity sensitivity."""
        aff = Afferents()
        
        # Zero velocity -> no velocity contribution
        ia_zero = aff.Ia(v=0, L=0.1, input=0)
        assert ia_zero < 1.0  # Only length term contributes
        
        # High velocity -> strong response
        ia_high = aff.Ia(v=1.0, L=0.1, input=0)  # 1 m/s
        assert ia_high > ia_zero
        
    def test_ib_force_threshold(self):
        """Test Ib afferent force threshold."""
        aff = Afferents()
        
        # Below threshold -> zero response
        ib_low = aff.Ib(F=2.0)  # F_th = 3.38 N
        assert ib_low == 0.0
        
        # Above threshold -> proportional response
        ib_high = aff.Ib(F=6.76)  # 2 * F_th
        assert ib_high == 1.0  # Normalized: (6.76-3.38)/3.38 = 1.0
        
    def test_simple_afferents_linearity(self):
        """Test simplified afferents return normalized linear responses."""
        aff = Simple_Afferents()
        
        # Ia: linear in velocity
        assert aff.Ia(v=0.059, L=0) == 1.0  # v = L_th
        assert aff.Ia(v=0.118, L=0) == 2.0  # v = 2*L_th
        
        # Ib: linear above threshold
        assert aff.Ib(F=3.38) == 0.0  # At threshold
        assert aff.Ib(F=6.76) == 1.0  # 2*threshold


class TestPendulum:
    """Tests for pendulum mechanics."""
    
    def test_natural_period(self):
        """Test calculation of natural oscillation period."""
        # T = 2π√(2ls/(3g))
        limb = Pendulum(ls=0.3)
        expected_T = 2 * np.pi * np.sqrt(2 * 0.3 / (3 * 9.81)) * 1000  # Convert to ms
        assert abs(limb.own_T - expected_T) < 1.0
        
    def test_gravity_restoring_torque(self):
        """Test that gravity provides restoring torque."""
        limb = Pendulum(q0=np.pi/2, b=0)  # Start horizontal, no friction
        limb.set_init_conditions()
        
        # Should accelerate downward due to gravity
        for _ in range(10):
            limb.step(dt=0.1, M=0)
        
        # Angle should decrease from π/2 toward 0 (downward)
        assert limb.q < np.pi/2
        # Angular velocity should be negative (clockwise)
        assert limb.w < 0
        
    def test_friction_damping(self):
        """Test angular friction damps motion."""
        limb = Pendulum(b=0.01, q0=np.pi/4, w0=10)  # Initial velocity
        limb.set_init_conditions()
        
        initial_w = limb.w
        for _ in range(100):
            limb.step(dt=0.1, M=0)
        
        # Velocity should decrease due to friction
        assert abs(limb.w) < abs(initial_w)


class TestOneDOFLimb:
    """Tests for limb with muscle actuation."""
    
    def test_muscle_length_geometry(self):
        """Test muscle length calculation from joint angle."""
        limb = OneDOFLimb(a1=0.06, a2=0.007)
        
        # At q=0, muscles should have minimum length
        L_min = limb.L(0)
        expected_min = np.sqrt(0.06**2 + 0.007**2 - 2*0.06*0.007)
        assert abs(L_min - expected_min) < 1e-6
        
        # At q=π, muscles should have maximum length
        L_max = limb.L(np.pi)
        expected_max = np.sqrt(0.06**2 + 0.007**2 + 2*0.06*0.007)
        assert abs(L_max - expected_max) < 1e-6
        
    def test_moment_arm_calculation(self):
        """Test moment arm (lever arm) calculation."""
        limb = OneDOFLimb(a1=0.06, a2=0.007, q0=np.pi/2)
        L = limb.L(np.pi/2)
        h = limb.h(L, np.pi/2)
        
        # At q=π/2, sin(q)=1, so h = a1*a2/L
        expected_h = 0.06 * 0.007 / L
        assert abs(h - expected_h) < 1e-6
        
    def test_force_to_torque_conversion(self):
        """Test conversion of muscle forces to joint torque."""
        limb = OneDOFLimb(q0=np.pi/2)
        
        # Equal flexor/extensor forces -> zero net torque
        limb.step(dt=0.1, F_flex=1.0, F_ext=1.0)
        # Torque should be small (numerical precision)
        assert abs(limb.M_tot) < 0.01
        
        # Flexor > extensor -> positive torque (flexion)
        limb.step(dt=0.1, F_flex=2.0, F_ext=1.0)
        assert limb.M_tot > 0


class TestAfferentedLimb:
    """Tests for integrated afferented limb."""
    
    def test_afferent_output_shape(self):
        """Test that afferent output has correct shape."""
        al = Afferented_Limb()
        al.set_init_conditions()
        al.step(dt=0.1, uf=5.0, ue=3.0)
        
        assert al.output.shape == (6,)
        # All values should be non-negative
        assert np.all(al.output >= 0)
        
    def test_flexor_extensor_independence(self):
        """Test that flexor/extensor afferents respond independently."""
        al = Afferented_Limb()
        al.set_init_conditions()
        
        # Only activate flexor
        al.step(dt=0.1, uf=10.0, ue=0)
        flexor_afferents = al.output[:3].copy()
        
        # Only activate extensor
        al.set_init_conditions()
        al.step(dt=0.1, uf=0, ue=10.0)
        extensor_afferents = al.output[3:].copy()
        
        # Flexor activation should primarily affect first 3 outputs
        assert np.mean(flexor_afferents) > np.mean(extensor_afferents[:3])
        # Extensor activation should primarily affect last 3 outputs
        assert np.mean(extensor_afferents) > np.mean(flexor_afferents[3:])


class TestVarLimb:
    """Tests for integrated neural-mechanical system."""
    
    def test_named_access(self):
        """Test named access to components."""
        from spikingnn_core import Izhikevich_IO_Network
        
        net = Izhikevich_IO_Network(N=4, names=['N1', 'N2', 'N3', 'N4'])
        al = Afferented_Limb()
        sys = Var_Limb(Network=net, Limb=al)
        
        # Test neuron names
        assert 'N1' in sys.names['neurons']
        
        # Test muscle names
        assert 'Flexor' in sys.names['muscles']
        
        # Test afferent names
        assert 'Ia_Flex' in sys.names['afferents']
        
    def test_set_weights_by_names(self):
        """Test setting weights by neuron names."""
        from spikingnn_core import Izhikevich_IO_Network
        
        net = Izhikevich_IO_Network(N=2, names=['A', 'B'])
        net.M[1, 0] = 1  # Create connection A→B
        al = Afferented_Limb()
        sys = Var_Limb(Network=net, Limb=al)
        
        # Set weight by name
        result = sys.set_weights_by_names('A', 'B', new_weight=2.5)
        assert result == True
        assert sys.net.W[1, 0] == 2.5
        
        # Try non-existent names
        result = sys.set_weights_by_names('X', 'Y', new_weight=1.0)
        assert result == False
        
    def test_closed_loop_step(self):
        """Test one step of closed-loop simulation."""
        from spikingnn_core import Izhikevich_IO_Network
        
        net = Izhikevich_IO_Network(
            input_size=2, output_size=2, afferent_size=6, N=4
        )
        al = Afferented_Limb()
        sys = Var_Limb(Network=net, Limb=al)
        
        # Initial step should not raise errors
        Iapp = np.array([1.0, 1.0])
        sys.step(dt=0.1, Iapp=Iapp)
        
        # Check that states have changed from initial conditions
        assert not np.allclose(sys.V, sys.net.c)  # Potentials should evolve
        assert sys.F_flex >= 0  # Forces should be non-negative
        assert sys.F_ext >= 0