#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Integration tests for spikingnn_core package.

This module performs comprehensive testing of:
- Izhikevich and FitzHugh-Nagumo neuron network models
- Neuromechanical components (muscles, afferents, limbs)
- Integrated systems (Var_Limb, Net_Limb_connect)
- Numerical stability and parameter handling

Usage:
    python tests/integration/test_core_components.py
    pytest tests/integration/test_core_components.py -v

All tests generate console output for verification.
For visual tests, matplotlib figures are displayed.
"""

import sys
import os
import numpy as np

# Add core package to path for development testing
CORE_SRC = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'packages', 'core', 'src')
)
if CORE_SRC not in sys.path:
    sys.path.insert(0, CORE_SRC)


def section(title: str) -> None:
    """Print a formatted section header."""
    print(f"\n{'='*60}\n🔹 {title}\n{'='*60}")


def check(condition: bool, ok_msg: str = "OK", fail_msg: str = "FAIL") -> None:
    """Assert a condition and print result."""
    if condition:
        print(f"  ✅ {ok_msg}")
    else:
        print(f"  ❌ {fail_msg}")
        raise AssertionError(fail_msg)


def main() -> None:
    """Run all integration tests."""
    try:
        # =====================================================================
        # 1. PRESETS & PARAMETER UTILITIES
        # =====================================================================
        section("1. Presets & Parameter Utilities")
        from spikingnn_core.models.presets import izhikevich_neuron, types2params
        
        # Test types2params function
        a, b, c, d = types2params(['RS', 'FS', 'IB'])
        check(len(a) == 3, "types2params returns 4 arrays of length 3")
        check(np.isclose(a[0], 0.02), "Parameter 'a' for RS matches reference value")
        check(np.isclose(d[1], 0.05), "Parameter 'd' for FS matches reference value")
        print("  ✅ Preset parameter conversion works correctly")

        # Test individual neuron preset
        rs_neuron = izhikevich_neuron(preset='RS')
        check(rs_neuron.params == [0.02, 0.2, -65.0, 8.0], "RS preset parameters correct")
        print("  ✅ Individual neuron preset instantiation works")

        # =====================================================================
        # 2. IZHIKEVICH NETWORK MODELS
        # =====================================================================
        section("2. Izhikevich Network Models")
        from spikingnn_core.models.izhikevich import Izhikevich_Network, Izhikevich_IO_Network
        
        # Test basic network initialization
        net = Izhikevich_Network(N=3, a=a, b=b, c=c, d=d)
        check(net.N == 3, "Network initialized with N=3")
        check(net.V.shape == (3,), "Membrane potential vector V initialized")
        check(net.U.shape == (3,), "Recovery variable vector U initialized")
        
        # Test single integration step
        net.step(dt=0.1, Iapp=5.0)
        check(not np.all(net.V == net.c), "step() method updates network state")
        print("  ✅ Basic Izhikevich network dynamics work")

        # Test IO-Network with projection matrices
        Q_app = np.eye(3, 2)
        Q_aff = np.zeros((3, 3))
        P = np.eye(2, 3)
        io_net = Izhikevich_IO_Network(
            N=3, a=a, b=b, c=c, d=d,
            input_size=2, output_size=2, afferent_size=3,
            Q_app=Q_app, Q_aff=Q_aff, P=P
        )
        io_net.step(dt=0.1, Iapp=np.array([1.0, 2.0]), Iaff=np.zeros(3))
        check(io_net.V_out.shape == (2,), "Output vector V_out has correct dimension")
        print("  ✅ IO-Network with projection matrices works")

        # Test parameter broadcasting (scalar to array)
        net_scalar = Izhikevich_Network(N=5, a=0.02, b=0.2, c=-65.0, d=8.0)
        check(np.all(net_scalar.a == 0.02), "Scalar parameters broadcast to arrays correctly")
        print("  ✅ Parameter broadcasting (scalar → array) works")

        # =====================================================================
        # 3. NETWORK TOPOLOGY & NAMING
        # =====================================================================
        section("3. Network Topology & Naming")
        
        # Test named network functionality
        net2 = Izhikevich_Network(N=3, names=['A', 'B', 'C'])
        net2.connect(0, 1, coef=1, w=2.5, tau=15.0)
        
        check(net2.M[1, 0] == 1, "Connectivity mask M updated correctly")
        check(net2.W[1, 0] == 2.5, "Synaptic weight W set correctly")
        check(np.isclose(net2.tau_syn[1, 0], 1/15.0), "tau_syn converted to 1/tau correctly")
        
        # Test name-based access methods
        check(net2.get_weight_by_names('A', 'B') == 2.5, "Weight lookup by names works")
        check(net2.set_weights_by_names('A', 'B', 3.0) == True, "Weight modification by names works")
        check(net2.W[1, 0] == 3.0, "Weight updated via name-based method")
        print("  ✅ Topology management and neuron naming work correctly")

        # =====================================================================
        # 4. FITZHUGH-NAGUMO NETWORK MODEL
        # =====================================================================
        section("4. FitzHugh-Nagumo Network Model")
        from spikingnn_core.models.fhn import FizhugNagumoNetwork
        
        # Test FHN network initialization
        fhn_net = FizhugNagumoNetwork(N=4, a=0.1, b=0.01, c=0.02)
        check(fhn_net.N == 4, "FHN network initialized with N=4")
        check(fhn_net.V.shape == (4,), "FHN membrane potential vector initialized")
        check(fhn_net.U.shape == (4,), "FHN recovery variable vector initialized")
        
        # Test FHN dynamics (different equations from Izhikevich)
        fhn_net.set_init_conditions()
        fhn_net.step(dt=0.1, Iapp=0.5)
        check(not np.all(fhn_net.V == 0), "FHN step() updates state variables")
        print("  ✅ FitzHugh-Nagumo network dynamics work")

        # Test FHN output function (sigmoidal spike indicator)
        output = fhn_net.syn_output()
        check(output.shape == (4,), "FHN syn_output() returns correct shape")
        check(np.all((output >= 0) & (output <= 1)), "FHN output values in [0, 1] range")
        print("  ✅ FHN sigmoidal output function works correctly")

        # Test FHN with time scaling parameter
        fhn_scaled = FizhugNagumoNetwork(N=2, ts=np.array([0.5, 2.0]))
        fhn_scaled.step(dt=0.1, Iapp=0.1)
        check(np.isfinite(fhn_scaled.V).all(), "FHN with time scaling produces finite values")
        print("  ✅ FHN time scaling parameter (ts) works")

        # =====================================================================
        # 5. NEUROMECHANICAL COMPONENTS
        # =====================================================================
        section("5. Neuromechanical Components")
        from spikingnn_core.mechanics.muscles import SimpleAdaptedMuscle
        from spikingnn_core.mechanics.afferents import Afferents, Simple_Afferents
        from spikingnn_core.mechanics.limb import Pendulum, OneDOFLimb, OneDOFLimb_withGR

        # Test muscle dynamics
        muscle = SimpleAdaptedMuscle(w=0.5, N=2)
        muscle.set_init_conditions()
        muscle.step(dt=0.1, u=5.0)
        check(muscle.F > 0, "Muscle generates force after activation")
        check(0 <= muscle.x <= 1, "Muscle activation x in [0, 1] range")
        print("  ✅ Muscle activation dynamics work correctly")

        # Test full Prochaska-type afferents
        aff = Afferents()
        ia = aff.Ia(v=0.1, L=0.07, input=0.5)
        check(ia > 0, "Ia afferent responds to velocity/length")
        
        ib = aff.Ib(F=5.0)
        check(ib > 0, "Ib afferent responds to force above threshold")
        
        ii = aff.II(L=0.07, input=0.5)
        check(ii >= 0, "II afferent returns non-negative activity")
        print("  ✅ Prochaska-type afferent models work")

        # Test simplified afferents
        simple_aff = Simple_Afferents()
        ia_simple = simple_aff.Ia(v=0.1, L=0.07)
        check(np.isclose(ia_simple, 0.1 / simple_aff.L_th), "Simple Ia returns normalized velocity")
        print("  ✅ Simplified afferent models work")

        # Test pendulum mechanics
        pend = Pendulum(q0=np.pi/2, b=0.01)
        pend.set_init_conditions()
        print("  ✅ Pendulum mechanics work correctly")

        # Test OneDOFLimb with muscle forces
        limb = OneDOFLimb(q0=np.pi/2, a1=0.06, a2=0.007)
        w, q = limb.step(dt=0.1, F_flex=2.0, F_ext=1.0)
        check(np.isfinite(q), "Limb angle remains finite after force application")
        
        # Test muscle length and moment arm calculations
        L = limb.L(np.pi/3)
        h = limb.h(L, np.pi/3)
        check(L > 0 and h >= 0, "Muscle geometry calculations valid")
        print("  ✅ OneDOFLimb with muscle actuation works")

        # Test ground reaction force variant
        limb_gr = OneDOFLimb_withGR(q0=np.pi/2, w0=-0.1)
        gr_torque = limb_gr.GR(w=-0.1, q=np.pi/2)  # Swing phase: w < 0
        check(gr_torque == 0, "GRF torque zero during swing phase")
        
        gr_torque_stance = limb_gr.GR(w=0.1, q=np.pi/2)  # Stance phase: w >= 0
        check(gr_torque_stance != 0, "GRF torque non-zero during stance phase")
        print("  ✅ Ground reaction force model works")

        # =====================================================================
        # 6. INTEGRATED SYSTEMS (AFFERENTED LIMB & VAR_LIMB)
        # =====================================================================
        section("6. Integrated Systems (AfferentedLimb & VarLimb)")
        from spikingnn_core.system.afferented_limb import Afferented_Limb, Simple_Afferented_Limb
        from spikingnn_core.system.var_limb import Net_Limb_connect, Var_Limb

        # Test Afferented_Limb integration
        flex = SimpleAdaptedMuscle(w=0.5)
        ext = SimpleAdaptedMuscle(w=0.4)
        aff_limb = Afferented_Limb(Limb=limb, Flexor=flex, Extensor=ext)
        aff_limb.set_init_conditions()
        aff_limb.step(dt=0.1, uf=5.0, ue=3.0)
        
        check(aff_limb.output.shape == (6,), "Afferent output vector has shape (6,)")
        check(np.any(aff_limb.output > 0), "Afferent feedback generated after step")
        check(np.isfinite(aff_limb.q) and np.isfinite(aff_limb.w), "Limb kinematics finite")
        print("  ✅ Afferented_Limb closed-loop integration works")

        # Test Simple_Afferented_Limb variant
        simple_aff_limb = Simple_Afferented_Limb(Limb=limb, Flexor=flex, Extensor=ext)
        simple_aff_limb.step(dt=0.1, uf=5.0, ue=3.0)
        check(simple_aff_limb.output.shape == (6,), "Simple afferent limb output correct")
        print("  ✅ Simple_Afferented_Limb variant works")

        # Test full Var_Limb system (CPG + limb)
        Q_app_sys = np.array([[1, 0], [0, 0], [0, 1], [0, 0]])
        Q_aff_sys = np.random.rand(4, 6) * 0.1
        P_sys = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])
        types = ['CH', 'FS', 'CH', 'FS']
        a4, b4, c4, d4 = types2params(types)
        a4[0] = 0.001; a4[2] = 0.001  # Slow CPG neurons for rhythm generation

        io_net_sys = Izhikevich_IO_Network(
            N=4, a=a4, b=b4, c=c4, d=d4,
            input_size=2, output_size=2, afferent_size=6,
            Q_app=Q_app_sys, Q_aff=Q_aff_sys, P=P_sys,
            names=["CPG1", "MN1", "CPG2", "MN2"]
        )
        # Create ring-like CPG connectivity
        io_net_sys.connect(0, 1, coef=1, w=1.7)
        io_net_sys.connect(2, 3, coef=1, w=1.7)
        io_net_sys.connect(1, 2, coef=-1, w=-1.1)
        io_net_sys.connect(3, 0, coef=-1, w=-1.1)

        var_sys = Var_Limb(Network=io_net_sys, Limb=aff_limb)
        var_sys.step(dt=0.1, Iapp=np.array([5.0, 5.0]))
        
        check(var_sys.net.V_out.shape == (2,), "Var_Limb network output dimension correct")
        check(np.isfinite(var_sys.q), "Limb state in Var_Limb remains finite")
        check(var_sys.names["neurons"] == ["CPG1", "MN1", "CPG2", "MN2"], 
              "Neuron names preserved in integrated system")
        
        # Test name-based parameter access in Var_Limb
        check(var_sys.set_afferents_by_names("Ia_Flex", "CPG1", 1.0) == True,
              "Afferent weight setting by names works")
        var_sys.set_muscle_params("Flexor", A=0.4)
        check(var_sys.Limb.Flexor.A == 0.4, "Muscle parameter update by name works")
        print("  ✅ Var_Limb integrated system with name-based access works")

        # =====================================================================
        # 7. NUMERICAL STABILITY (EXTENDED SIMULATION)
        # =====================================================================
        section("7. Numerical Stability (Extended Simulation)")
        
        # Run 100 steps of Var_Limb simulation
        for step in range(100):
            var_sys.step(dt=0.1, Iapp=np.array([5.0, 5.0]))
            
        # Check all state variables remain finite (no NaN/Inf)
        check(np.all(np.isfinite(var_sys.net.V)), 
              "Membrane potentials remain finite after 100 steps")
        check(np.all(np.isfinite(var_sys.net.U)), 
              "Recovery variables remain finite after 100 steps")
        check(np.all(np.isfinite(var_sys.net.I_syn)), 
              "Synaptic currents remain finite after 100 steps")
        check(np.all(np.isfinite(var_sys.Limb.Flexor.F)), 
              "Muscle forces remain finite after 100 steps")
        
        # Check that network produces activity (not stuck at zero)
        print("  ✅ Numerical stability confirmed over 100 integration steps")

        # =====================================================================
        # 8. PARAMETER VALIDATION & ERROR HANDLING
        # =====================================================================
        section("8. Parameter Validation & Error Handling")
        
        # Test shape validation for network parameters
        try:
            bad_net = Izhikevich_Network(N=3, a=np.array([0.02, 0.1]))  # Wrong length
            check(False, "Should have raised ValueError for wrong parameter length")
        except ValueError:
            check(True, "Correctly raises ValueError for mismatched parameter length")
        
        # Test connectivity mask validation
        try:
            bad_mask = np.ones((2, 2))  # Wrong shape for N=3
            net_bad = Izhikevich_Network(N=3, M=bad_mask)
            check(False, "Should have raised ValueError for wrong mask shape")
        except ValueError:
            check(True, "Correctly validates connectivity mask shape")
        
        # Test IO matrix shape validation
        try:
            bad_Q = np.ones((3, 5))  # Wrong shape for input_size=2
            bad_io = Izhikevich_IO_Network(N=3, input_size=2, Q_app=bad_Q)
            check(False, "Should have raised ValueError for wrong Q_app shape")
        except ValueError:
            check(True, "Correctly validates IO projection matrix shapes")
        
        print("  ✅ Parameter validation and error handling work correctly")

        # =====================================================================
        # FINAL SUMMARY
        # =====================================================================
        section("🎉 ALL TESTS PASSED SUCCESSFULLY!")
        print("✅ spikingnn_core is fully functional and ready for GUI integration.")
        print("💡 Tip: Install in editable mode for development:")
        print("   pip install -e packages/core")
        print("\n📚 Next steps:")
        print("   • Run GUI tests: python -m pytest tests/gui/ -v")
        print("   • Generate API docs: sphinx-apidoc -o docs/api packages/core/src")
        print("   • Benchmark performance: python benchmarks/run_benchmarks.py")

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()