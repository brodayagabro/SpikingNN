#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Integration tests for spikingnn_core components.

This module contains comprehensive tests for:
- Izhikevich and FitzHugh-Nagumo neuron networks
- Neuromechanical components (muscles, afferents, limbs)
- Integrated systems (Var_Limb, Net_Limb_connect)
- Visualization utilities

All tests generate matplotlib plots for visual verification.
"""

import numpy as np
from scipy import signal as sig
from matplotlib import pyplot as plt

# Import from the modular core package
from spikingnn_core import (
    Izhikevich_Network,
    Izhikevich_IO_Network,
    NameNetwork,
    types2params,
    izhikevich_neuron,
    FizhugNagumoNetwork,
    SimpleAdaptedMuscle,
    Afferents,
    Simple_Afferents,
    Pendulum,
    OneDOFLimb,
    OneDOFLimb_withGR,
    Afferented_Limb,
    Simple_Afferented_Limb,
    Net_Limb_connect,
    Var_Limb,
)
from spikingnn_core.utils.net_preparation import (
    create_firing_rastr,
    find_bursts,
    get_bursts_regions,
    get_brusts_duration,
    get_brust_frequency,
)


def run_net(
    T: np.ndarray,
    net: NameNetwork,
    I_app: callable,
    I_aff: callable
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run a neural network simulation over a discrete time array.
    
    Parameters
    ----------
    T : np.ndarray
        Discrete time array (ms).
    net : NameNetwork
        Neural network object with step() method.
    I_app : callable
        Function returning applied input current array for each time step.
    I_aff : callable
        Function returning afferent input array for each time step.
        
    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Arrays U and V of shape (len(T), N) containing recovery variable
        and membrane potential trajectories for all neurons.
    """
    dt = T[1] - T[0]
    N = len(net)
    U = np.zeros((len(T), N))
    V = np.zeros((len(T), N))
    
    for i, t in enumerate(T):
        U[i] = net.U_prev
        V[i] = net.V_prev
        net.step(dt=dt, Iapp=I_app(t), Iaff=I_aff(t))
    
    return U, V


def test_IzhIOnet_step():
    """
    Test basic step functionality of Izhikevich_IO_Network.
    
    Verifies that a single integration step executes without errors
    with proper input/output matrix dimensions.
    """
    Q_app = np.array([
        [1, 0],
        [0, 1],
        [0, 0],
        [0, 0]
    ])
    Q_aff = np.array([
        [1, 0],
        [0, 1],
        [0, 0],
        [0, 0]
    ])
    P = np.array([
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    
    net = Izhikevich_IO_Network(
        input_size=2,
        output_size=2,
        afferent_size=2,
        N=4,
        Q_app=Q_app,
        Q_aff=Q_aff,
        P=P
    )
    
    I_app = np.array([0, 0])
    I_aff = np.zeros(2)
    net.step(dt=0.1, Iapp=I_app, Iaff=I_aff)
    print("✅ test_IzhIOnet_step passed")


def test_IzhIOnet_sym():
    """
    Test symmetric connectivity pattern in Izhikevich_IO_Network.
    
    Generates and plots membrane potentials and spike raster for a 
    4-neuron network with ring-like connectivity and random afferent input.
    """
    Q_app = np.array([
        [1, 0],
        [0, 1],
        [0, 0],
        [0, 0]
    ])
    Q_aff = np.array([
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [1, 0, 1, 0, 1, 0],
        [0, 0, 0, 1, 0, 1]
    ])
    P = np.array([
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    W = np.array([
        [0, 0, 0, -1],
        [0, 0, 1, 0],
        [0, -1, 0, 0],
        [1, 0, 0, 0]
    ])
    tau_syn = np.array([
        [1, 1, 1, 20],
        [1, 1, 10, 1],
        [1, 20, 1, 1],
        [10, 1, 1, 1]
    ])

    net = Izhikevich_IO_Network(
        input_size=2,
        output_size=2,
        afferent_size=6,
        N=4,
        Q_app=Q_app,
        Q_aff=Q_aff,
        P=P,
        W=W
    )
    net.set_init_conditions(v_noise=np.random.normal(size=net.N))
    net.set_synaptic_relax_constant(tau_syn)
    
    T = np.linspace(0, 500, 2000)
    I_app = lambda t: np.array([0, 0])
    I_aff = lambda t: np.zeros(6) + 2 * np.random.rand(6)
    
    U, V = run_net(T, net, I_app, I_aff)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
    
    # Membrane potentials
    for i in range(net.N):
        ax1.plot(T, V[:, i], label=f"V({net.names[i]})")
    ax1.set_xlabel("Time (ms)")
    ax1.set_ylabel("Membrane potential (mV)")
    ax1.legend()
    ax1.set_title("Membrane Potentials")
    ax1.grid(alpha=0.3)
    
    # Spike raster
    firing_rastr = create_firing_rastr(V, T, V_peak=30)
    ax2.scatter(firing_rastr[0], firing_rastr[1], s=0.5, c='black')
    ax2.set_yticks(list(range(net.N)))
    ax2.set_yticklabels(net.names)
    ax2.set_xlabel("Time (ms)")
    ax2.set_ylabel("Neuron")
    ax2.set_title("Spike Raster")
    ax2.grid(alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.show()
    print("✅ test_IzhIOnet_sym passed")


def test_Afferents():
    """
    Test afferent feedback models (Ia, II, Ib types) with sinusoidal limb motion.
    
    Generates plots showing:
    - Limb kinematics (angle, velocity, length, force)
    - Afferent activity patterns with threshold indicators
    """
    afferents = Afferents()
    T = np.linspace(0, 10, 1000)  # Time in seconds
    
    # Dynamics settings
    freq = 0.5  # rad/s, frequency of limb oscillations
    A_q = np.pi / 5  # Amplitude of angle
    A_f = 4  # Amplitude of force
    a1, a2 = 0.07, 0.01  # Muscle attachment geometry
    
    # Set afferent length threshold based on geometry
    afferents.L_th = np.sqrt(a1**2 + a2**2)
    
    # Generate limb kinematics
    q = A_q * np.cos(2 * np.pi * freq * T) + np.pi / 2  # Angle
    w = -A_q * np.sin(2 * np.pi * freq * T)  # Angular velocity
    
    # Calculate muscle length and moment arm
    L = np.sqrt(a1**2 + a2**2 - 2 * a1 * a2 * np.cos(q))
    h = a1 * a2 * np.sin(q) / L  # Moment arm
    v = w * h  # Muscle velocity
    
    # Calculate muscle force
    F = -A_f * np.cos(2 * np.pi * freq * T)
    
    # Motoneuron input (noise)
    input_signal = 0.1 * np.random.rand(len(T))
    
    # Calculate afferent activities
    Ia = afferents.Ia(v, L, input_signal)
    II = afferents.II(L, input_signal)
    Ib = afferents.Ib(F)
    
    # Identify threshold crossings for annotation
    firing_L = np.where(L >= afferents.L_th)[0]
    firing_v = np.where(v > 0)[0]
    firing_F = np.where(F >= afferents.F_th)[0]
    
    # Plotting
    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    
    # Row 1: Kinematics
    ax = axes[0, 0]
    ax.plot(T, q, label="q (angle)")
    ax.plot(T, w, label="w (velocity)")
    ax.plot(T, input_signal, label="input", linewidth=0.5, alpha=0.7)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Value")
    ax.legend()
    ax.set_title("Limb Kinematics & Input")
    ax.grid(alpha=0.3)
    
    ax = axes[1, 0]
    ax.plot(T, F, label="F (force)")
    ax.axhline(y=afferents.F_th, color='red', linestyle='--', label="F_th")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Force (N)")
    ax.legend()
    ax.set_title("Muscle Force")
    ax.grid(alpha=0.3)
    
    ax = axes[2, 0]
    ax.plot(T, L, label="L (length)")
    ax.plot(T, h, label="h (moment arm)")
    ax.plot(T, v, label="v (velocity)")
    ax.axhline(y=afferents.L_th, color='red', linestyle='--', label="L_th")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Value")
    ax.legend()
    ax.set_title("Muscle Geometry")
    ax.grid(alpha=0.3)
    
    # Row 2: Afferent activities
    ax = axes[0, 1]
    ax.plot(T, Ia, label="Ia", color='blue')
    ax.vlines(T[firing_v], np.min(Ia), np.max(Ia), color='orange', alpha=0.3, label="v > 0")
    ax.vlines(T[firing_L], np.min(Ia), np.max(Ia), color='yellow', alpha=0.3, label="L >= L_th")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Activity")
    ax.legend()
    ax.set_title("Ia-type Afferent Activity")
    ax.grid(alpha=0.3)
    
    ax = axes[1, 1]
    ax.plot(T, II, label="II", color='green')
    ax.vlines(T[firing_L], np.min(II), np.max(II), color='yellow', alpha=0.3, label="L >= L_th")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Activity")
    ax.legend()
    ax.set_title("II-type Afferent Activity")
    ax.grid(alpha=0.3)
    
    ax = axes[2, 1]
    ax.plot(T, Ib, label="Ib", color='red')
    ax.vlines(T[firing_F], np.min(Ib), np.max(Ib), color='yellow', alpha=0.3, label="F >= F_th")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Activity")
    ax.legend()
    ax.set_title("Ib-type Afferent Activity")
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    print("✅ test_Afferents passed")


def test_all():
    """
    Full integration test: Izhikevich network + muscles + limb.
    
    Simulates a 4-neuron CPG controlling antagonistic muscles
    attached to a one-degree-of-freedom limb. Plots:
    - Membrane potentials
    - Limb angle trajectory
    - Spike raster
    - Muscle activation dynamics
    """
    N = 4
    types = ['IB', 'IB', 'RS', 'IB']
    A, B, C, D = types2params(types)
    
    net = Izhikevich_Network(N=N, a=A, b=B, c=C, d=D)
    net.set_init_conditions(v_noise=np.random.normal(size=N))
    
    # Ring-like connectivity with alternating excitation/inhibition
    W = np.array([
        [0, 0, 0, -1.1],
        [0.7, 0, 0, 0],
        [0, -1.1, 0, 0],
        [0, 0, 0.7, 0]
    ])
    net.M = np.ones((N, N))
    net.set_weights(W)
    
    tau_syn = np.random.randint(10, 20, (N, N))
    net.set_synaptic_relax_constant(tau_syn)
    
    # Simulation parameters
    T = np.linspace(0, 10000, 20000)
    I_base = np.zeros(N)
    I_base[0] = 5
    I_base[1] = 5
    I_app = lambda t: (I_base + 3 * np.random.normal(size=N)) * (t < 3000)
    
    # Muscles and limb
    flexor = SimpleAdaptedMuscle(w=0.5)
    extensor = SimpleAdaptedMuscle(w=0.4)
    Limb = OneDOFLimb(q0=np.pi/2, b=0.005, a1=0.2, a2=0.05, m=0.3, ls=0.3)
    
    # Run simulation using legacy run() function
    U, V, Cn_f, X_f, F_f, Cn_e, X_e, F_e, W_out, Q = run(
        net, flexor, extensor, Limb, T, I_app
    )
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Membrane potentials
    ax = axes[0, 0]
    for i in range(N):
        ax.plot(T, V[:, i], label=f"{net.names[i]}")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("V (mV)")
    ax.legend()
    ax.set_title("Membrane Potentials")
    ax.grid(alpha=0.3)
    
    # Limb angle
    ax = axes[0, 1]
    ax.plot(T, Q, label=r'$q$ (angle)')
    ax.axhline(y=np.pi/2, color='red', linestyle='--', label=r'$\pi/2$')
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Angle (rad)")
    ax.legend()
    ax.set_title("Limb Angle")
    ax.grid(alpha=0.3)
    
    # Spike raster
    ax = axes[1, 0]
    firing_rastr = create_firing_rastr(V, T, V_peak=30)
    ax.scatter(firing_rastr[0], firing_rastr[1], s=0.3, c='black')
    ax.set_yticks(list(range(N)))
    ax.set_yticklabels(net.names)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Neuron")
    ax.set_title("Spike Raster")
    ax.grid(alpha=0.3, axis='x')
    
    # Muscle dynamics
    ax = axes[1, 1]
    ax.plot(T, Cn_f, label="Cn_f (flexor activation)")
    ax.plot(T, Cn_e, label="Cn_e (extensor activation)")
    ax.plot(T, F_f, label="F_f (flexor force)", linestyle='--')
    ax.plot(T, F_e, label="F_e (extensor force)", linestyle='--')
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Value")
    ax.legend()
    ax.set_title("Muscle Dynamics")
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    print("✅ test_all passed")


def run_pendulum(T: np.ndarray, Limb: Pendulum, M: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Run pendulum simulation with time-varying torque.
    
    Parameters
    ----------
    T : np.ndarray
        Time array (ms).
    Limb : Pendulum
        Pendulum object.
    M : np.ndarray
        Applied torque array (N·m).
        
    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Angular velocity and angle arrays.
    """
    dt = T[1] - T[0]
    W = np.zeros(len(T))
    Q = np.zeros(len(T))
    
    for i in range(len(T)):
        Q[i] = Limb.q
        W[i] = Limb.w
        Limb.step(dt=dt, M=M[i])
    
    return W, Q


def test_Pendulum():
    """
    Test basic pendulum dynamics with sinusoidal driving torque.
    
    Plots angle and angular velocity over time.
    """
    Limb = Pendulum(q0=np.pi/2 - 0.1, b=0.01)
    T = np.linspace(0, 2000, 20000)
    M = 0.01 * np.sin(T / 100)  # Small driving torque
    
    W, Q = run_pendulum(T, Limb, M)
    
    plt.figure(figsize=(10, 4))
    plt.plot(T, Q, label="q (angle)")
    plt.plot(T, W, label="w (velocity)")
    plt.xlabel("Time (ms)")
    plt.ylabel("Value")
    plt.legend()
    plt.title("Pendulum Dynamics")
    plt.grid(alpha=0.3)
    plt.show()
    print("✅ test_Pendulum passed")


def run_OneDOFLimb(T: np.ndarray, Limb: OneDOFLimb, Flex: np.ndarray, Ext: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Run one-degree-of-freedom limb simulation with muscle forces.
    
    Parameters
    ----------
    T : np.ndarray
        Time array (ms).
    Limb : OneDOFLimb
        Limb object.
    Flex : np.ndarray
        Flexor force array (N).
    Ext : np.ndarray
        Extensor force array (N).
        
    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Angular velocity and angle arrays.
    """
    dt = T[1] - T[0]
    W = np.zeros(len(T))
    Q = np.zeros(len(T))
    
    for i in range(len(T)):
        Q[i] = Limb.q
        W[i] = Limb.w
        Limb.step(dt=dt, F_flex=Flex[i], F_ext=Ext[i])
    
    return W, Q


def test_OneDOFLimb():
    """
    Test OneDOFLimb with sinusoidal muscle forces.
    
    Plots angle and angular velocity trajectories.
    """
    Limb = OneDOFLimb(q0=np.pi/2 - 1, a1=7, a2=30)
    print(f"Limb natural period: {Limb.own_T:.2f} ms")
    
    T = np.linspace(0, 2000, 20000)
    F_flex = 0.1 * np.cos(np.pi / 1000 * T)
    F_ext = 0.1 * np.sin(np.pi / 1000 * T)
    
    W, Q = run_OneDOFLimb(T, Limb, F_flex, F_ext)
    
    plt.figure(figsize=(10, 4))
    plt.plot(T, Q, label="q (angle)")
    plt.plot(T, W, label="w (velocity)")
    plt.xlabel("Time (ms)")
    plt.ylabel("Value")
    plt.legend()
    plt.title("OneDOFLimb Dynamics")
    plt.grid(alpha=0.3)
    plt.show()
    print("✅ test_OneDOFLimb passed")


def test_OneDOFLimb_withGR():
    """
    Test OneDOFLimb_withGR (with ground reaction force).
    
    Plots angle and angular velocity with stance-phase GRF.
    """
    Limb = OneDOFLimb_withGR(q0=np.pi/2, w0=-0.1, a1=7, a2=30, b=0.01)
    print(f"Limb natural period: {Limb.own_T:.2f} ms")
    
    T = np.linspace(0, 2000, 20000)
    F_flex = 0.1 * np.cos(np.pi / 1000 * T)
    F_ext = 0.1 * np.sin(np.pi / 1000 * T)
    
    W, Q = run_OneDOFLimb(T, Limb, F_flex, F_ext)
    
    plt.figure(figsize=(10, 4))
    plt.plot(T, Q, label="q (angle)")
    plt.plot(T, W, label="w (velocity)")
    plt.xlabel("Time (ms)")
    plt.ylabel("Value")
    plt.legend()
    plt.title("OneDOFLimb_withGR Dynamics")
    plt.grid(alpha=0.3)
    plt.show()
    print("✅ test_OneDOFLimb_withGR passed")


def run_Aff_Limb(T: np.ndarray, AL: Afferented_Limb, uf: np.ndarray, ue: np.ndarray) -> tuple:
    """
    Run afferented limb simulation with neural drive signals.
    
    Parameters
    ----------
    T : np.ndarray
        Time array (ms).
    AL : Afferented_Limb
        Afferented limb object.
    uf : np.ndarray
        Flexor neural drive array.
    ue : np.ndarray
        Extensor neural drive array.
        
    Returns
    -------
    tuple
        (M_tot, F_flex, F_ext, W, Q, Output) arrays.
    """
    dt = T[1] - T[0]
    M_tot = np.zeros(len(T))
    F_f = np.zeros(len(T))
    F_e = np.zeros(len(T))
    W = np.zeros(len(T))
    Q = np.zeros(len(T))
    Output = np.zeros((len(T), 6))
    
    for i, t in enumerate(T):
        F_f[i] = AL.Flexor.F_prev
        F_e[i] = AL.Extensor.F_prev
        M_tot[i] = AL.Limb.M_tot
        Q[i] = AL.Limb.q
        W[i] = AL.Limb.w
        Output[i] = AL.output
        AL.step(dt=dt, uf=uf[i], ue=ue[i])
    
    return M_tot, F_f, F_e, W, Q, Output


def test_Afferented_Limb():
    """
    Test Afferented_Limb with modulated square-wave neural drive.
    
    Generates comprehensive plots of:
    - Control signals and muscle forces
    - Limb kinematics and muscle geometry
    - Ia and II afferent activities with threshold annotations
    """
    flexor = SimpleAdaptedMuscle(w=0.5, N=2)
    extensor = SimpleAdaptedMuscle(w=0.4, N=2)
    Limb = OneDOFLimb(q0=np.pi/2, b=0.00, a1=0.4, a2=0.05, m=0.3, ls=0.3)
    AL = Afferented_Limb(Limb=Limb, Flexor=flexor, Extensor=extensor)
    
    T = np.linspace(0, 10000, 20000)
    mod_sig = np.sin(1 * np.pi * T / 1000)
    
    # Generate modulated square-wave neural drive
    uf = 15 * (sig.square(2 * np.pi * T / 50, duty=0.2) + 1)
    uf = np.where(mod_sig > 0.6, uf, 0)
    ue = 15 * (sig.square(2 * np.pi * T / 50, duty=0.2) + 1)
    ue = np.where(mod_sig < -0.6, ue, 0)
    
    M, F_f, F_e, W, Q, Output = run_Aff_Limb(T, AL, uf, ue)
    Ia_f, II_f, Ib_f = Output[:, 0], Output[:, 1], Output[:, 2]
    Ia_e, II_e, Ib_e = Output[:, 3], Output[:, 4], Output[:, 5]
    
    # Calculate muscle geometry for annotation
    L_f = AL.Limb.L(Q)
    L_e = AL.Limb.L(np.pi - Q)
    hf = AL.Limb.h(L_f, Q)
    he = AL.Limb.h(L_e, np.pi - Q)
    vf = W * hf
    ve = -W * he
    
    firing_Lf = np.where(L_f >= AL.Afferents.L_th)[0]
    firing_Le = np.where(L_e >= AL.Afferents.L_th)[0]
    firing_vf = np.where(vf > 0)[0]
    firing_ve = np.where(ve > 0)[0]
    
    # Plot 1: Control, forces, kinematics, geometry
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    
    ax = axes[0, 0]
    ax.plot(T, uf, label='uf (flexor drive)', linewidth=0.5)
    ax.plot(T, ue, label='ue (extensor drive)', linewidth=0.5)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Neural drive")
    ax.legend()
    ax.set_title("Control Signals")
    ax.grid(alpha=0.3)
    
    ax = axes[1, 0]
    ax.plot(T, F_f, label="F_flex")
    ax.plot(T, F_e, label="F_ext")
    ax.axhline(y=AL.Afferents.F_th, color='red', linestyle='--', label="F_th")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Force (N)")
    ax.legend()
    ax.set_title("Muscle Forces")
    ax.grid(alpha=0.3)
    
    ax = axes[2, 0]
    ax.plot(T, Q, label="q (angle)")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Angle (rad)")
    ax.legend()
    ax.set_title("Limb Angle")
    ax.grid(alpha=0.3)
    
    ax = axes[0, 1]
    ax.plot(T, L_f, label="L_f (flexor length)")
    ax.plot(T, L_e, label="L_e (extensor length)")
    ax.axhline(y=AL.Afferents.L_th, color='red', linestyle='--', label="L_th")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Length (m)")
    ax.legend()
    ax.set_title("Muscle Lengths")
    ax.grid(alpha=0.3)
    
    ax = axes[1, 1]
    ax.plot(T, hf, label="h_f (flexor moment arm)")
    ax.plot(T, he, label="h_e (extensor moment arm)")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Moment arm (m)")
    ax.legend()
    ax.set_title("Moment Arms")
    ax.grid(alpha=0.3)
    
    ax = axes[2, 1]
    ax.plot(T, vf, label="v_f (flexor velocity)")
    ax.plot(T, ve, label="v_e (extensor velocity)")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Velocity (m/s)")
    ax.legend()
    ax.set_title("Muscle Velocities")
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Plot 2: Ia afferent activities
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6))
    
    ax1.plot(T, Ia_f, label='Ia_f', color='blue')
    ax1.vlines(T[firing_vf], 0, 0.5, color='orange', alpha=0.4, label="v_f > 0")
    ax1.vlines(T[firing_Lf], 0.1, 0.6, color='yellow', alpha=0.4, label="L_f >= L_th")
    ax1.set_xlabel("Time (ms)")
    ax1.set_ylabel("Activity")
    ax1.legend()
    ax1.set_title("Ia-type Afferent Activity (Flexor)")
    ax1.grid(alpha=0.3)
    
    ax2.plot(T, Ia_e, label='Ia_e', color='blue')
    ax2.vlines(T[firing_ve], 0, 0.5, color='orange', alpha=0.4, label="v_e > 0")
    ax2.vlines(T[firing_Le], 0.1, 0.6, color='yellow', alpha=0.4, label="L_e >= L_th")
    ax2.set_xlabel("Time (ms)")
    ax2.set_ylabel("Activity")
    ax2.legend()
    ax2.set_title("Ia-type Afferent Activity (Extensor)")
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Plot 3: II afferent activities
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 4))
    
    ax1.plot(T, II_f, label='II_f', color='green')
    ax1.vlines(T[firing_Lf], 0, 0.1, color='yellow', alpha=0.4, label="L_f >= L_th")
    ax1.set_xlabel("Time (ms)")
    ax1.set_ylabel("Activity")
    ax1.legend()
    ax1.set_title("II-type Afferent Activity (Flexor)")
    ax1.grid(alpha=0.3)
    
    ax2.plot(T, II_e, label='II_e', color='green')
    ax2.vlines(T[firing_Le], 0, 0.1, color='yellow', alpha=0.4, label="L_e >= L_th")
    ax2.set_xlabel("Time (ms)")
    ax2.set_ylabel("Activity")
    ax2.legend()
    ax2.set_title("II-type Afferent Activity (Extensor)")
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    print("✅ test_Afferented_Limb passed")


def test_FHN_Network():
    """
    Test FitzHugh-Nagumo network dynamics.
    
    Simulates a 4-neuron FHN network with ring connectivity
    and plots membrane potential trajectories.
    """
    N = 4
    net = FizhugNagumoNetwork(N=N)
    net.a = np.array([-0.1, -0.1, 0.1, 0.1])
    net.V_th = 1 * np.ones(N)
    net.ts = np.array([0.1, 0.1, 0.1, 0.1])
    net.V = 0.5 + np.random.rand(N)
    
    # Ring connectivity
    net.M = np.ones((N, N)) - np.eye(N)
    W = np.array([
        [0, 0, 0, -1.1],
        [0.7, 0, 0, 0],
        [0, -1.1, 0, 0],
        [0, 0, 0.7, 0]
    ])
    net.set_weights(W)
    
    T = np.linspace(0, 4000, 20000)
    I_base = np.zeros(N)
    I_app = lambda t: (I_base + 0.01 * np.random.normal(size=N)) * (t < 3000)
    I_aff = lambda t: 0
    
    U, V = run_net(T, net, I_app, I_aff)
    
    plt.figure(figsize=(10, 4))
    for i in range(N):
        plt.plot(T, V[:, i], label=f"{net.names[i]}")
    plt.xlabel("Time (ms)")
    plt.ylabel("Membrane potential")
    plt.legend()
    plt.title("FitzHugh-Nagumo Network Dynamics")
    plt.grid(alpha=0.3)
    plt.show()
    print("✅ test_FHN_Network passed")


def run(net, flexor, extensor, Limb, T, Iapp):
    """
    Legacy running procedure for limb controlled by two muscles.
    
    Note: This function is kept for backward compatibility.
    New code should use the Var_Limb or Net_Limb_connect classes directly.
    
    Parameters
    ----------
    net : NameNetwork
        Neural network object.
    flexor, extensor : SimpleAdaptedMuscle
        Muscle objects.
    Limb : OneDOFLimb
        Limb object.
    T : np.ndarray
        Time array.
    Iapp : callable
        Input current function.
        
    Returns
    -------
    tuple
        State variable arrays: U, V, Cn_f, X_f, F_f, Cn_e, X_e, F_e, W, Q
    """
    dt = T[1] - T[0]
    N = len(net)
    U = np.zeros((len(T), N))
    V = np.zeros((len(T), N))
    Cn_f = np.zeros(len(T))
    X_f = np.zeros(len(T))
    F_f = np.zeros(len(T))
    Cn_e = np.zeros(len(T))
    X_e = np.zeros(len(T))
    F_e = np.zeros(len(T))
    W = np.zeros(len(T))
    Q = np.zeros(len(T))
    
    alpha_f, alpha_e = 1, 1
    
    for i, t in enumerate(T):
        U[i] = net.U_prev
        V[i] = net.V_prev
        Cn_f[i] = flexor.Cn_prev
        X_f[i] = flexor.x
        F_f[i] = flexor.F_prev
        Cn_e[i] = extensor.Cn_prev
        X_e[i] = extensor.x
        F_e[i] = extensor.F_prev
        Q[i] = Limb.q
        W[i] = Limb.w
        
        net.step(dt=dt, Iapp=Iapp(t))
        uf = alpha_f * net.output[0]
        ue = alpha_f * net.output[2]
        
        flexor.step(dt=dt, u=uf)
        extensor.step(dt=dt, u=ue)
        Limb.step(dt=dt, F_flex=flexor.F, F_ext=extensor.F)
    
    return U, V, Cn_f, X_f, F_f, Cn_e, X_e, F_e, W, Q


def test_FHN_Network_with_Limb():
    """
    Test FitzHugh-Nagumo network connected to limb mechanics.
    
    Integrates FHN network dynamics with muscle activation and limb motion.
    """
    N = 4
    net = FizhugNagumoNetwork(N=N)
    net.a = np.array([-0.1, -0.1, 0.1, 0.1])
    net.V_th = 0.1 * np.ones(N)
    net.ts = np.array([0.1, 0.1, 0.1, 0.1])
    net.V = 0.5 + np.random.rand(N)
    
    net.M = np.ones((N, N)) - np.eye(N)
    W = np.array([
        [0, 0, 0, -1.1],
        [0.7, 0, 0, 0],
        [0, -1.1, 0, 0],
        [0, 0, 0.7, 0]
    ])
    tau_syn = np.array([
        [1, 1, 1, 20],
        [1, 1, 10, 1],
        [1, 20, 1, 1],
        [10, 1, 1, 1]
    ])
    net.set_weights(W)
    net.set_synaptic_relax_constant(tau_syn)
    
    T = np.linspace(0, 10000, 50000)
    I_base = np.zeros(N)
    I_app = lambda t: (I_base + 0.01 * np.random.normal(size=N)) * (t < 3000)
    
    flexor = SimpleAdaptedMuscle(w=0.5, N=1)
    extensor = SimpleAdaptedMuscle(w=0.5, N=1)
    Limb = OneDOFLimb(q0=np.pi/2, b=0.005, a1=0.2, a2=0.05, m=0.3, ls=0.3)
    
    U, V, Cn_f, X_f, F_f, Cn_e, X_e, F_e, W_out, Q = run(
        net, flexor, extensor, Limb, T, I_app
    )
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Membrane potentials
    ax = axes[0, 0]
    for i in range(N):
        ax.plot(T, V[:, i], label=f"{net.names[i]}")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("V")
    ax.legend()
    ax.set_title("FHN Membrane Potentials")
    ax.grid(alpha=0.3)
    
    # Limb angle
    ax = axes[0, 1]
    ax.plot(T, Q, label=r'$q$')
    ax.axhline(y=np.pi/2, color='red', linestyle='--', label=r'$\pi/2$')
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Angle (rad)")
    ax.legend()
    ax.set_title("Limb Angle")
    ax.grid(alpha=0.3)
    
    # Spike raster
    ax = axes[1, 0]
    firing_rastr = create_firing_rastr(V, T, V_peak=0.25)
    ax.scatter(firing_rastr[0], firing_rastr[1], s=0.2, c='black')
    ax.set_yticks(list(range(N)))
    ax.set_yticklabels(net.names)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Neuron")
    ax.set_title("Spike Raster")
    ax.grid(alpha=0.3, axis='x')
    
    # Muscle dynamics
    ax = axes[1, 1]
    ax.plot(T, Cn_f, label="Cn_f")
    ax.plot(T, Cn_e, label="Cn_e")
    ax.plot(T, F_f, label="F_f", linestyle='--')
    ax.plot(T, F_e, label="F_e", linestyle='--')
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Value")
    ax.legend()
    ax.set_title("Muscle Dynamics")
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    print("✅ test_FHN_Network_with_Limb passed")


def test_Net_Limb_connect():
    """
    Test integrated Net_Limb_connect system (CPG + limb).
    
    Simulates a 4-neuron CPG with afferent feedback controlling
    a one-degree-of-freedom limb. Plots network activity, muscle
    forces, afferent signals, and limb kinematics.
    """
    Q_app = np.array([
        [1, 0],
        [0, 0],
        [0, 1],
        [0, 0]
    ])
    Q_aff = np.random.rand(4, 6)
    P = np.array([
        [1, 0, 0, 0],
        [0, 0, 1, 0]
    ])
    
    types = ['CH', 'FS', 'CH', 'FS']
    A, B, C, D = types2params(types)
    A[0] = 0.001  # Slow CPG neurons
    A[2] = 0.001
    
    net = Izhikevich_IO_Network(
        input_size=2,
        output_size=2,
        afferent_size=6,
        N=4,
        Q_app=Q_app,
        Q_aff=Q_aff,
        P=P
    )
    net.set_params(a=A, b=B, c=C, d=D)
    
    # Ring connectivity for CPG
    W = np.array([
        [0, 0, 0, -1.1],
        [1.7, 0, 0, 0],
        [0, -1.1, 0, 0],
        [0, 0, 1.7, 0]
    ])
    net.M = np.ones((4, 4))
    net.set_weights(W)
    tau_syn = 20 * np.ones((4, 4))
    net.set_synaptic_relax_constant(tau_syn)
    
    # Limb components
    flexor = SimpleAdaptedMuscle(w=0.5, N=2)
    extensor = SimpleAdaptedMuscle(w=0.4, N=2)
    Limb = OneDOFLimb(q0=np.pi/2 + 0.4, b=0.001, a1=0.4, a2=0.05, m=0.3, ls=0.3)
    AL = Afferented_Limb(Limb=Limb, Flexor=flexor, Extensor=extensor)
    
    # Integrated system
    sys = Net_Limb_connect(Network=net, Limb=AL)
    
    # Simulation
    T = np.linspace(0, 20000, 50000)
    I_app = lambda t: np.zeros(2)  # No external input
    
    V = np.zeros((len(T), 4))
    F_flex = np.zeros(len(T))
    F_ext = np.zeros(len(T))
    Afferents = np.zeros((len(T), 6))
    Q = np.zeros(len(T))
    W_out = np.zeros(len(T))
    
    dt = T[1] - T[0]
    for i, t in enumerate(T):
        V[i] = sys.net.V_prev
        F_flex[i] = sys.F_flex
        F_ext[i] = sys.F_ext
        Afferents[i] = sys.Limb.output
        Q[i] = sys.q
        W_out[i] = sys.w
        sys.step(dt=dt, Iapp=I_app(t))
    
    # Plotting
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Network potentials
    ax = axes[0, 0]
    for i in range(4):
        ax.plot(T, V[:, i], label=f"{sys.net.names[i]}")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("V (mV)")
    ax.legend()
    ax.set_title("CPG Membrane Potentials")
    ax.grid(alpha=0.3)
    
    # Muscle forces
    ax = axes[0, 1]
    ax.plot(T, F_flex, label='Flexor')
    ax.plot(T, F_ext, label='Extensor')
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Force (N)")
    ax.legend()
    ax.set_title("Muscle Forces")
    ax.grid(alpha=0.3)
    
    # Afferent signals
    ax = axes[1, 0]
    aff_types = ['Ia_f', 'II_f', 'Ib_f', 'Ia_e', 'II_e', 'Ib_e']
    for i in range(6):
        ax.plot(T, Afferents[:, i], label=aff_types[i], linewidth=0.5)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Activity")
    ax.legend(fontsize=8)
    ax.set_title("Afferent Feedback Signals")
    ax.grid(alpha=0.3)
    
    # Limb kinematics
    ax = axes[1, 1]
    ax.plot(T, Q, label="q (angle)")
    ax.plot(T, W_out, label="w (velocity)")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Value")
    ax.legend()
    ax.set_title("Limb Kinematics")
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    print("✅ test_Net_Limb_connect passed")


def main():
    """
    Run all integration tests sequentially.
    
    Each test generates matplotlib plots for visual verification.
    Close each figure window to proceed to the next test.
    """
    print("🧪 Starting spikingnn_core integration tests\n")
    
    tests = [
        ("OneDOFLimb", test_OneDOFLimb),
        ("Pendulum", test_Pendulum),
        ("OneDOFLimb_withGR", test_OneDOFLimb_withGR),
        ("Full CPG+Limb", test_all),
        ("IzhIO step", test_IzhIOnet_step),
        ("IzhIO symmetric", test_IzhIOnet_sym),
        ("Afferents", test_Afferents),
        ("Afferented_Limb", test_Afferented_Limb),
        ("FHN Network", test_FHN_Network),
        ("FHN+Limb", test_FHN_Network_with_Limb),
        ("Net_Limb_connect", test_Net_Limb_connect),
    ]
    
    for name, test_func in tests:
        print(f"\n▶️ Running: {name}")
        try:
            test_func()
        except KeyboardInterrupt:
            print("\n⚠️ Interrupted by user")
            break
        except Exception as e:
            print(f"❌ {name} failed: {e}")
            import traceback
            traceback.print_exc()
            break
    
    print("\n✅ All tests completed")


if __name__ == "__main__":
    main()