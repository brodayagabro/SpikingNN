
import numpy as np
from scipy import signal as sig
from matplotlib import pyplot as plt
from Izh_net import *
from net_preparation import *


def run_net(T, net, I_app, I_aff):
    """
    Procedure of running network 
    args:
    T - discrete time array
    net - network object
    I_app - applied current
    I_aff - afferents activity
    retrun U, V - state of network array with shape(len(T), N), 
    N - size of
    """
    dt = T[1] - T[0]
    N = len(net)
    U = np.zeros((len(T), N))
    V = np.zeros((len(T), N))
    for i, t in enumerate(T):
        U[i] = net.U_prev
        V[i] = net.V_prev
        net.step(dt=dt, Iapp = I_app(t), Iaff=I_aff(t))
    return U, V


def test_IzhIOnet_step():
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
    net = Izhikevich_IO_Network(input_size=2, output_size=2, N=4, Q_app=Q_app,Q_aff = Q_aff , P=P)
    I_app = np.array([0, 0])
    I_aff = np.zeros(2)
    net.step(dt=0.1, Iapp=I_app, Iaff=I_aff)


def test_IzhIOnet_sym():
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

    net = Izhikevich_IO_Network(input_size=2, output_size=2,
                                afferent_size=6, N=4, Q_app=Q_app,
                                Q_aff = Q_aff , P=P, W=W)
    net.set_init_conditions(np.random.normal(size=net.N))
    net.set_synaptic_relax_constant(tau_syn)
    T = np.linspace(0, 500, 2000)
    I_app = lambda t: np.array([0, 0])
    I_aff = lambda t: np.zeros(6) + 2*np.random.rand(6)
    U, V = run_net(T, net, I_app, I_aff)
    plt.subplot(211)
    for i in range(net.N):
        #plt.plot(T, U[:, i], label=f"U({net.names[i]})")
        plt.plot(T, V[:, i], label=f"V({net.names[i]})")
    plt.legend()
    plt.subplot(212)
    firing_rastr = create_firing_rastr(V, T, 30)
    plt.scatter(firing_rastr[0], firing_rastr[1], s=0.1)
    plt.yticks(list(range(net.N)), net.names)
    plt.plot()
    plt.show()
    

def test_Afferents():
    # Create afferents
    afferents = Afferents()
    # Set discrete time
    T = np.linspace(0, 10, 1000) # time in seconds
    
    ### Dinamics settings
    # fequensi of limb oscilations
    freq = 0.5 # rad/s
    # Amplitude of angle
    A_q = np.pi/5
    # Amplitude of force
    A_f = 4
    # Muscle params
    a1 = 0.07
    a2 = 0.01
    # Set threshold Length of afferents
    afferents.L_th = np.sqrt(a1**2+a2**2)
    # Creating arrays of angle, rotation speed
    q = (A_q*np.cos(2*np.pi*freq*T) + np.pi/2)
    w = -A_q*np.sin(2*np.pi*freq*T)
    # Calculate muscle length
    L = np.sqrt(a1**2 + a2**2 - 2*a1*a2*np.cos(q))
    firing_L = np.where(L >= afferents.L_th)
    # Calculate momentum arm
    h = a1*a2*np.sin(q)/L
    # calc muscle speed
    v = w*h
    firing_v = np.where(v>0)
    # calc force
    F = -A_f*np.cos(2*np.pi*freq*T)
    firing_F = np.where(F >= afferents.F_th)
    # set input from motoneuron
    input = 0.1*np.random.rand(len(T))
    # calculate afferents' activity
    Ia = afferents.Ia(v, L, input)
    II = afferents.II(L, input)
    Ib = afferents.Ib(F)

    # Plotting
    plt.figure()
    
    plt.subplot(321)
    plt.plot(T, q, label="q")
    plt.plot(T, w, label='w')
    plt.plot(T, input, label='input', linewidth=0.5)
    plt.legend()
    plt.xlabel("t, seconds", loc='left')
    
    plt.subplot(325)
    plt.plot(T, F)
    plt.plot(T, afferents.F_th*np.ones(len(T)), color="red", label="F_th")
    plt.legend()
    plt.ylabel('F, N')
    plt.xlabel("t, seconds", loc="left")
    
    plt.subplot(323)
    plt.plot(T, L, label="L")
    plt.plot(T, h, label="h")
    plt.plot(T, v, label='v')
    plt.plot(T, afferents.L_th*np.ones(len(T)), color="red", label="L_th")
    plt.legend()
    plt.xlabel("t, seconds", loc='left')
    
    plt.subplot(322)
    plt.title("Ia-type activity")
    plt.plot(T, Ia, label='Ia')
    plt.xlabel("t, seconds", loc='left')
    plt.vlines(T[firing_v], np.min(Ia), np.max(Ia), color='orange', alpha=0.1, label="streching")
    plt.vlines(T[firing_L], np.min(Ia), np.max(Ia), color='yellow', alpha=0.1, label="L>=L_th")
    plt.legend()
    
    plt.subplot(324)
    plt.title('II-type activity')
    plt.plot(T, II, label="II")
    plt.xlabel('t, seconds', loc='left')
    plt.vlines(T[firing_L], np.min(II), np.max(II), color='yellow', alpha=0.1, label="L>=L_th")
    plt.legend()
    
    plt.subplot(326)
    plt.title("Ib-type activity")
    plt.plot(T, Ib, label='Ib')
    plt.ylabel('Ib')
    plt.vlines(T[firing_F], np.min(Ib), np.max(Ib), color='yellow', alpha=0.1, label="F>=F_th")
    plt.xlabel("t, seconds", loc='left')
    plt.legend()
    plt.show()


def test_all():
    # Quantity of neurons
    N = 4

    # Create neurons
    preset_list = ['RS', 'IB', 'CH', 'FS', 'TC', 'RZ', 'LTS']
    #types = [preset_list[np.random.randint(0, 7)] for i in range(N)]
    types = ['IB', 'IB', 'RS', 'IB']
    print(types)
    # Create params' array from neurons
    A, B, C, D = types2params(types)
    print(A, B, C, D)
    net = Izhikevich_Network(N=N, a=A, b=B, c=C, d=D)
    net.set_init_conditions(np.random.normal(size=N))
    # Neuron_weigths
    W = [
            [0, 0, 0, -1.1],
            [0.7, 0, 0, 0],
            [0, -1.1, 0, 0],
            [0, 0, 0.7, 0]
        ]
    net.M = np.ones((N, N))
    net.set_weights(W)
    tau_syn = np.random.randint(10, 20, (N, N))
    net.set_synaptic_relax_constant(tau_syn)
    print(net.tau_syn)
    print(net.W)
    print(net.M)
    T = np.linspace(0, 10000, 20000)
    I = np.zeros(N)
    I[0] = 5
    I[1] = 5
    input = lambda t: (I + 3*np.random.normal(size=N))*(t<3000)
    flexor = SimpleAdaptedMuscle(w = 0.5)
    extensor = SimpleAdaptedMuscle(w = 0.4)
    Limb = OneDOFLimb(q0=np.pi/2, b=0.005, a1 = 0.2, a2=0.05, m=0.3, l=0.3)
    U, V, Cn_f, X_f, F_f, Cn_e, X_e, F_e, W, Q = run(net, flexor, extensor, Limb, T, input)
    plt.subplot(221)
    for i in range(N):
        plt.plot(T, V[:, i], label=f"{net.names[i]}")
    plt.legend()
    
    plt.subplot(222)
    #plt.plot(T, W, label=r"$\dot{q}$")
    plt.plot(T, Q, label = r'$q$')
    plt.axhline(y=np.pi/2, color='red', label = r"$\pi/2$")
    plt.legend()

    plt.subplot(223)
    firing_rastr = create_firing_rastr(V, T, 30)
    plt.scatter(firing_rastr[0], firing_rastr[1], s=0.1)
    plt.yticks(list(range(N)), net.names)
    plt.subplot(4, 2, 6)
    plt.plot(T, Cn_f, label="Cn_f")
    plt.plot(T, Cn_e, label="Cn_e")
    plt.legend()
    plt.subplot(4, 2, 8)
    plt.plot(T, X_f, label="X_f")
    plt.plot(T, F_f, label="F_f")
    plt.plot(T, X_e, label="X_e")
    plt.plot(T, F_e, label="F_e")
    plt.legend()
    plt.show()

def run_pendulum(T, Limb, M):
    dt = T[1] - T[0]
    N = len(T)
    W = np.zeros(len(T))
    Q = np.zeros(len(T))
    for i in range(N):
        Q[i] = Limb.q
        W[i] = Limb.w
        Limb.step(dt=dt, M=M[i])
    return W, Q

def test_Pendulum():
    Limb = Pendulum(q0=np.pi/2-.1, b=0.01)
    T = np.linspace(0, 2000, 20000)
    M = 0.0*np.sin(T/100)
    W, Q = run_pendulum(T, Limb, M)
    plt.figure()
    plt.plot(T, Q)
    plt.plot(T, W)
    plt.show()

def run_OneDOFLimb(T, Limb, Flex, Ext):
    dt = T[1] - T[0]
    N = len(T)
    W = np.zeros(len(T))
    Q = np.zeros(len(T))
    for i in range(N):
        Q[i] = Limb.q
        W[i] = Limb.w
        Limb.step(dt=dt, F_flex=Flex[i], F_ext=Ext[i])
    return W, Q

def test_OneDOFLimb():
    Limb = OneDOFLimb(q0=np.pi/2-1, a1=7, a2=30)
    print(Limb.own_T)
    T = np.linspace(0, 2000, 20000)
    F_flex= 0*np.cos(np.pi/1000*T)
    F_ext= 0*np.sin(np.pi/1000*T)
    W, Q = run_OneDOFLimb(T, Limb, F_flex, F_ext)
    plt.figure()
    plt.plot(T, Q)
    plt.plot(T, W)
    plt.show()


def test_OneDOFLimb_withGR():
    Limb = OneDOFLimb_withGR(q0=np.pi/2, w0=-0.1, a1=7, a2=30, b=0.01)
    print(Limb.own_T)
    T = np.linspace(0, 2000, 20000)
    F_flex= 0*np.cos(np.pi/1000*T)
    F_ext= 0*np.sin(np.pi/1000*T)
    W, Q = run_OneDOFLimb(T, Limb, F_flex, F_ext)
    plt.figure()
    plt.plot(T, Q)
    plt.plot(T, W)
    plt.show()


def run_Aff_Limb(T, AL, uf, ue):
    dt = T[1] - T[0]
    M = np.zeros(len(T))
    F_f = np.zeros(len(T))
    F_e = np.zeros(len(T))
    W = np.zeros(len(T))
    Q = np.zeros(len(T))
    Output = np.zeros((len(T), 6))
    for i, t in enumerate(T):
        F_f[i] = AL.Flexor.F_prev
        F_e[i] = AL.Extensor.F_prev
        M[i] = AL.Limb.M_tot  
        Q[i] = AL.Limb.q
        W[i] = AL.Limb.w
        Output[i] = AL.output
        AL.step(dt=dt, uf=uf[i], ue=ue[i])
    return M, F_f, F_e, W, Q, Output 

def test_Afferented_Limb():
    """
    1) first of all lest generate meandr signal from scipy.signal.square
    and send it to Limb
    2) Lets generate output from simple spg and send it to system
    """ 
    flexor = SimpleAdaptedMuscle(w = 0.5, N=2)
    extensor = SimpleAdaptedMuscle(w = 0.4, N=2)
    Limb = OneDOFLimb(q0=np.pi/2, b=0.00, a1 = 0.4,
                      a2= 0.05, m=0.3, l=0.3)
    AL = Afferented_Limb(
                Limb = Limb,
                Flexor = flexor,
                Extensor = extensor
            )
    

    T = np.linspace(0, 10000, 20000)
    mod_sig = np.sin(1*np.pi*T/1000)

    uf = 15*(sig.square(2*np.pi*T/50, duty=0.2)+1)
    uf = np.where(mod_sig>0.6, uf, 0)
    ue = 15*(sig.square(2*np.pi*T/50, duty=0.2)+1)
    ue = np.where(mod_sig<-0.6, ue, 0)

    M, F_f, F_e, W, Q, Output=run_Aff_Limb(
            T, AL, uf, ue)
    Ia_f = Output[:, 0]
    Ia_e = Output[:, 3]
    II_f = Output[:, 1]
    II_e = Output[:, 4]
    Ib_f = Output[:, 2]
    Ib_e = Output[:, 5]

    # Calculate muscle length
    L_f = AL.Limb.L(Q)
    firing_Lf = np.where(L_f >= AL.Afferents.L_th)
    L_e = AL.Limb.L(np.pi-Q)
    firing_Le = np.where(L_e >= AL.Afferents.L_th)
    # Calculate momentum arm
    hf = AL.Limb.h(L_f, Q)
    he = AL.Limb.h(L_e, np.pi-Q)
    # calc muscle speed
    vf = W*hf
    ve = -W*he
    firing_vf = np.where(vf>0)
    firing_ve = np.where(ve>0)
    
    # Plotting
    plt.figure()
    
    plt.subplot(321)
    plt.title("Control signal")
    plt.plot(T, uf, label='uf', linewidth=0.5)
    plt.plot(T, ue, label='ue', linewidth=0.5)
    plt.legend()
    plt.xlabel("t, seconds", loc='left')
    
    plt.subplot(322)
    plt.title("muscles")
    plt.plot(T, F_e, label="F_ext")
    plt.plot(T, F_f, label="F_flex")
    plt.plot(T, AL.Afferents.F_th*np.ones(len(T)),
             color="red", label="F_th")
    plt.legend()
    plt.ylabel('F, N')
    plt.xlabel("t, seconds", loc="left")
    
    plt.subplot(323)
    #plt.plot(T, M, label="M")
    #plt.plot(T, hf, label="hf")
    #plt.plot(T, he, label="he")
    plt.plot(T, Q, label="q")
    plt.legend()
    plt.subplot(324)
    plt.title('Muscle length dynamics')
    plt.plot(T, L_f, label="L_f")
    #plt.plot(T, vf, label='vf')
    plt.plot(T, L_e, label="L_e")
    #plt.plot(T, vf, label='vf')
    plt.plot(T, AL.Afferents.L_th*np.ones(len(T)), color="red", label="L_th")
    plt.legend()
    plt.xlabel("t, seconds", loc='left')
    plt.show()



    plt.figure()
    
    plt.subplot(211)
    plt.title("Ia-type activity")
    m = 0
    M = 0.5
    st = 0.5
    plt.vlines(T[firing_vf], m, M, color='orange', alpha=0.5, label="f_strech")
    plt.vlines(T[firing_Lf], m+st, M+st, color='yellow', alpha=0.5,label="Lf>=L_th")
    plt.plot(T, Ia_f, label='Ia_f', color='k')
    plt.legend()

    plt.subplot(212)
    plt.vlines(T[firing_ve], m, M, color='orange', alpha=0.5, label="e_strech")
    plt.vlines(T[firing_Le], m+st, M+st, color='yellow', alpha=0.5,label='Le>=L_th')
    plt.plot(T, Ia_e, label='Ia_e', color='k')
    plt.legend()
    plt.xlabel("t, seconds", loc='left')
    plt.show()

    plt.figure()
    
    plt.subplot(211)
    plt.title("II-type activity")
    m = 0
    M = 0.1
    st = 0
    plt.vlines(T[firing_Lf], m+st, M+st, color='yellow', alpha=0.5,label="Lf>=L_th")
    plt.plot(T, II_f, label='Ia_f', color='k')
    plt.legend()

    plt.subplot(212)
    plt.vlines(T[firing_Le], m+st, M+st, color='yellow', alpha=0.5,label='Le>=L_th')
    plt.plot(T, II_e, label='Ia_e', color='k')
    plt.legend()
    plt.xlabel("t, seconds", loc='left')
    plt.show()

def test_FHN_Network():
    N = 4
    net = FizhugNagumoNetwork(N=N)
    net.a = np.array([-0.1, -0.1, 0.1, 0.1])
    net.V_th = 1*np.ones(N)
    net.ts = np.array([0.1, 0.1, 0.1, 0.1])
    net.V = 0.5 + np.random.rand(N)
    # Neuron_weigths
    net.M = np.ones((N, N))-np.eye(N)
    W = [
            [0, 0, 0, -1.1],
            [0.7, 0, 0, 0],
            [0, -1.1, 0, 0],
            [0, 0, 0.7, 0]
        ]
    net.set_weights(W)
    tau_syn = np.random.rand(N, N)
    #net.set_synaptic_relax_constant(tau_syn)
    print(net.tau_syn)
    print(net.W)
    #print(net.M)
    T = np.linspace(0, 4000, 20000)
    I = np.zeros(N)
    I[0] = 0.
    I[1] = 0.
    input = lambda t: (I + 0.01*np.random.normal(size=N))*(t<3000)
    I_aff = lambda t: 0
    U, V = run_net(T, net, input, I_aff) 
    for i in range(N):
        plt.plot(T, V[:, i], label=f"{net.names[i]}")
    plt.legend()
    plt.show()


def run(net, flexor, extensor, Limb, T, Iapp):
    """
    Running procedure of Limb with one DOF in gravity with friction
    controlled by to muscles
    Return State variables:
    U, V, Cn_f, X_f, F_f, Cn_e, X_e, F_e, W, Q 
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
    alpha_f = 1
    alpha_e = 1
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
        uf = alpha_f*net.output[0]
        ue = alpha_f*net.output[2]
        flexor.step(dt=dt, u=uf)
        extensor.step(dt=dt, u=ue)
        Limb.step(dt=dt, F_flex = flexor.F, F_ext = extensor.F)
    return U, V, Cn_f, X_f, F_f, Cn_e, X_e, F_e, W, Q 

def test_FHN_Network_with_Limb():
    
    N = 4
    net = FizhugNagumoNetwork(N=N)
    net.a = np.array([-0.1, -0.1, 0.1, 0.1])
    net.V_th = 0.1*np.ones(N)
    net.ts = np.array([0.1, 0.1, 0.1, 0.1])
    net.V = 0.5 + np.random.rand(N)
    # Neuron_weigths
    net.M = np.ones((N, N))-np.eye(N)
    W = [
            [0, 0, 0, -1.1],
            [0.7, 0, 0, 0],
            [0, -1.1, 0, 0],
            [0, 0, 0.7, 0]
        ]

    tau_syn = np.array([
            [1, 1, 1, 20],
            [1, 1, 10, 1],
            [1, 20, 1, 1],
            [10, 1, 1, 1]
        ])
    net.set_weights(W)
    net.set_synaptic_relax_constant(tau_syn)
    print(net.tau_syn)
    print(net.W)
    print(net.M)
    T = np.linspace(0, 10000, 50000)
    I = np.zeros(N)
    I[0] = 0.
    I[1] = 0.
    input = lambda t: (I + 0.01*np.random.normal(size=N))*(t<3000)
    I_aff = lambda t: 0
    flexor = SimpleAdaptedMuscle(w = 0.5, N=1)
    extensor = SimpleAdaptedMuscle(w = 0.5, N=1)
    Limb = OneDOFLimb(q0=np.pi/2, b=0.005, a1 = 0.2, a2=0.05, m=0.3, l=0.3)
    U, V, Cn_f, X_f, F_f, Cn_e, X_e, F_e, W, Q = run(net, flexor, extensor, Limb, T, input)
    plt.subplot(221)
    for i in range(N):
        plt.plot(T, V[:, i], label=f"{net.names[i]}")
    plt.legend()
    
    plt.subplot(222)
    #plt.plot(T, W, label=r"$\dot{q}$")
    plt.plot(T, Q, label = r'$q$')
    plt.axhline(y=np.pi/2, color='red', label = r"$\pi/2$")
    plt.legend()

    plt.subplot(223)
    firing_rastr = create_firing_rastr(V, T, 0.25)
    plt.scatter(firing_rastr[0], firing_rastr[1], s=0.1)
    plt.yticks(list(range(N)), net.names)
    plt.subplot(4, 2, 6)
    plt.plot(T, Cn_f, label="Cn_f")
    plt.plot(T, Cn_e, label="Cn_e")
    plt.legend()
    plt.subplot(4, 2, 8)
    plt.plot(T, X_f, label="X_f")
    plt.plot(T, F_f, label="F_f")
    plt.plot(T, X_e, label="X_e")
    plt.plot(T, F_e, label="F_e")
    plt.legend()
    plt.show()

def test_Net_Limb_connect():
    Q_app = np.array([
            [1, 0],
            [0, 1],
            [0, 0],
            [0, 0]
        ])
    print(Q_app.shape)
    Q_aff = np.array([
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0]
        ])
    P = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
    types = ['IB', 'IB', 'RS', 'IB']
    print(types)
    # Create params' array from neurons
    A, B, C, D = types2params(types)
    print(A, B, C, D)
    net = Izhikevich_IO_Network(input_size = 2, output_size = 2, 
                                afferent_size = 6,
                                N=4, Q_app = Q_app,
                                Q_aff = Q_aff , P = P)
    net.set_params(a=A, b=B, c=C, d=D) 
    # Neuron_weigths
    W = [
            [0, 0, 0, -1.1],
            [0.7, 0, 0, 0],
            [0, -1.1, 0, 0],
            [0, 0, 0.7, 0]
        ]
    
    N=4
    net.M = np.ones((N, N))
    net.set_weights(W)
    tau_syn = np.array([
            [1, 1, 1, 20],
            [1, 1, 10, 1],
            [1, 20, 1, 1],
            [10, 1, 1, 1]
        ])
    net.set_synaptic_relax_constant(tau_syn)
    print(net.tau_syn)
    print(net.W)
    print(net.M)
    
    # Limb settings
    flexor = SimpleAdaptedMuscle(w = 0.5, N=2)
    extensor = SimpleAdaptedMuscle(w = 0.4, N=2)
    Limb = OneDOFLimb(q0=np.pi/2-0.5, b=0.001, a1 = 0.4,
                      a2= 0.05, m=0.3, l=0.3)
    AL = Afferented_Limb(
                Limb = Limb,
                Flexor = flexor,
                Extensor = extensor
            )

    # Creating all system
    sys = Net_Limb_connect(Network=net,
                           Limb = AL)

    
    T = np.linspace(0, 7000, 50000)
    I = np.zeros(2)
    I[0] = 5.
    I[1] = 5
    input = lambda t: (I)*(t<3000)
    V = np.zeros((len(T), N))
    F_flex = np.zeros(len(T))
    F_ext = np.zeros(len(T))
    Afferents = np.zeros((len(T), 6))
    Q = np.zeros(len(T))
    W = np.zeros(len(T))
    dt = T[1] - T[0]
    for i, t in enumerate(T):
        V[i] = sys.net.V_prev
        F_flex[i] = sys.Limb.Flexor.F_prev
        F_ext[i] = sys.Limb.Extensor.F_prev
        Afferents[i] = sys.Limb.output
        Q[i] = sys.Limb.Limb.q
        W[i] = sys.Limb.Limb.w
        sys.step(dt = dt, Iapp = input(t))

    plt.figure()
    plt.subplot(221)
     
    for i in range(N):
        plt.plot(T, V[:, i], label=f"{sys.net.names[i]}")
    plt.legend()

    plt.subplot(222)
    plt.plot(T, F_flex, label='flexor')
    plt.plot(T, F_ext, label='extensor')
    plt.legend()

    plt.subplot(223)
    aff_types = ['Ia_f', 'II_f', 'Ib_f', 'Ia_e', 'II_e', 'Ib_f' ]
    for i in range(6):
        plt.plot(T, Afferents[:, i], label=aff_types[i])
    plt.legend()

    plt.subplot(224)
    plt.plot(T, Q, label="Q")
    plt.plot(T, W, label='W')
    plt.legend()
    plt.show()
        
        




if __name__=="__main__":
    #test_OneDOFLimb()
    #test_Pendulum()
    #test_OneDOFLimb_withGR()
    #test_all()    
    #test_IzhIOnet_step()
    #test_IzhIOnet_sym()
    #test_Afferents()
    #test_Afferented_Limb()
    #test_FHN_Network()
    #test_FHN_Network_with_Limb()
    test_Net_Limb_connect()
