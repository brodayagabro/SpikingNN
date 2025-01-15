from Izh_net import *
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ["XDG_SESSION_TYPE"] = "xcb"

def create_markin_net():
    # Create neurons
    # Set neurons' names
    names = ['RG_F', 'RG_E', 'PF_F', 'PF_E', 'IN_F', 'IN_E', 'IN_f', 'Inab_E', 'MN_F', 'MN_E']
    types = ['CH', 'CH', 'RS', 'RS', 'FS', 'FS', 'FS', 'FS', 'FS', 'FS']
    # Create params' array from neurons
    Params = np.zeros((10, 4))
    for i in range(10):
        neuron = izhikevich_neuron(preset = types[i])
        Params[i] = neuron.params
        del neuron
    print(Params)
    Markin_net = Izhikevich_Network(N=10)
    # Set Neuron params
    Params[0, 0] = 0.001
    Params[1, 0] = 0.001
    Params[0, 1] = 0.46
    Params[1, 1] = 0.46
    Params[0, 2] = -45
    Params[1, 2] = -45
    Params[0, 3] = 2
    Params[1, 3] = 2
    Markin_net.set_a(Params[:, 0])
    print(Markin_net.a)
    Markin_net.set_b(Params[:, 1])
    print(Markin_net.b)
    Markin_net.set_c(Params[:, 2])
    print(Markin_net.c)
    Markin_net.set_d(Params[:, 3])
    print(Markin_net.d)
    # Connect Neurons
    for i in range(10):
        Markin_net.set_name(i, names[i])
    Markin_net.print_names()
    Markin_net.connect(0, 2, 1, w=1, tau=10)
    Markin_net.connect(1, 3, 1, w=1, tau=10)
    Markin_net.connect(5, 0, -1, w=-1.1, tau=20)
    Markin_net.connect(5, 2, -1, w=-1.1, tau=20)
    Markin_net.connect(4, 1, -1, w=-1.1, tau=20)
    Markin_net.connect(4, 3, -1, w=-1.1, tau=20)
    Markin_net.connect(5, 6, -1, w=-1.1, tau=20)
    Markin_net.connect(6, 7, -1, w=-1.1, tau=20)
    #Markin_net.connect(7, 9, -1, w=0, tau=10)
    Markin_net.connect(2, 8, 1, w=1, tau=20)
    Markin_net.connect(3, 9, 1, w=1, tau=20)
    Markin_net.connect(3, 7, 1, w=1, tau=20)
    Markin_net.connect(0, 4, 1, w=1, tau=20)
    Markin_net.connect(1, 5, 1, w=1, tau=20)
    Markin_net.print_connections()
    #W = 2*(np.ones((10, 10)))
    #W *= Markin_net.M
    #TAU = 10*np.ones((10, 10))
    #print(W)
    #Markin_net.set_weights(W)
    #Markin_net.set_synaptic_relax_constant(TAU)
    print(Markin_net.W)
    print(Markin_net.tau_syn)
    Markin_net.set_init_conditions()
    return Markin_net

def run(net, flexor, extensor, Limb, T, Iapp):
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
        uf = net.output[-2]
        ue = net.output[-1]
        flexor.step(dt=dt, u=uf)
        extensor.step(dt=dt, u=ue)
        Limb.step(dt=dt, F_flex = flexor.F, F_ext = extensor.F)
    return U, V, Cn_f, X_f, F_f, Cn_e, X_e, F_e, W, Q 

        

def create_firing_rastr(V, T, V_peak):
    firing_idx = np.where(V>V_peak)
    return (T[firing_idx[0]], firing_idx[1])

from matplotlib import pyplot as plt

if __name__=="__main__":
    net = create_markin_net()
    T = np.linspace(0, 20000, 40000)
    N = net.N
    I = np.zeros(N)
    I[0] = 0
    I[1] = 0
    I[-1] = 0
    input = lambda t: (I + 2*np.random.rand(N))*(t<1000)
    flexor = SimpleAdaptedMuscle(w=0.00)
    extensor = SimpleAdaptedMuscle(w=0.00)
    Limb = OneDOFLimb(q0=np.pi/2-0.1, b=0.001)
    U, V, Cn_f, X_f, F_f, Cn_e, X_e, F_e, W, Q = run(net, flexor, extensor, Limb, T, input)
    plt.subplot(221)
    for i in range(N):
        plt.plot(T, V[:, i], label=f"{net.names[i]}")
    plt.legend()
    
    plt.subplot(222)
    #plt.plot(T, W, label=r"$\cdot{q}$")
    plt.plot(T, Q, label = r'$q$')
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
    #plt.plot(T, X_f, label="X_f")
    plt.plot(T, F_f, label="F_f")
    #plt.plot(T, X_e, label="X_e")
    plt.plot(T, F_e, label="F_e")
    plt.legend()
    plt.show()

"""
    T = np.linspace(0, 2000, 20000)
    W, Q = run_pendulum(T)
    plt.figure()
    plt.plot(T, Q)
    plt.plot(T, W)
    plt.show()
"""
