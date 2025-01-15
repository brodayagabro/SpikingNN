from Izh_net import *
import matplotlib.pyplot as plt
# Running procedure

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
        uf = net.output[0]
        ue = net.output[2]
        flexor.step(dt=dt, u=uf)
        extensor.step(dt=dt, u=ue)
        Limb.step(dt=dt, F_flex = flexor.F, F_ext = extensor.F)
    return U, V, Cn_f, X_f, F_f, Cn_e, X_e, F_e, W, Q 

# Creating network
N = 4
a = 0.001
b = 0.46
c = -45
d = 2
A = a*np.ones(N)
A[1] = 0.1; A[3] = 0.1 
B = b*np.ones(N)
B[1] = 0.2; B[3] = 0.2
C = c*np.ones(N)
C[1] = -65; C[3] = -65
D = d*np.ones(N)
D[1] = 0.05; D[3] = 0.05
#net.set_a(A)
#net.set_b(B)
#net.set_c(C)
#net.set_d(D)

net = Izhikevich_Network(N=N, a=A, b=B, c=C, d=D)
net.set_init_conditions()
net.set_weights(np.array([
        [0, 0, 0, -1.1],
        [0.7, 0, 0, 0],
        [0, -1.1, 0, 0],
        [0, 0, 0.7, 0]
    ]))
net.set_synaptic_relax_constant(
        np.array(
            [
                [1, 1, 1, 40],
                [20, 1, 1, 1],
                [1, 40, 1, 1],
                [1, 1, 20, 1]
                ]
            )
        )
# Limb settings
flexor = SimpleAdaptedMuscle(w=1)
extensor = SimpleAdaptedMuscle(w=1)
Limb = OneDOFLimb(q0=np.pi/2, w0=0, a1=0.2, a2=0.07, m=0.3, ls=0.3, b=0.001,)

# Inupt
I = np.zeros(N)
I[0] = 0
I[-1] = 0
input = lambda t: I + 2*np.random.rand(N)

#simulation
T = np.linspace(0, 6000, 12000)
U, V, Cn_f, X_f, F_f, Cn_e, X_e, F_e, W, Q = run(net, flexor, extensor, Limb, T, input)

#visualization
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
