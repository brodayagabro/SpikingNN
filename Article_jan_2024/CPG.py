import numpy as np


class izhikevich_neuron:
    
    """
    Izhikevich neuron class 
    - save info about single Izhikevich neuron in group
    """
    def __init__(self, **kwargs):
        """
        keyword params:
        - id, name - from neuron;
        - preset - Neocortical neurons in the mammalian brain can be classified into
        several types according to the pattern of spiking and bursting seen in
        intracellular recordings. All excitatory cortical cells are divided into the
        following classes:
            + 'RS'(regular spiking): a=0.02, b=0.2, c=-65, d=8;
            + 'IB'(intrinsically bursting): a=0.02, b-0.2, c=-55, d=4;
            + 'CH'(chattering): a=0.02, b=0.2, c=-50,d=2;
            + 'FS'(fast spiking): a=0.1, b=0.2, c=-65, d=0.05;
            + 'TC'(thalamo-cortical): a=0.02, b=0.25, c=-65, d=0.05;
            + 'RZ'(resonator): a=0.1, b=0.26, c=-65, d=8;
            + 'LTS'(low-threshold spiking): a=0.02, b=0.25, c=-65, d=2;
            example: preset = 'RS' or set None to use another parametrs

        - a=0.02, b=0.2, c=-65, d=2 - standart parameters of Izhikevich neuron(when preset=None);
        - ap_threshold - threshold voltage(default 30mV)
        """
        preset_list = ['RS', 'IB', 'CH', 'FS', 'TC', 'RZ', 'LTS', None]
        preset = kwargs.get('preset', None)
        self.ap_threshold = kwargs.get('ap_threshold', 30)
        param_list = [
                [0.02, 0.2, -65, 8],
		[0.02, 0.2, -55, 4],
		[0.02, 0.2, -50, 2],
	        [0.1, 0.2, -65, 2],
		[0.02, 0.25, -65, 0.05],
		[0.1, 0.26, -65, 8],
		[0.02, 0.25, -65, 2],
		[
                    kwargs.get('a', .02),
                    kwargs.get('b', .2),
                    kwargs.get('c', -65.0),
                    kwargs.get('d', 2.0)
                ]
            ]
        idx = preset_list.index(preset)
        assert preset in preset_list,f'Preset {preset} does not exist! Use one from {preset_list}'
        self.params = param_list[idx]


class Izhikevich_Network:

    def __init__(self, N=10, **kwargs):
        self.N = N
        self.a = 0.02 * np.ones(N)
        self.b = 0.2 * np.ones(N)
        self.c = -65 * np.ones(N)
        self.d = 2 * np.ones(N)
        self.V_peak = 30
        self.tau_syn = np.ones((N, N))
        self.W = np.random.rand(N, N) - 0.5
        self.I_syn=np.zeros((N, N))
        self.output = np.zeros(N)
        self.C = kwargs.get("C", np.eye(N))
        self.A = kwargs.get("A", np.eye(N))


    def __len__(self):
        return self.N

    def set_a(self, a):
        N = self.N
        if len(a) == N:
            self.a = a
        else:
            raise Exception(f"Excepted a with size {N}, but size {len(a)} was sent...")

    def set_b(self, b):
        N = self.N
        if len(b) == N:
            self.b = b
        else:
            raise Exception(f"Excepted b with size {N}, but size {len(b)} was sent...")

    def set_c(self, c):
        N = self.N
        if len(c) == N:
            self.c = c
        else:
            raise Exception(f"Excepted c with size {N}, but size {len(c)} was sent...")

    def set_d(self, d):
        N = self.N
        if len(d) == N:
            self.d = d
        else:
            raise Exception(f"Excepted d with size {N}, but size {len(d)} was sent...")

    def set_weights(self, W):
        N = self.N
        if np.shape(W) == (N, N):
            self.W = W
        else:
            raise Exception(f"Excepted W with shape({N}, {N}), but shape{np.shape(W)} was sent...")
    
    def set_synaptic_relax_constant(self, relax_constant):
        N = self.N
        if np.shape(relax_constant) == (N, N):
            self.tau_syn = 1/relax_constant;
        else:
            raise Exception(f"Excepted relax_constant matrix with shape ({N}, {N}), but shape {np.shape(relax_constant)} was sent...")
	
    def run_state(self, U, V, I_app, I_syn):
        dVdt = 0.04*np.power(V, 2) + 5*V + 140 - U + I_app + np.sum(I_syn, axis=1);
        dUdt = self.a*(self.b*V - U);
        return (dVdt, dUdt)
    
    def set_init_conditions(self):
        self.V = self.c
        self.U = self.c*self.b
        self.U_prev = self.U
        self.V_prev = self.V

    def step(self, dt = 0.1, Iapp = 0, Iaff = None):
        dVdt, dUdt = self.run_state(self.U_prev, self.V_prev, Iapp, self.I_syn)
        self.V = np.where(self.V_prev >= self.V_peak, self.c, self.V_prev + dt*dVdt)
        self.U = np.where(self.V_prev >= self.V_peak, self.U_prev + self.d, self.U_prev + dt*dUdt)
        dI_syn = dt * (-self.I_syn * self.tau_syn + self.W * (np.where(self.V_prev >= self.V_peak, self.V_peak , 0)))
        self.I_syn += dI_syn
        self.V_prev = np.where(self.V >= self.V_peak, self.V_peak + 1, self.V)
        self.U_prev = self.U
        self.output = np.where(self.V >= self.V_peak, self.V_peak, 0)


     

class SimpleAdaptedMuscle:
    """
    Equetions:
    -- Muscle synapse
    dCn(t)/dt + Cn(t)/tau_c = u(t), u - input
    x(t) = Cn^m/(Cn^m + k^m)
    -- Output Force
    dF(t)/dt + F(t)/tau_1 = Ax(t)
    """
    tau_c = 1/71 # 1/ms
    tau_1 = 1/130 # 1/ms
    m = 2.5
    k = 0.75
    A = 0.074 # 1/ms

    def __init__(self, **kwargs):
        """
        Arguments:
        l - init muscle length (meters)
        """
        self.l = kwargs.get('l', 0.05) # init muscle length in m 
        self.Cn = 0
        self.Cn_prev = 0
        self.F = 0
        self.F_prev = 0
        self.x = 0

    def step(self, dt = 0.1, u=0):
        self.Cn = self.Cn_prev + dt*(u - self.Cn_prev*self.tau_c)
        self.x = self.Cn**self.m/(self.Cn**self.m + self.k**self.m)
        self.F = self.F_prev + dt*(self.A*self.x - self.F_prev*self.tau_1)
        self.F_prev = self.F
        self.Cn_prev = self.Cn



def run(net, flexor, extensor, T, Iapp):
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
    for i, t in enumerate(T):
        U[i] = net.U_prev
        V[i] = net.V_prev
        Cn_f[i] = flexor.Cn_prev
        X_f[i] = flexor.x
        F_f[i] = flexor.F_prev
        Cn_e[i] = extensor.Cn_prev
        X_e[i] = extensor.x
        F_e[i] = extensor.F_prev
        net.step(dt=dt, Iapp=Iapp(t))
        uf = net.output[0]
        ue = net.output[2]
        flexor.step(dt=dt, u=uf)
        extensor.step(dt=dt, u=ue)
    return U, V, Cn_f, X_f, F_f, Cn_e, X_e, F_e 
        

def create_firing_rastr(V, T, V_peak):
    firing_idx = np.where(V>V_peak)
    return (T[firing_idx[0]], firing_idx[1])

from matplotlib import pyplot as plt

if __name__=="__main__":
    # Creating network
    N = 4
    net = Izhikevich_Network(N=N)
    net.set_init_conditions()
    a = 0.001
    b = 0.46
    c = -45
    d = 2
    N1 = izhikevich_neuron(preset=None, a = a, b = b, c = c, d=d) 
    N2 = izhikevich_neuron(preset=None, a = a, b = b, c = c, d=d)
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
    A = a*np.ones(N)
    A[1] = 0.1; A[3] = 0.1 
    B = b*np.ones(N)
    B[1] = 0.2; B[3] = 0.2
    C = c*np.ones(N)
    C[1] = -65; C[3] = -65
    D = d*np.ones(N)
    D[1] = 0.05; D[3] = 0.05
    net.set_a(A)
    net.set_b(B)
    net.set_c(C)
    net.set_d(D)
    
    I = np.zeros(N)
    I[0] = 0
    I[-1] = 0
    input = lambda t: I + 2*np.random.rand(N)
    flexor = SimpleAdaptedMuscle()
    extensor = SimpleAdaptedMuscle()
    T = np.linspace(0, 6000, 12000)
    U, V, Cn_f, X_f, F_f, Cn_e, X_e, F_e = run(net, flexor, extensor, T, input)
    plt.subplot(211)
    for i in range(N):
        plt.plot(T, V[:, i], label=f"{i}")
    plt.legend()
    plt.subplot(223)
    firing_rastr = create_firing_rastr(V, T, 30)
    plt.scatter(firing_rastr[0], firing_rastr[1], s=0.1)
    plt.subplot(4, 2, 6)
    plt.plot(T, Cn_f, label="Cn_flex")
    plt.plot(T, Cn_e, label="Cn_ext")
    plt.legend()
    plt.subplot(4, 2, 8)
    plt.plot(T, X_f, label="X_flex")
    plt.plot(T, F_f, label="F_flex")
    plt.plot(T, X_e, label="X_ext")
    plt.plot(T, F_e, label="F_ext")
    plt.legend()

    plt.show()


