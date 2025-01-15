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

def types2params(types):
    """
    types - list with types of neoruns from 
    Izhikevich presets:
        + 'RS'(regular spiking): a=0.02, b=0.2, c=-65, d=8;
        + 'IB'(intrinsically bursting): a=0.02, b-0.2, c=-55, d=4;
        + 'CH'(chattering): a=0.02, b=0.2, c=-50,d=2;
        + 'FS'(fast spiking): a=0.1, b=0.2, c=-65, d=0.05;
        + 'TC'(thalamo-cortical): a=0.02, b=0.25, c=-65, d=0.05;
        + 'RZ'(resonator): a=0.1, b=0.26, c=-65, d=8;
        + 'LTS'(low-threshold spiking): a=0.02, b=0.25, c=-65, d=2;
    return lists: a, b, c, d
    """
    N = len(types)
    Params = np.zeros((N, 4))
    for i in range(N):
        neuron = izhikevich_neuron(preset = types[i])
        Params[i] = neuron.params
        del neuron
    return Params[:, 0], Params[:, 1], Params[:, 2], Params[:, 3]

class Network:
    """
    Class Network
    Properties:
    N - quantity of neurons
    M(N, N) - mask of network connetctions if M[i, j]!=0 connection exists
    W(N, N) - matrix of synaptic weights
    tau_syn(N, N) - relaxation constants of sinaptic current
    Methods:
    connect(self, i, j, coef) - Устанавливает значение coef в маску сети между нейронами i -> j
    set_weights(W) - set weights' matrix with values from W with check network mask with current rule:
    if M[i, j] != 0 then Network.W[i, j] = W[i, j] else Network[i, j] = 0
    """
    def __init__(self, **kwargs):
        """
        args:
        N - size of network(default 10)
        M - matrix of network mask(shape:(N, N)) (default np.ones((10, 10)))
        W - sinaptic weigths(shape:(N, N)) (default np.ones(10, 10))
        TAU - relaxation times of synaptic current (shape: (N, N)) (default: np.ones((10, 10)))
        """
        self.N = kwargs.get('N', 10)
        self.M = kwargs.get('M', np.ones((self.N, self.N)))  # Маска соединений
        if self.M.shape != (self.N, self.N):
            raise Exception("Unexpected shape of M")
        self.W = kwargs.get('W', np.ones((self.N, self.N)))  # Веса связей
        if self.W.shape != (self.N, self.N):
            raise Exception("Unexpected shape of W")
        self.tau_syn = 1/kwargs.get('tau_syn', np.ones((self.N, self.N)))  # Константы
        if self.tau_syn.shape != (self.N, self.N):
            raise Exception("Unexpected shape of tau_syn")

    def __len__(self):
        return self.N
    
    def connect(self, i, j, coef, w=1, tau=10):
        coef = np.sign(coef)
        """
        Соединяет нейрон i с нейроном j, устанавливая sign(coef) в позицию (i, j) маски M.
        coef < 0 - тормозная связь
        coef > 0 - возбуждающая связь
        """
        if 0 <= i < self.N and 0 <= j < self.N:
            self.M[j, i] = coef
            self.W[j, i] = w
            self.tau_syn[j, i] = 1/tau
        else:
            raise ValueError("Индексы i и j должны быть в диапазоне от 0 до N-1")

    def set_weights(self, W):
        """
        Метод для Установки значений весов
        """
        N = self.N
        if np.shape(W) == (N, N):
            self.W = np.where(self.M != 0, W, 0)
        else:
            raise Exception(f"Excepted W with shape({N}, {N}), but shape{np.shape(W)} was sent...")
    
    def set_synaptic_relax_constant(self, relax_constant : np.ndarray):
        """
        Метод для установки времени релаксации синаптического тока
        relax_constant - матрица времен релаксации в мс
        """
        N = self.N
        if np.shape(relax_constant) == (N, N):
            self.tau_syn = 1/relax_constant;
        else:
            raise Exception(f"Excepted relax_constant matrix with shape ({N}, {N}), but shape {np.shape(relax_constant)} was sent...")

class NameNetwork(Network):
    """
    Класс: Именованая Сеть
    - Хранит методы класс Сеть
    - Хранит имена нейронов
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.names = kwargs.get('names', [f"Neuron_{i}" for i in range(self.N)])  # Имена нейронов по умолчанию
        if len(self.names) != self.N:
            raise Exception("Unexpected len of names")

    def set_name(self, i, name):
        """
        Устанавливает имя для нейрона i.
        """
        if 0 <= i < self.N:
            self.names[i] = name
        else:
            raise ValueError("Индекс i должен быть в диапазоне от 0 до N-1")

    def print_names(self):
        """
        Выводит имена нейронов в сети и их индексы в списке неронов
        Можно использовать для определения номера нейрона по имени
        """
        for i in range(self.N):
            print(f"[{i} : {self.names[i]}]")

    def print_connections(self):
        """
        Выводит имена нейронов в формате "пресинапс -> постсинапс".
        """
        for i in range(self.N):
            for j in range(self.N):
                if self.M[j, i] != 0:
                    print(f"{self.names[i]} -> {self.names[j]} type: {self.M[j, i]}")

# Необходимо дописать класс для определения параметров входа и параметров выхода...
class Izhikevich_Network(NameNetwork):
    """
    Network of Izhikevich neurons
    """
    def __init__(self, **kwargs):
        """
        Params:
        Inherit from Name network
        keyword arguments:
        a, b, c, d - Izhikevich constants
        default: a = 0.1, b=0.2, c=-65, d=4
        """
        super().__init__(**kwargs)
        self.a = kwargs.get('a', 0.1 * np.ones(self.N))
        self.b = kwargs.get('b', 0.2 * np.ones(self.N))
        self.c = kwargs.get('c', -65 * np.ones(self.N))
        self.d = kwargs.get('d', 4 * np.ones(self.N))
        self.V_peak = 30
        self.I_syn=np.zeros((self.N, self.N))
        self.output = np.zeros(self.N)
        self.V = self.c
        self.U = self.c*self.b
        self.U_prev = self.U
        self.V_prev = self.V

    def set_params(self, **kwags):
        """
        Method to change Izhikevich params
        keyword arguments:
        a, b, c, d
        """
        a = kwargs.get('a', self.a)
        b = kwargs.get('b', self.b)
        c = kwargs.get('c', self.c)
        d = kwargs.get('d', self.d)
        
        if len(a) == N:
            self.a = a
        else:
            raise Exception(f"Excepted a with size {N}, but size {len(a)} was sent...")

        if len(b) == N:
            self.b = b
        else:
            raise Exception(f"Excepted b with size {N}, but size {len(b)} was sent...")
        
        if len(c) == N:
            self.c = c
        else:
            raise Exception(f"Excepted c with size {N}, but size {len(c)} was sent...")
        
        if len(d) == N:
            self.d = d
        else:
            raise Exception(f"Excepted d with size {N}, but size {len(d)} was sent...")

	
    def run_state(self, U, V, I_syn, I_app):
        dVdt = 0.04*np.power(V, 2) + 5*V + 140 - U + I_app + np.sum(I_syn, axis=1);
        dUdt = self.a*(self.b*V - U);
        return (dVdt, dUdt)
    
    def set_init_conditions(self):
        self.V = self.c
        self.U = self.c*self.b
        self.U_prev = self.U
        self.V_prev = self.V

    def step(self, dt = 0.1, Iapp=0):
        dVdt, dUdt = self.run_state(self.U_prev, self.V_prev, self.I_syn, Iapp)
        self.V = np.where(self.V_prev >= self.V_peak, self.c, self.V_prev + dt*dVdt)
        self.U = np.where(self.V_prev >= self.V_peak, self.U + self.d, self.U_prev + dt*dUdt)
        # synaptic dynamics must be rewritung in own class
        dI_syn = dt * (-self.I_syn * self.tau_syn + self.W * (np.where(self.V_prev >= self.V_peak, self.V_peak , 0)))
        self.I_syn += dI_syn
        self.V_prev = np.where(self.V >= self.V_peak, self.V_peak + 1, self.V)
        self.U_prev = self.U
        self.output = np.where(self.V >= self.V_peak, self.V_peak, 0)

class Izhikevich_IO_Network(Izhikevich_Network):
    """
    Inherit all properties and methods from Izhikevich network
    But It has properties like matrix of input and matrix of output
    """
    def __init__(self, **kwargs):
        """
        Has arguments from Izhikevich_Network cunstructor
        input_size - dimension of input
        N - dimension of state
        output_size - dimension of output
        Q_app - matrix(input_size, N) of input current
        Q_aff - matrix(input_size, N) of afferent current
        P - matrix(N, output_size) of output
        """
        super().__init__(**kwargs);
        self.input_size = kwargs.get('input_size', self.N)
        self.afferent_size = kwargs.get('afferent_size', self.N)
        self.output_size = kwargs.get('output_size', self.N)
        
        self.Q_app = kwargs.get('Q_app', np.ones((self.input_size, self.N)))
        if np.shape(self.Q_app) != (self.N, self.input_size):
            raise Exception (f"Applicatian matrix Q_app must have wrong shape ({self.N}, {self.input_size})")
        
        self.Q_aff = kwargs.get('Q_aff', np.ones((self.afferent_size, self.N)))
        if np.shape(self.Q_aff) != (self.N, self.input_size):
            raise Exception (f"Applicatian matrix Q_aff must have wrong shape ({self.N}, {self.input_size})")
        
        self.P = kwargs.get('P', np.ones((self.N, self.output_size)))
        if np.shape(self.P) != (self.output_size, self.N):
            raise Exception (f"Output matrix P must have wrong shape ({self.output_size}, {self.N})")
    
    def step(self, dt = 0.1, I_app = 0, I_aff = 0):
        I_in = self.Q_app.dot(I_app) + self.Q_aff.dot(I_aff)
        super().step(dt=0.1, Iapp=I_in)
        self.V_out = self.P.dot(self.output)
        

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
    net = Izhikevich_IO_Network(input_size=2, output_size=2, N=4, Q_app=Q_app,
                                Q_aff = Q_aff , P=P)
    I_app = np.array([0, 0])
    I_aff = np.zeros(2)
    net.step(dt=0.1, I_app=I_app, I_aff=I_aff)

def run_net(T, net, I_app, I_aff):
    dt = T[1] - T[0]
    net.set_init_conditions()
    N = len(net)
    U = np.zeros((len(T), N))
    V = np.zeros((len(T), N))
    for i, t in enumerate(T):
        U[i] = net.U_prev
        V[i] = net.V_prev
        net.step(dt=dt, I_app = I_app(t), I_aff = I_aff(t))
    return U, V

def test_IzhIOnet_sym():
    Q_app = np.array([
            [1, 0],
            [0, 1],
            [0, 0],
            [0, 0]
        ])
    Q_aff = np.array([
            [0, 0],
            [0, 0],
            [0, 1],
            [1, 0]
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
                                afferent_size=2, N=4, Q_app=Q_app,
                                Q_aff = Q_aff , P=P, W=W)
    net.set_synaptic_relax_constant(tau_syn)
    T = np.linspace(0, 500, 2000)
    I_app = lambda t: np.array([5, 7]) + 2*np.random.rand(2)
    I_aff = lambda t: np.array([0, 0]) + 2*np.random.rand(2)
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
    

def create_firing_rastr(V, T, V_peak):
    """
Create Spikes from time dependency for all neurons
    """
    firing_idx = np.where(V>V_peak)
    return (T[firing_idx[0]], firing_idx[1])

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
        w - weight of neuron-muscle synapse
        """
        self.w = kwargs.get('w', 0.5)
        self.Cn = 0
        self.Cn_prev = 0
        self.F = 0
        self.F_prev = 0
        self.x = 0

    def step(self, dt = 0.1, u=0):
        self.Cn = self.Cn_prev + dt*(self.w*u - self.Cn_prev*self.tau_c)
        self.x = self.Cn**self.m/(self.Cn**self.m + self.k**self.m)
        self.F = self.F_prev + dt*(self.A*self.x - self.F_prev*self.tau_1)
        self.F_prev = self.F
        self.Cn_prev = self.Cn

class Afferents:
    """
    Class describes activity of afferents' axons by Prohazka equetions
    Afferents generate impulses aproximated by output current.
    Idea borrowed from article:
    doi: 10.1111/j.1749-6632.2010.05435.x
    'Afferent control of locomotor CPG: insights from a simple
neuromechanical model' by Markin S. et. all.
    """
    p_v = 0.6
    k_v = 6.2
    k_dI = 2
    k_dII = 1.5
    k_nI = 0.06
    k_nII = 0.06
    k_f = 1
    L_th = 0.059 #m
    F_th = 3.38# N
    const_I = 0
    const_II = 0
    def __init__(self):
        pass

    def Ia(self, v, L, input):
        """
        Calculate Ia-axon type activity by formula:
        Ia = k_v*v_norm^p_v + k_dI*d_norm + k_nI*input + const_I
        arguments:
        v - muscle velocity
        L - muscle lenght
        input - motoneuron's activity
        """
        v_norm = np.where(v>=0, v/self.L_th, 0)
        d_norm = np.where(L>=self.L_th, (L-self.L_th)/self.L_th, 0)
        Ia = self.k_v*v_norm**self.p_v + self.k_dI*d_norm + self.k_nI*input + self.const_I
        return Ia

    def Ib(self, F):
        """
        Calculate Ib-axon activity by formula:
        Ib = k_f*F_norm
        F_norm - normilized muscle force
        aruments:
        F - muscle force
        """
        F_norm = np.where(F>=self.F_th, (F-self.F_th)/self.F_th, 0)
        return self.k_f*F_norm

    def II(self, L, input):
        """
        Calculate II-axon activity by formula:
        II = k_dII*d_norm + k_nII*input + const_II
        d_norm - normilized muscle lenght
        const_II - default activity
        arguments:
        L - muscle length
        input - activity of motoneuron
        """
        d_norm = np.where(L>=self.L_th, (L-self.L_th)/self.L_th, 0)
        return self.k_dII*d_norm + self.k_nII*input + self.const_II

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

class Pendulum:
    """
    Pendulum class
    params:
    m - mass(kg)
    ls - length(m)
    b - angular viscosity(kg*m^2/(ms*rad))
    q0 - initial angle
    w0 - initial rotation
    """
    g = 9.81#m/s^2
    def __init__(self, m=0.3, ls=0.3, b=0.002, **kwargs):
        self.m = m
        self.ls = ls
        self.J = m*ls**2/3 # Inertia moment
        self.b = b
        self.q = kwargs.get('q0', np.pi/2)
        self.w = kwargs.get('w0', 0)
        self.q_prev = self.q
        self.w_prev = self.w
        self.own_T = 2*np.pi*np.sqrt(2*ls/(3*self.g))


    def step(self, dt = 0.1, M = 0):
        """
        Calculation of simulation step
        dt - discreate time step (ms)
        M - applied momentum (N*m)
        """
        self.w = self.w_prev + 0.001*dt*(0.5*self.g*self.m*self.ls*np.cos(self.q_prev) - self.b*self.w_prev + M)/self.J
        self.q = self.q_prev + 0.001*dt*(self.w_prev)
        self.q_prev = self.q
        self.w_prev = self.w

class OneDOFLimb(Pendulum):
    """
    Class of 1 degree of freedom Limb 
    inherit from Pendudulum
    calculate Forces' momentum from mucles
    """
    def __init__(self, **kwargs):
        """
        Init params:
        From Pendulum:
        m - mass(g)
        ls - length(mm)
        b - angular viscosity(g*mm^2/(ms*rad))
        q0 - initial angle
        w0 - initial rotation
        a1, a2 - tendon attachment points(mm)
        """
        super().__init__(**kwargs)
        self.a1 = kwargs.get('a1', 0.06)
        self.a2 = kwargs.get('a2', 0.007)

    def L(self, q):
        """
        calculation muscle length from q
        """
        return np.sqrt(self.a1**2 + self.a2**2 - 2*self.a1*self.a2*np.cos(q))

    def h(self, L, q):
        return self.a1*self.a2*np.sin(q)/L

    def step(self, dt=0.1, F_flex = 0, F_ext = 0, M=0):
        """
        Input Forces:
        Flexor: F_flex(N)
        Extensor: F_ext(N)
        """
        L_flex = self.L(self.q)
        L_ext = self.L(np.pi-self.q)
        h_flex = self.h(L_flex, self.q)
        h_ext = self.h(L_ext, np.pi-self.q)
        M_tot = F_flex*h_flex - F_ext*h_ext + M
        super().step(dt=dt, M=M_tot)

class OneDOFLimb_withGR(OneDOFLimb):
    """
    Inherit all properties from OneDOFLimb,
    but have groud reaction force in Newton equetions
    """
    M_GRmax = 0.585 # N*m
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def GR(self, w, q):
        return np.where(
                w >= 0, # stance phase
                - self.M_GRmax*np.cos(q),
                0 # swing phase
                )
         
    def step(self, dt=0.1, F_flex = 0, F_ext = 0):
        """
        Input Forces:
        Flexor: F_flex(N)
        Extensor: F_ext(N)
        """
        super().step(dt=dt, F_flex=F_flex, F_ext=F_ext,
                     M = self.GR(self.w, self.q))



        


from matplotlib import pyplot as plt


def run(net, flexor, extensor, Limb, T, Iapp):
    dt = T[1] - T[0]
    net.set_init_conditions()
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
        uf = 0#net.output[-2]
        ue = net.output[-1]
        flexor.step(dt=dt, u=uf)
        extensor.step(dt=dt, u=ue)
        Limb.step(dt=dt, F_flex = flexor.F, F_ext = extensor.F)
    return U, V, Cn_f, X_f, F_f, Cn_e, X_e, F_e, W, Q 

def test_all():
    # Quantity of neurons
    N = 4

    # Create neurons
    preset_list = ['RS', 'IB', 'CH', 'FS', 'TC', 'RZ', 'LTS']
    types = [preset_list[np.random.randint(0, 7)] for i in range(N)]
    print(types)
    # Create params' array from neurons
    A, B, C, D = types2params(types)
    print(A, B, C, D)
    net = Izhikevich_Network(N=N, a=A, b=B, c=C, d=D)
    
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
    T = np.linspace(0, 20000, 50000)
    I = np.zeros(N)
    I[0] = 5
    I[1] = 5
    input = lambda t: (I + 3*np.random.rand(N))*(t<1000)
    flexor = SimpleAdaptedMuscle(w = 1)
    extensor = SimpleAdaptedMuscle(w = 1)
    Limb = OneDOFLimb(q0=np.pi/2, b=0.005, a1 = 0.2, a2=0.05, m=0.3, l=0.3)
    U, V, Cn_f, X_f, F_f, Cn_e, X_e, F_e, W, Q = run(net, flexor, extensor, Limb, T, input)
    plt.subplot(221)
    for i in range(N):
        plt.plot(T, V[:, i], label=f"{net.names[i]}")
    plt.legend()
    
    plt.subplot(222)
    plt.plot(T, W, label=r"$\dot{q}$")
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

if __name__=="__main__":
    #test_OneDOFLimb()
    #test_Pendulum()
    #test_OneDOFLimb_withGR()
    #test_all()    
    #test_IzhIOnet()
    #test_IzhIOnet_sym()
    test_Afferents()
