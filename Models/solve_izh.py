import numpy as np

class Network:
    """
        class to describe typical network, synaptic connection
        Size - counts of neurons in this network
        Links - matrix of weight in this network
        Tau - matrix of relax time of synaptic current
        Synapses model: dIsyn/dt = W*v[spike] - Isyn/tau
    """
    def __init__(self, size=1, W=[], T=[]):
        self._N = size;
        self._W = np.array(W);
        self._T = np.array(T);

    @property
    def N(self):
        return self._N
    @property
    def W(self):
        return self._W
    @property
    def T(self):
        return self._T

class Izhikevich_Network(Network):
    def __init__(self, size = 1, W=[], T=[], **kwargs):
        Network.__init__(self, size=size, W=W, T=T);
        self._params = {
                "a": None,
                "b": None,
                "c": None,
                "d": None,
                "vpeak":None
                }
        for key, val in kwargs.items():
            if len(val) != size:
                raise Exception(f"size({key}) != {size}")
            self._params[key] = np.array(val)

    @property
    def params(self):
        return self._params

    @property 
    def a(self):
        return self._params["a"]
    @property
    def b(self):
        return self._params["b"]
    @property
    def c(self):
        return self._params["c"]

    @property
    def d(self):
        return self._params["d"]
    @property
    def v_peak(self):
        return self._params["v_peak"];
         


if __name__=="__main__":
    # Network testing
    net = Network(size=1, W=[1], T=[10])
    assert(net.N==1);
    assert(net.W==[1]);
    assert(net.T==[10]);
    # Izhikevich_Network testing
    Izhnet = Izhikevich_Network(size=1, W=[1], T=[10], a=[10], b=[13],
            c=[13], d=[10], v_peak = [30]);
    assert(Izhnet.N == 1);
    assert(Izhnet.W == [1]);
    assert(Izhnet.T == [10]);
    assert(Izhnet.a == [10]);
    assert(Izhnet.b == [13]);
    assert(Izhnet.c == [13]);
    assert(Izhnet.d == [10]);
    assert(Izhnet.v_peak == [30]);

def solve_izh(T, I, network):
    """
        Method to solve Izhikevich network activity in time use Euler method
        T - simulation time,
        I - amount of stymulation current
        network - network object describing activity 
    """
    Neurons = network.N # count of neurons in network
    index = np.arange(Neurons) # neuron neuron_indexes
    
    #***********************************************#
    # Euler method
    dt = T[1]-T[0]
    N = len(T) # count of time iterations 
    v = np.zeros((N, Neurons)) # membran potential
    u = np.zeros((N, Neurons)) # helphes variable in Izhikevich model
    I_syn = np.zeros((N, Neurons))
    v_peak = network.v_peak # array of threshold
    ## simulation params
    a = network.a
    b = network.b
    c = network.c
    d = network.d
    W = network.W
    Tau = network.T
    # initial conditions
    v[0] = c
    u[0] = b*c
    I_syn[0] = np.array([0, 2])
    #***********************************************#
    # iterations
    for n in range(0, N-1):
        t = T[n]
        v[n+1] = v[n] + dt*(0.04*np.power(v[n], 2) + 5*v[n] + 140 - u[n] + I[n] + I_syn[n]);
        u[n+1] = u[n] + dt*a*(b*v[n] - u[n]);
        I_syn[n+1] = I_syn[n] - dt*(Tau.dot(I_syn[n]));

        #finding spikes
        fired = v[n+1] > v_peak
        if fired.any():
            print(T[n])
            # reset when spikes
            v[n][fired] = v_peak[fired]
            v[n+1][fired] = c[fired]
            u[n+1][fired] = u[n][fired] + d[fired]
            I_syn[n+1][~fired] += W.dot(v[n])[~fired]
    return (v, u, I_syn)

    

