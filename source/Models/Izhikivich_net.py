import numpy as np
class Izhikevich_Network:

    def __init__(self, N=10):
        self._N = N
        self._a = 0.1 * np.ones(N)
        self._b = 0.2 * np.ones(N)
        self._c = -65 * np.ones(N)
        self._d = 4 * np.ones(N)
        self.V_peak = 30
        self._tau_syn = np.ones((N, N))
        self._W = np.random.rand(N, N) - 0.5
        self.input = lambda t: np.ones(N)
    
    @property
    def a(self):
        return self._a
    
    @property
    def b(self):
        return self._b
    
    @property
    def c(self):
        return self._c
    
    @property
    def d(self):
        return self._d

    def set_a(self, a):
        N = self._N
        if len(a) == N:
            self._a = a
        else:
            raise Exception(f"Excepted a with size {N}, but size {len(a)} was sent...")

    def set_b(self, b):
        N = self._N
        if len(b) == N:
            self._b = b
        else:
            raise Exception(f"Excepted b with size {N}, but size {len(b)} was sent...")

    def set_c(self, c):
        N = self._N
        if len(c) == N:
            self._c = c
        else:
            raise Exception(f"Excepted c with size {N}, but size {len(c)} was sent...")

    def set_d(self, d):
        N = self._N
        if len(d) == N:
            self._d = d
        else:
            raise Exception(f"Excepted d with size {N}, but size {len(d)} was sent...")

    def set_weights(self, W):
        N = self._N
        if np.shape(W) == (N, N):
            self._W = W
        else:
            raise Exception(f"Excepted W with shape({N}, {N}), but shape{np.shape(W)} was sent...")
    
    def set_synaptic_relax_constant(self, relax_constant):
        N = self._N
        if np.shape(relax_constant) == (N, N):
            self._tau_syn = relax_constant;
        else:
            raise Exception(f"Excepted relax_constant matrix with shape ({N}, {N}), but shape {np.shape(relax_constant)} was sent...")
	
    def run_state(self, U, V, I_syn, t):
        dVdt = 0.04*np.power(V, 2) + 5*V + 140 - U + self.input(t) + np.sum(I_syn, axis=1);
        dUdt = self.a*(self.b*V - U);
        return (dVdt, dUdt)
        
    def calc_sol(self, U0, V0, T):
        I_syn=np.zeros((self._N, self._N))
        U = np.zeros((len(T), self._N))
        V = np.zeros((len(T), self._N))
        dt = T[1] - T[0]
        U[0] = U0
        V[0] = V0
        for n in range(0, len(T)-1):
            t = T[n]
            dVdt, dUdt = self.run_state(U[n], V[n], I_syn, t)
            V[n+1] = np.where(V[n] >= self.V_peak, self.c, V[n] + dt*dVdt)
            V[n] =  np.where(V[n] >= self.V_peak, self.V_peak+1, V[n])
            U[n+1] = np.where(V[n] >= self.V_peak, U[n]+self.d, U[n] + dt*dUdt)
            dI_syn = dt*(-I_syn*self._tau_syn + self._W * (np.where(V[n]>=self.V_peak, V[n] , 0)))
            I_syn = I_syn + dI_syn
        return {
            'v': V,
            'u': U
            }
    
    def firing_rastr(self, V, T):
        M = len(T)
        firing_indices = np.where(sol['v']>=30)
        return (T[firing_indices[0]], firing_indices[1])

from matplotlib import pyplot as plt

if __name__=="__main__":
    T = np.linspace(0, 200, 1000)
    net = Izhikevich_Network(N=10)
    net.input = lambda t: 10*np.random.rand(10)
    V0 = net.c
    U0 = net.b*net.c
    sol = net.calc_sol(U0, V0, T)
    firings = net.firing_rastr(sol['v'], T)
    plt.subplot(121)
    plt.plot(firings[0], firings[1], "k.")
    plt.subplot(122)
    plt.plot(T, sol['v'][:, 0])
    plt.show()

