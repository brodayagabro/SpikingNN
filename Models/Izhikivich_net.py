import numpy as np
from ipywidgets import interact
#%matplotlib inline
from scipy.fft import fft, fftfreq
import warnings
import pandas as pd
warnings.filterwarnings('ignore') 
from matplotlib import pyplot as plt
#%config InlineBackend.figure_format = 'svg' 

A = np.array([
    [0, 0],
    [-0.3, 0]
])

V_peak = 30
# Численное решение системы методом Эйлера
a = np.array([0.1, 0.02]);
b = np.array([0.2, 0.2]);
c = np.array([-65, -65]);
d = np.array([8, 8]);
tau = [10, 5]
def solve(T, I, V_peak):
    count_of_neurons = 2
    neuron_indexes = np.arange(count_of_neurons)
    I = np.zeros(count_of_neurons);
    # Euler method
    dt = T[1]-T[0]
    N = len(T)
    v = np.zeros((N, count_of_neurons))
    u = np.zeros((N, count_of_neurons))
    I_syn = np.zeros((N, count_of_neurons))
    v[0] = c
    u[0] = b * c
    v_peak = V_peak * np.ones(count_of_neurons)
    firings_t = np.array([])
    firings_N = np.array([])
    for n in range(0, N-1):
        t = T[n]
        I = Id(t)
        #I[0] = Id(t); I[1] = Id(t);
        v[n+1] = v[n] + dt*(0.04*np.power(v[n], 2) + 5*v[n] + 140 - u[n] + I + I_syn[n]);
        u[n+1] = u[n] + dt*a*(b*v[n] - u[n]);   # computing next step
        I_syn[n+1] = I_syn[n] - dt*(I_syn[n]/tau)
        fired = v[n+1] > v_peak # find spikes in next step
        if fired.any():
            v[n][fired] = v_peak[fired]; # equalising potential to v_peak
            v[n+1][fired] = c[fired] # reset potential
            u[n+1][fired] = u[n][fired] + d[fired]; # reset u
            I_syn[n+1][~fired] += A.dot(v[n])[~fired]
            #print(A.dot(v[n]), v[n])
            fired_numbers = neuron_indexes[fired]
            fired_times = t+0*fired_numbers
            firings_t = np.concatenate((firings_t, fired_times)) # save firings
            firings_N = np.concatenate((firings_N, fired_numbers))
    return (v, u,  (firings_t, firings_N), I_syn)



# simulation
count_of_neurons = 2

Tmax = 500
N = Tmax*3
T = np.linspace(0, Tmax, N)
ampl = np.array([7.5, 5])
Id = lambda t: ampl if t>10 else ampl
v, u, firings, I_syn = solve(T, Id, V_peak)
firings_t=firings[0]
firings_N=firings[1]
plt.scatter(firings_t, firings_N, 0.3)
plt.plot(T, v[:, 0], label="v1")
plt.plot(T, v[:, 1], label="v2")
#plt.plot(T, u[:, 0], label="u1")
#plt.plot(T, u[:, 1], label="u2")
#plt.plot(T, I_syn[:, 0], label="I_syn1")
plt.plot(T, I_syn[:, 1], 'k--', label="I_syn2")
plt.title(f"tau = {tau[1]}ms")
plt.xlabel("t, ms")
plt.legend(loc=1)
plt.grid()
plt.savefig(f"tau({tau[1]}).svg")
plt.show()