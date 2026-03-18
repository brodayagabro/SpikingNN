import numpy as np
import scipy as sp
from neuron_group import *
from synapses import *


class network(element):
    

    def __init__(self, neurons : [neuron_group], synapese : [synapses], dt=0.1):
        self.dt = dt
        self.neurons = neurons
        self.synapses = synapses
        for neuron in self.neurons:
            neuron.dt = dt
        for synaps in self.synapses:
            synaps.dt = dt

    def step(self, **kwargs):
        for synaps in self.synapses:
            synaps.propogate()
            synaps.step()

        for neuron in self.neurons:
            neuron.step()

class runner(element):


    def __init__(self, network):
        self.network = network

    def create_state_dict(self):
        

    def simulate(self, T : np.ndarray):
        network.dt = T[1] - T[0]
        for t in T:
            network.step()


if __name__ == "__main__":

    dt =0.1;
    time_scale=1
    N = 2# source
    M = 2# target
    neurons_source = {}
    neurons_target = {}
    ids = range(0, N)
    for id in ids:
        neurons_source[id] = izhikevich_neuron(preset='RS')

    ids = range(0, M)
    for id in ids:
        neurons_target[id] = izhikevich_neuron(preset='RS')

    T = np.arange(0, 300, dt)
    source = izhikevich_neuron_group(size=N, neurons=neurons_source,
            time_scale = time_scale, dt=dt);
    source.update_coefs();
    source.set_Iapp(Iapp_new = 6*np.linspace(1, N, N))
    source.check_ready_to_run()
    #print(len(source))
    
    target = izhikevich_neuron_group(size=M, neurons=neurons_target, time_scale=time_scale, dt=dt);
    target.update_coefs()
    target.check_ready_to_run()
    #print(len(target))

    syn = Simple_synapse(source, target, dt=dt, time_scale=time_scale);
    #print(syn.weights);
    tau = 0.1*np.ones((N, M))
    W = np.random.rand(N, M)*10
    syn.set_weigths(W)
    syn.set_synaptic_relax_constant(tau)
    
    V_source = np.zeros((len(T), N))
    V_target = np.zeros((len(T), M))
    I_syn = np.zeros((len(T), N, M))
    for i in range(len(T)):
      V_source[i] = source.get_v()
      V_target[i] = target.get_v()
      I_syn[i] = syn.I_syn
      syn.propogate()
      source.dynamics()
      target.dynamics()
      syn.step()
    plt.figure()
    plt.subplot(221)
    plt.title("presyn")
    for i in range(N):
        plt.plot(T, V_source[:, i])
    plt.subplot(222)
    plt.title('postsyn')
    for j in range(M):
        plt.plot(T, V_target[:, j])

    plt.subplot(212)
    plt.title('I_syn')
    for i in range(N):
        for j in range(M):
            plt.plot(T, I_syn[:, i, j])
    plt.show()




