from synapses import * 
from neuron_group import *

if __name__=="__main__":
    dt =0.1;
    time_scale=1
    N = 1# source
    M = 1# target
    neurons_source = {}
    neurons_target = {}
    ids = range(0, N)
    for id in ids:
        neurons_source[id] = izhikevich_neuron(preset='RS')

    ids = range(0, M)
    for id in ids:
        neurons_target[id] = izhikevich_neuron(preset='RS')

    T = np.arange(0, 300, dt)
    g1 = izhikevich_neuron_group(size=N, neurons=neurons_source,
            time_scale = time_scale, dt=dt);
    g1.update_coefs();
    g1.set_Iapp(Iapp_new = 6*np.linspace(1, N, N))
    g1.check_ready_to_run()
    #print(len(source))
    
    g2 = izhikevich_neuron_group(size=M, neurons=neurons_target, time_scale=time_scale, dt=dt);
    g2.update_coefs()
    g2.check_ready_to_run()
    #print(len(target))

    syn12 = Simple_synapse(g1, g2, dt=dt, time_scale=time_scale);
    #print(syn.weights);
    syn21 = Simple_synapse(g2, g1, dt=dt, time_scale=time_scale);
    tau = 0.1*np.ones((N, M))
    W = np.random.rand(N, M)*10
    syn12.set_weigths(W)
    syn12.set_synaptic_relax_constant(tau)
    
    syn21.set_weigths(W)
    syn21.set_synaptic_relax_constant(tau)
    
    V_g1 = np.zeros((len(T), N))
    V_g2 = np.zeros((len(T), M))
    I_syn12 = np.zeros((len(T), N, M))
    I_syn21 = np.zeros((len(T), N, M))
    for i in range(len(T)):
      V_g1[i] = g1.get_v()
      V_g2[i] = g2.get_v()
      I_syn12[i] = syn12.I_syn
      I_syn21[i] = syn21.I_syn
      syn12.propogate()
      syn21.propogate()
      g1.dynamics()
      g2.dynamics()
      syn12.dynamics()
      syn21.dynamics()
    plt.figure()
    plt.subplot(221)
    plt.title("g1")
    for i in range(N):
        plt.plot(T, V_g1[:, i])
    plt.subplot(222)
    plt.title('g2')
    for j in range(M):
        plt.plot(T, V_g2[:, j])

    plt.subplot(413)
    plt.ylabel('I_syn12')
    for i in range(N):
        for j in range(M):
            plt.plot(T, I_syn21[:, i, j])

    plt.subplot(414)
    plt.ylabel('I_syn21')
    for i in range(N):
        for j in range(M):
            plt.plot(T, I_syn21[:, i, j])
    plt.show()




