import numpy as np
from neuron_group import *
from neuron_group import *

class synapses(element):
    """
    Basic class of synapses.
    Also this class can include plastisicty model which can implmented
    in dynamics or some learning rule methods
    """
    def __init__ (self, source, target, **kwargs):
        """
        source - source(pre-synaptic) neuron group
        target - target(post-synaptic) neuron group
        dt - time step in microseconds

        In initiation synaptic weigths seted as random matrix with size
        (source.size, target.size)

        I_syn - Synaptic, calculated by model defined in dynamic
        """
        try:
            self._source_size = source.size;
            self._target_size = target.size;
        except:
            Exception("source/target error...")
        N = self._source_size;
        M = self._target_size;
        self.source = source
        self.target = target
        self.dt = kwargs.get('dt', 0.5)#ms
        self._weights = np.ones((N, M)) 
        self.mask = np.ones((N, M))
        self.I_syn = np.zeros_like(self._weights)
    
    @property
    def weights(self):
        """
        Return matrix of weigths
        """
        return self._weights

    @property
    def source_size(self):
        """
        Return size of pre-synaptic neuron group
        """
        return self._source_size

    @property
    def target_size(self):
        """
        Return post-synaptic neuron-group size
        """
        return self._target_size

    def set_weigths(self, new_weights = []):
        """
        Set weitghts of synaptic matrix as you want
        """
        N = self._source_size;
        M = self._target_size;
        if np.shape(new_weights) == (N, M):
            self._weights = np.copy(new_weights)
        else:
            raise Exception(f"Excepted matrix shape ({N}, {M})")

    def save_weigths(self, name):
        """Save weigths as name.npy file"""
        np.save(name, self._weights);

    def load_weigths(self, name):
        """Load weights matrix from *.npy file"""
        try:
            buffer = np.load(name)
        except:
            print(f"can't open file{name}")
            exit()
        N = self._target_size;
        M = self._source_size;
        if np.shape(buffer) == (N, M):
            self._weights = buffer
        else:
            raise Exception("Loaded file contents matrix with uncorrect shape...")

    def dynamics(self):
        """
        will be realized in child classes
        dinamics related to neuron_model and synapses model
        """
        pass

    def propogate(self):
        """
        Related to dynamics
        Main task - to calculate vector of synaptic current
        By formula: I_syn_post = I_syn(v_pre)
        """
        pass

    def connect(source_id, target_id):
        """Method od Synapses to connect neuron from source to neuron from target with id - source_id, target_id"""
        N = self._source_size;
        M = self._target_size;
        if (source_id >= N or target_id>=M):
            raise Exception(f"indexes is out of bound of neurons")
        else:
            self.mask[source_id, target_id] = 1

    def disconnect(source_id, target_id):
        """Method od Synapses to connect neuron from source to neuron from target with ids - source_id, target_id"""
        N = self._source_size;
        M = self._target_size;
        if (source_id >= N or target_id>=M):
            raise Exception(f"indexes is out of bound of neurons")
        else:
            self.mask[source_id, target_id] = 0


class Simple_synapse(synapses):
    """
    Class calculate synaptic currents like delta-increase if ij element of synaptic current matrix on j-post-synaptic neuron spiked
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs);
        self.time_scale = kwargs.get('time_scale', 1)
        N = self._source_size;
        M = self._target_size;
        self._tau_syn = np.ones((N, M))
    
    def dynamics(self):
        """
        Calculate dynamics of synaptic current
        """
        dI_syndt = self.time_scale*(-self.I_syn*self._tau_syn 
                + (self._weights.T * self.source.output()).T)
        self.I_syn += dI_syndt*self.dt;

    def propogate(self):
        """
        Apply synaptic curret to post-synaps
        """
        self.target.input(np.sum(self.I_syn, axis=0))

    def set_synaptic_relax_constant(self, relax_constant):
        N = self._source_size;
        M = self._target_size;
        if np.shape(relax_constant) == (N, M):
            self._tau_syn = relax_constant;
        else:
            raise Exception(f"Excepted relax_constant matrix with shape ({N}, {N}), but shape {np.shape(relax_constant)} was sent...")


    def save_relax_const(self, name="R"):
        """Save Relaxation constant matrix to name.npy"""
        np.save(name+'.npy', self._tau_syn)

    def load_ralax_const(self, name='R.npy'):
        buffer = np.load(name)
        N = self._source_size;
        M = self._target_size;
        if np.shape(buffer) == (N, M):
            self._tau_syn = buffer
        else:
            raise Exception("Loaded error! Array has unexpected shape...")

from matplotlib import pyplot as plt

if __name__=="__main__":
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
      syn.dynamics()
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





