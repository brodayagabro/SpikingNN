import numpy as np

class element:

    def __init__(self, **kwargs):
        """
            arguments;
            -------------------------------
            dt - time step in chosen time_scale
            time_scale - if 1000 - milliseconds
                        if 1 - seconds
            name(string) - name of neuron if it need
        """
        self.name = kwargs.get('name', None)
        self.dt = kwargs.get('dt', 0.5)#ms
        self.time_scale=kwargs.get('time_scale', 1)#
        self.dt=self.dt/self.time_scale;


class neuron_group(element):
    
    """
    Basic class of neuron populations
    """
    def __init__(self, size=10, **kwargs):
        """
        keyword parameters:
        -------------------------------
        size: size of neuron group,
        neurons: list of neuron objects
        Iapp: applicative current to group. 
        May be scalar or 1D array with the same size as neuron_group

        """
        super().__init__(**kwargs)
        self._size = size
        self.neurons = kwargs.get('neurons', {})
        self.Iapp = kwargs.get("Iapp", 0)
   
    def set_Iapp(self, Iapp_new=0):
        self.Iapp = Iapp_new

    def __len__(self):
        return self._size

    @property
    def size(self):
        return self._size

    def step(self):

        """
            Will be realized in child classes
        """
        pass

    def add_neuron(self, neuron, id=None):
        if id==None:
            id = len(self.neurons)
            self._size+=1
        self.neurons[id] = neuron
    
    def _check_size_(self):
        if self._size!=len(self.neurons):
            self._size = len(self.neurons)

    def input(self, input):
        """
            Method to set an input - 
            1D array with the same size as neuron group
        """
        if len(input) == self._size:
            self.__input__ = input
        else:
            raise Exception(f"Input size must be ({self._size},)...")

    def output(self):
        """
            
            Will be realized in child classes
        """
        pass
       
      
        
class izhikevich_neuron(element):
    
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
        super().__init__(**kwargs)
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
                    kwargs.get('c', -65),
                    kwargs.get('d', 2)
                ]
            ]
        idx = preset_list.index(preset)
        assert preset in preset_list,f'Preset {preset} does not exist! Use one from {preset_list}'
        self._a = param_list[idx][0]
        self._b = param_list[idx][1]
        self._c = param_list[idx][2]
        self._d = param_list[idx][3]
    
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


class LIF_neuron(neuron_group):
    def __init__(self):
        neuron_group.__init__()


class izhikevich_neuron_group(neuron_group):

    """
    group of Izhikevich neurons - container converting
    list of neurons to computational model implemented numpy. 
    """
    def __init__(self, **kwargs):
        """
        """
        super().__init__(**kwargs);
        self.__input__ = 0
        self._v_ = {}
        self._u_ = {}

    def update_coefs(self):
        """
        Private method creating parameter's vertors.
        """
        self._check_size_()
        self.__a__ = np.zeros(self._size)
        self.__b__ = np.zeros(self._size)
        self.__c__ = np.zeros(self._size)
        self.__d__ = np.zeros(self._size)
        self.__ap_threshold__ = 30*np.ones(self._size)
        for id , neuron in self.neurons.items():
            self.__a__[id] = neuron.a
            self.__b__[id] = neuron.b
            self.__c__[id] = neuron.c
            self.__d__[id] = neuron.d
            self.__ap_threshold__[id] = neuron.ap_threshold

    def __check_state_variables_to_run__(self):
        """
        Compare dimensions of params with group size 
        and correct params
        """
        if len(self._v_) != self._size:
            self._v_ = np.zeros(self._size)
        if len(self._u_) != self._size:
            self._u_ = np.zeros(self._size)
        if ((len(self.__a__) != self._size) or
            (len(self.__b__) != self._size) or
            (len(self.__c__) != self._size) or
            (len(self.__d__) != self._size)):
                self.update_coefs()

    def check_ready_to_run(self):
        """
        Public methos cheking all attributs of object 
        ready to simulate.
        """
        self._check_size_()
        self.__check_state_variables_to_run__()
        self._v_ = -70
        self._u_ = self._v_ * self.__b__
        return True


    @property
    def a(self):
        return self.__a__
    
    @property
    def b(self):
        return self.__b__

    @property
    def c(self):
        return self.__c__

    @property
    def d(self):
        return self.__d__

    @property
    def ap_threshold(self):
        return self.__ap_threshold__

    def get_v(self):
        return self._v_

    def get_u(self):
        return self._u_

    def output(self):
        return np.where(self._v_>=self.__ap_threshold__,
                self.__ap_threshold__, 0)
    
    def step(self):
        dVdt = self.time_scale*(0.04*np.power(self._v_, 2) +
                5*self._v_ + 140 - self._u_ +
                self.__input__ + self.Iapp
                );
        dUdt = self.time_scale*self.__a__*(self.__b__*self._v_ -
                self._u_);
        
        self._v_ = np.where(
                self._v_ >= self.__ap_threshold__, 
                self.__c__,
                self._v_ + self.dt*dVdt
            )
        self._u_ = np.where(
                self._v_ >= self.__ap_threshold__,
                self._u_ + self.__d__,
                self._u_ + self.dt*dUdt
            )



            

from matplotlib import pyplot as plt
# Need to learn how to write unit tests
if __name__=="__main__":
    ing = izhikevich_neuron_group(size=4)
    neurons = {}
    ids = range(0, len(ing))
    for id in ids:
        ing.add_neuron(izhikevich_neuron(preset='RS'))
        neurons[id] = izhikevich_neuron(preset='RS')
    dt=0.1
    ing1 = izhikevich_neuron_group(size=4, neurons=neurons, time_scale = 1000, dt=dt)
    print(ing1.time_scale)
    print(ing1.dt)
    ing1.update_coefs()
    assert len(ing1) == len(ing1.neurons)
    
    assert np.all(ing1.a == np.array([0.02, 0.02, 0.02, 0.02]))
    assert np.all(ing1.b == np.array([0.2, 0.2, 0.2, 0.2]))
    assert np.all(ing1.c == np.array([-65, -65, -65, -65]))
    assert np.all(ing1.d == np.array([8, 8, 8, 8]))
    assert np.all(ing1.ap_threshold == 30*np.ones(4))
    print(ing1.check_ready_to_run())
    T = np.arange(0, 1, ing1.dt)
    V = np.zeros((len(T), 4))
    print(V[:5])
    ing1.set_Iapp(Iapp_new = 10)
    for i in range(len(T)):
        V[i] = ing1.get_v()
        ing1.input(10*np.random.rand(4)) 
        ing1.step();
    print(V[:5])
    plt.plot(T, V[:, 0])
    plt.plot(T, V[:, 1])
    plt.show()
