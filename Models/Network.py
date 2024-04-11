import numpy as np
import abc
# Basic class of Network description
class Network(object, metaclass=abc.ABCMeta):
    """
    Basic network class.
    This class is used to create the frame of network by setting weigts
    """
    def __init__(self, size=1, weigths=[]):
        self._size_ = size;
        self._weigths_ = weigths;


    @property
    def weigths(self):
        '''Getting weights of the network(Property).'''
        return self._weigths_

    @abc.abstractmethod
    def _model_(self):
        '''Neuron dynamics mathematical model(method).'''
        pass
        
class Synapses(object, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __init__(self):
        pass
    @abc.abstractmethod
    def model(self):
        '''Synapses dynamics maths model(method).'''
        pass
    
class Izhikevich_Network(Network):
    def __init__(self, **kwargs):
        Network.__init__(size=kwargs['size'], weigths=kwargs['weigths']);

    def _model_(self, V, U, I, Isyn):
        dV = 0.04*np.power(V, 2) + 5*V + 140 - U + I + Isyn;
        dU = a(b*V-u)
        
#help(Synapses)
#help(Network)
#help(Izhikevich_Network)

        

    

