import unittest
import numpy as np
from synapses import synapses
from neuron_group import *

class TestSynapes(unittest.TestCase):
    def setUp(self):
        sour = neuron_group(size=2)
        tar = neuron_group(size=2)
        self.synapse = synapses(source=sour,target=tar)

    def test_properties(self):
        self.assertTrue(np.all(self.synapse.weights 
            == np.ones((2, 2))))
        self.assertTrue(self.synapse.target_size == 2)
        self.assertTrue(self.synapse.source_size == 2)

    def test_set_weights(self):
        W = np.zeros((2, 2))
        self.synapse.set_weigths(W)
        self.assertTrue(np.all(self.synapse.weights == W))
        W = np.zeros((2, 3))
        with self.assertRaises(Exception):
            self.synapse.set_weigths(W)
    
    def test_save_load_weights(self):
        name = 'W.npy'
        try:
            self.synapse.save_weigths(name)
        except: 
            pass
        self.synapse.load_weigths(name)
        self.assertTrue(np.all(self.synapse.weights 
            == np.zeros((2, 2))))


if __name__=="__main__":
    unittest.main()
