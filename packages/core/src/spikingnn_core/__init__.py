# packages/core/src/spikingnn_core/__init__.py
"""
SpikingNN Core — Computational engine for spiking neural network simulation.

This package contains mathematical neuron models (Izhikevich, FitzHugh-Nagumo),
synaptic connectivity mechanisms, neuromechanical components (muscles, afferents, limbs),
and a system for integrating networks with peripheral components (Var_Limb).

Example usage:
    >>> from spikingnn_core import Izhikevich_Network, types2params
    >>> a, b, c, d = types2params(['RS', 'FS'])
    >>> net = Izhikevich_Network(N=2, a=a, b=b, c=c, d=d)
    >>> net.step(dt=0.1, Iapp=5.0)
"""

from .models.presets import types2params, izhikevich_neuron
from .models.izhikevich import Izhikevich_Network, Izhikevich_IO_Network
from .models.fhn import FizhugNagumoNetwork
from .network.connectivity import Network, NameNetwork

__all__ = [
    "types2params",
    "izhikevich_neuron",
    "Izhikevich_Network",
    "Izhikevich_IO_Network",
    "FizhugNagumoNetwork",
    "Network",
    "NameNetwork",
]