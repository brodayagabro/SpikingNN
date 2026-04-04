"""
🧠 SpikingNN - Spiking Neural Network Simulator

Модель Izhikevich с интерактивной визуализацией
"""

__version__ = "0.0.2"
__author__ = 'Kovalev Nickolai'
__email__ = 'kovalev.na@phystech.edu'

from .Izh_net import (
    Izhikevich_Network,
    Izhikevich_IO_Network,
    Network,
    NameNetwork,
    types2params,
    izhikevich_neuron,
)

__all__ = [
    "Izhikevich_Network",
    "Izhikevich_IO_Network",
    "Network",
    "NameNetwork",
    "types2params",
    "izhikevich_neuron",
]