"""
🧠 SpikingNN - Spiking Neural Network Simulator

Модель Izhikevich с интерактивной визуализацией
"""

__version__ = "0.0.2"
__author__ = 'Kovalev Nickolai'
__email__ = 'kovalev.na@phystech.edu'

from .core.Izh_net import (
    Izhikevich_Network,
    Izhikevich_IO_Network,
    Network,
    NameNetwork,
    types2params,
    izhikevich_neuron,
)

from .core.Var_Limb import Var_Limb

from .api import create_simulation, run_simulation, run_simulation_async, get_results, Simulation, Results

__all__ = [
    "Izhikevich_Network",
    "Izhikevich_IO_Network",
    "Network",
    "NameNetwork",
    "types2params",
    "izhikevich_neuron",
    "Var_Limb",
    "create_simulation",
    "run_simulation",
    "run_simulation_async",
    "get_results",
    "Simulation",
    "Results"
]
