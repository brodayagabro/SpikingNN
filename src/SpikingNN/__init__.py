# src/SpikingNN/__init__.py
"""
SpikingNN: Core computational engine for spiking neural networks and
neuromechanical CPG modeling.

Provides Izhikevich/FHN dynamics, synaptic propagation, limb biomechanics,
and afferent feedback loops. GUI and scripts are distributed separately
under apps/ and scripts/.

Example:
    >>> from SpikingNN.core import Izhikevich_Network
    >>> net = Izhikevich_Network(N=4)
    >>> net.step(dt=0.1, Iapp=5.0)

Библиотека SpikingNN: Вычислительное ядро для спайковых нейронных сетей
и нейромеханического моделирования CPG.

Предоставляет динамику Ижикеича/ФХН, синаптическую передачу, биомеханику
конечностей и петли афферентной обратной связи. GUI и скрипты распространяются
отдельно в папках apps/ и scripts/.

Пример:
    >>> from SpikingNN.core import Izhikevich_Network
    >>> net = Izhikevich_Network(N=4)
    >>> net.step(dt=0.1, Iapp=5.0)
"""
__version__ = "0.2.0"

# ✅ Исправленные импорты согласно новой структуре
from .core.Izh_net import (
    Izhikevich_Network,
    Izhikevich_IO_Network,
    Network,
    NameNetwork,
    types2params,
    izhikevich_neuron
)
from .models.Var_Limb import Var_Limb, Afferented_Limb
from .utils.net_preparation import find_bursts, create_firing_rastr

__all__ = [
    "Izhikevich_Network", "Izhikevich_IO_Network", "Network", "NameNetwork",
    "types2params", "izhikevich_neuron", "Var_Limb", "Afferented_Limb",
    "find_bursts", "create_firing_rastr"
]