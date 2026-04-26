"""
Integrated neuromechanical systems.

This module contains classes that combine neural networks with
mechanical components to create closed-loop sensorimotor systems.
"""

from .afferented_limb import Afferented_Limb, Simple_Afferented_Limb
from .var_limb import Net_Limb_connect, Var_Limb, run

__all__ = [
    "Afferented_Limb", "Simple_Afferented_Limb",
    "Net_Limb_connect", "Var_Limb", "run",
]