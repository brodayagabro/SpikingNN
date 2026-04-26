"""
Neuromechanical components for spiking neural network simulation.

This module contains models for:
- Muscle dynamics (SimpleAdaptedMuscle)
- Afferent feedback (Afferents, Simple_Afferents)
- Limb mechanics (Pendulum, OneDOFLimb, OneDOFLimb_withGR)

These components can be combined with neural networks to create
closed-loop neuromechanical systems.
"""

from .muscles import SimpleAdaptedMuscle
from .afferents import Afferents, Simple_Afferents
from .limb import Pendulum, OneDOFLimb, OneDOFLimb_withGR

__all__ = [
    "SimpleAdaptedMuscle",
    "Afferents", "Simple_Afferents",
    "Pendulum", "OneDOFLimb", "OneDOFLimb_withGR",
]