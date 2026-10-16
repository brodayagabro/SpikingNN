"""Numerical contract shared by training, selection, testing and replay."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Mechanics:
    """Fixed hyperparameters, never genes or mutation coordinates."""
    mass_kg: float = .3
    length_m: float = .3
    viscosity: float = .001
    a1_m: float = .2
    a2_m: float = .05
    gravity_m_s2: float = 9.81
    muscle_torque_factor: float = 10.


MECHANICS = Mechanics()
DT_MS = 0.1
MODEL = "SpikingNN.Afferents/v2"
POPULATION_SCHEMA = "stepping_population_v2"
FITNESS = "tracking_v2_mean_lower_quartile_min"
