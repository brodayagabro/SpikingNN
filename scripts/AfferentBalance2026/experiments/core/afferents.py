"""Numba translation of SpikingNN.core.Izh_net.Afferents, including all terms."""
import numpy as np
from numba import njit
from SpikingNN.core.Izh_net import Afferents

PARAMETER_NAMES = ("p_v", "k_v", "k_dI", "k_dII", "k_nI", "k_nII",
                   "k_f", "const_I", "const_II")
PARAMETERS = tuple(float(getattr(Afferents, name)) for name in PARAMETER_NAMES)


@njit(cache=True, inline="always")
def afferent_values(velocity, length, force, activation, length_threshold,
                    force_threshold, parameters=PARAMETERS):
    p_v, k_v, k_dI, k_dII, k_nI, k_nII, k_f, const_I, const_II = parameters
    v = max(velocity / length_threshold, 0.)
    d = max((length - length_threshold) / length_threshold, 0.)
    f = max((force - force_threshold) / force_threshold, 0.)
    return (k_v * v**p_v + k_dI * d + k_nI * activation + const_I,
            k_dII * d + k_nII * activation + const_II, k_f * f)


@njit(cache=True)
def afferent_response(velocity, length, force, activation, length_threshold,
                      force_threshold, parameters=PARAMETERS):
    return np.array(afferent_values(velocity, length, force, activation,
                                   length_threshold, force_threshold, parameters))
