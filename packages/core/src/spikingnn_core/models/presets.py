# packages/core/src/spikingnn_core/models/presets.py
"""
Izhikevich neuron preset module.

Contains a parameter container class and a function for converting lists of neuron types
into arrays of parameters a, b, c, d used during network initialization.
All values are based on the work of E. M. Izhikevich (2003) 
"Simple Model of Spiking Neurons", IEEE TNN.
"""

import numpy as np


class izhikevich_neuron:
    """
    Parameter container for a single Izhikevich neuron.
    
    This class does not perform computations; it only stores and validates parameters
    for subsequent use in network models.
    
    Parameters
    ----------
    preset : str, optional
        Neuron type from the list: 'RS', 'IB', 'CH', 'FS', 'TC', 'RZ', 'LTS'.
        If None, custom parameters a, b, c, d are used.
    a, b, c, d : float, optional
        Custom model parameters (effective only when preset=None).
    ap_threshold : float, optional
        Threshold voltage for spike detection (default: 30 mV).
        
    Attributes
    ----------
    params : list[float]
        List of 4 elements [a, b, c, d].
    ap_threshold : float
        Spike generation threshold.
        
    Examples
    --------
    >>> n = izhikevich_neuron(preset='RS')
    >>> print(n.params)
    [0.02, 0.2, -65, 8]
    """
    
    def __init__(self, **kwargs):
        preset_list = ['RS', 'IB', 'CH', 'FS', 'TC', 'RZ', 'LTS', None]
        preset = kwargs.get('preset', None)
        self.ap_threshold = kwargs.get('ap_threshold', 30.0)
        
        # Izhikevich preset parameters
        param_list = [
            [0.02, 0.2, -65.0, 8.0],      # RS
            [0.02, 0.2, -55.0, 4.0],      # IB
            [0.02, 0.2, -50.0, 2.0],      # CH
            [0.1,  0.2, -65.0, 0.05],     # FS
            [0.02, 0.25, -65.0, 0.05],    # TC
            [0.1,  0.26, -65.0, 8.0],     # RZ
            [0.02, 0.25, -65.0, 2.0],     # LTS
            [kwargs.get('a', 0.02), 
             kwargs.get('b', 0.2), 
             kwargs.get('c', -65.0), 
             kwargs.get('d', 2.0)]
        ]
        
        if preset not in preset_list:
            raise ValueError(f"Preset '{preset}' not found. Available: {preset_list[:-1]}")
            
        idx = preset_list.index(preset)
        self.params = param_list[idx]


def types2params(types: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Converts a list of neuron types into arrays of parameters a, b, c, d.
    
    Parameters
    ----------
    types : list[str]
        List of neuron type strings (length N).
        
    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        Tuple of four arrays of shape (N,) containing parameters a, b, c, d.
        
    Raises
    ------
    ValueError
        If an unknown preset is present in the list.
        
    Examples
    --------
    >>> a, b, c, d = types2params(['RS', 'FS', 'CH'])
    >>> print(a)
    [0.02 0.1  0.02]
    """
    N = len(types)
    Params = np.zeros((N, 4))
    for i, t in enumerate(types):
        neuron = izhikevich_neuron(preset=t)
        Params[i] = neuron.params
    return Params[:, 0], Params[:, 1], Params[:, 2], Params[:, 3]