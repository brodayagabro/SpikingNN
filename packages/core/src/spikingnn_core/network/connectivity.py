# packages/core/src/spikingnn_core/network/connectivity.py
"""
Base classes for network topology.

This module defines connectivity structure (mask M), weight matrix (W),
and synaptic current relaxation constants (tau_syn).
"""

import numpy as np


class Network:
    """
    Base class for neural networks without specific dynamics.
    
    Stores connectivity topology and provides methods for modifying weights
    and synaptic constants. Serves as a parent class for specific neuron models.
    
    Parameters
    ----------
    N : int, optional
        Number of neurons (default: 10).
    M : np.ndarray, optional
        Connectivity mask of shape (N, N). M[j, i] != 0 indicates a connection i → j.
    W : np.ndarray, optional
        Synaptic weight matrix of shape (N, N).
    tau_syn : np.ndarray, optional
        Matrix of synaptic current relaxation times (in ms). 
        Stored internally as 1/tau_syn for computational efficiency.
        
    Attributes
    ----------
    N : int
        Number of neurons.
    M : np.ndarray
        Connectivity mask.
    W : np.ndarray
        Weight matrix.
    tau_syn : np.ndarray
        Inverse relaxation constants (1/ms).
    """
    
    def __init__(self, **kwargs):
        self.N = kwargs.get('N', 10)
        
        self.M = kwargs.get('M', np.ones((self.N, self.N)))
        if self.M.shape != (self.N, self.N):
            raise ValueError(f"Expected mask of shape ({self.N}, {self.N}), got {self.M.shape}")
            
        self.W = kwargs.get('W', np.ones((self.N, self.N)))
        if self.W.shape != (self.N, self.N):
            raise ValueError(f"Expected weights of shape ({self.N}, {self.N}), got {self.W.shape}")
            
        tau_raw = kwargs.get('tau_syn', np.ones((self.N, self.N)))
        self.tau_syn = 1.0 / tau_raw
        if self.tau_syn.shape != (self.N, self.N):
            raise ValueError(f"Expected tau_syn of shape ({self.N}, {self.N}), got {self.tau_syn.shape}")

    def __len__(self) -> int:
        """Returns the number of neurons in the network."""
        return self.N

    def connect(self, i: int, j: int, coef: float, w: float = 1.0, tau: float = 10.0) -> None:
        """
        Establishes a directed connection from neuron i to neuron j.
        
        Parameters
        ----------
        i : int
            Index of the presynaptic neuron.
        j : int
            Index of the postsynaptic neuron.
        coef : float
            Connection sign: >0 for excitatory, <0 for inhibitory, =0 to remove connection.
        w : float, optional
            Magnitude of synaptic weight (default: 1.0).
        tau : float, optional
            Synaptic relaxation time in ms (default: 10.0).
            
        Raises
        ------
        ValueError
            If indices are outside the range [0, N-1].
        """
        coef = np.sign(coef)
        if not (0 <= i < self.N and 0 <= j < self.N):
            raise ValueError(f"Indices must be in range [0, {self.N-1}], got i={i}, j={j}")
            
        self.M[j, i] = coef
        self.W[j, i] = w
        self.tau_syn[j, i] = 1.0 / tau

    def set_weights(self, W: np.ndarray) -> None:
        """
        Sets a new weight matrix respecting the connectivity mask.
        
        Weights are applied only where M != 0. In other positions,
        the W matrix is zeroed to prevent fictitious currents.
        
        Parameters
        ----------
        W : np.ndarray
            Weight matrix of shape (N, N).
            
        Raises
        ------
        ValueError
            If matrix shape does not match (N, N).
        """
        if W.shape != (self.N, self.N):
            raise ValueError(f"Expected shape ({self.N}, {self.N}), got {W.shape}")
        self.W = np.where(self.M != 0, W, 0.0)

    def set_synaptic_relax_constant(self, relax_constant: np.ndarray) -> None:
        """
        Sets synaptic current relaxation times.
        
        Parameters
        ----------
        relax_constant : np.ndarray
            Matrix of τ values in ms of shape (N, N).
            Internally converted to 1/τ for integration step optimization.
            
        Raises
        ------
        ValueError
            If matrix shape does not match (N, N).
        """
        if relax_constant.shape != (self.N, self.N):
            raise ValueError(f"Expected shape ({self.N}, {self.N}), got {relax_constant.shape}")
        self.tau_syn = 1.0 / relax_constant


class NameNetwork(Network):
    """
    Extension of Network with support for neuron names.
    
    Allows accessing connections and weights via string identifiers,
    simplifying debugging and configuration of complex networks.
    
    Parameters
    ----------
    names : list[str], optional
        List of neuron names of length N. If not specified, generates ['Neuron_0', ...].
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.names = kwargs.get('names', [f"Neuron_{i}" for i in range(self.N)])
        if len(self.names) != self.N:
            raise ValueError(f"Name list length ({len(self.names)}) does not match N={self.N}")

    def set_name(self, i: int, name: str) -> None:
        """Sets the name for the neuron at index i."""
        if not (0 <= i < self.N):
            raise ValueError(f"Index out of range [0, {self.N-1}]")
        self.names[i] = name

    def print_names(self) -> None:
        """Prints a table mapping indices to neuron names to console."""
        for i, name in enumerate(self.names):
            print(f"[{i}] : {name}")

    def print_connections(self) -> None:
        """Prints all existing connections in format 'source -> target type: sign'."""
        for i in range(self.N):
            for j in range(self.N):
                if self.M[j, i] != 0:
                    sign = "excitatory" if self.M[j, i] > 0 else "inhibitory"
                    print(f"{self.names[i]} -> {self.names[j]} type: {sign}")

    def get_weight_by_names(self, source_name: str, target_name: str) -> float | None:
        """
        Returns the weight of a connection between two neurons by their names.
        
        Returns
        -------
        float | None
            Connection weight or None if names not found or no connection exists.
        """
        try:
            src = self.names.index(source_name)
            tgt = self.names.index(target_name)
            return self.W[tgt, src] if self.M[tgt, src] != 0 else None
        except ValueError:
            return None

    def set_weights_by_names(self, source_name: str, target_name: str, new_weight: float) -> bool:
        """
        Modifies connection weight by neuron names, preserving the sign from mask M.
        
        Returns
        -------
        bool
            True if connection was modified, False if names not found or no connection.
        """
        try:
            src = self.names.index(source_name)
            tgt = self.names.index(target_name)
            if self.M[tgt, src] != 0:
                self.W[tgt, src] = np.sign(self.M[tgt, src]) * abs(new_weight)
                return True
            return False
        except ValueError:
            return False
        
    def connect_net(self, source_name: str, target_name: str) -> bool:
        """Checks whether a connection exists between two neurons by name."""
        try:
            src = self.names.index(source_name)
            tgt = self.names.index(target_name)
            return self.M[tgt, src] != 0
        except ValueError:
            return False