"""
Implementation of the Izhikevich spiking neuron model.

Contains classes for a single network and a network with input/output matrices.
Equations are based on:
    E. M. Izhikevich, "Simple Model of Spiking Neurons", IEEE TNN 2003.
"""

import numpy as np
from ..network.connectivity import NameNetwork


class Izhikevich_Network(NameNetwork):
    """
    Izhikevich network with synaptic dynamics.
    
    Implements the ODE system:
        dV/dt = 0.04*V^2 + 5*V + 140 - U + I_app + Σ(I_syn)
        dU/dt = a*(b*V - U)
    With threshold reset when V >= 30 mV:
        V ← c, U ← U + d
    
    Parameters
    ----------
    a, b, c, d : np.ndarray or float
        Model parameters of shape (N,) or scalars (broadcast).
    V_peak : float, optional
        Spike detection threshold (default: 30 mV).
        
    Attributes
    ----------
    V, U : np.ndarray
        Current membrane potential and recovery variable.
    V_prev, U_prev : np.ndarray
        Values from the previous time step (for Euler integration).
    I_syn : np.ndarray
        Matrix of synaptic currents, shape (N, N).
    output : np.ndarray
        Vector of output spikes (0 or V_peak).
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        N = self.N
        
        # Helper function for safe parameter processing
        def _process_param(key, default_val):
            val = kwargs.get(key, default_val)
            arr = np.asarray(val)
            # If a scalar is provided, expand it to an array of size N
            if arr.ndim == 0 or arr.size == 1:
                return np.full(N, float(arr))
            return arr

        self.a = _process_param('a', 0.02)
        self.b = _process_param('b', 0.2)
        self.c = _process_param('c', -65.0)
        self.d = _process_param('d', 4.0)
        
        # Validate parameter dimensions
        for name, arr in [('a', self.a), ('b', self.b), ('c', self.c), ('d', self.d)]:
            if arr.shape[0] != N:
                raise ValueError(
                    f"Parameter '{name}' has shape {arr.shape}, "
                    f"but network size N={N}. Expected length {N}."
                )
        
        self.V_peak = kwargs.get('V_peak', 30.0)
        self.I_syn = np.zeros((self.N, self.N))
        self.output = np.zeros(self.N)
        
        # Initialize states
        self.V = self.c.copy()
        self.U = (self.c * self.b).copy()
        self.V_prev = self.V.copy()
        self.U_prev = self.U.copy()

    def set_params(self, **kwargs) -> None:
        """
        Updates parameters a, b, c, d and resets initial conditions.
        
        Parameters
        ----------
        a, b, c, d : array-like
            New parameters. Must have length N or be scalars.
        """
        N = self.N
        for key in ['a', 'b', 'c', 'd']:
            if key in kwargs:
                val = np.asarray(kwargs[key])
                if val.size == 1:
                    setattr(self, key, np.full(N, val.item()))
                elif len(val) == N:
                    setattr(self, key, val)
                else:
                    raise ValueError(f"Parameter '{key}' must have length {N} or be a scalar")
        self.set_init_conditions(v_noise=np.zeros(self.N))

    def run_state(self, U: np.ndarray, V: np.ndarray, I_syn: np.ndarray, I_app: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Computes the right-hand sides of the Izhikevich equations.
        
        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            (dVdt, dUdt) of shape (N,).
        """
        dVdt = 0.04 * V**2 + 5*V + 140 - U + I_app + np.sum(I_syn, axis=1)
        dUdt = self.a * (self.b * V - U)
        return dVdt, dUdt

    def set_init_conditions(self, **kwargs) -> None:
        """
        Resets the network state to initial values.
        
        Parameters
        ----------
        v_noise : np.ndarray, optional
            Gaussian noise for initializing V, shape (N,).
        """
        v_noise = kwargs.get('v_noise', np.zeros(self.N))
        self.V = self.c + v_noise
        self.U = self.c * self.b
        self.V_prev = self.V.copy()
        self.U_prev = self.U.copy()
        self.I_syn = np.zeros((self.N, self.N))
        self.output = np.zeros(self.N)

    def step(self, dt: float = 0.1, Iapp: float | np.ndarray = 0.0) -> None:
        """
        Performs one integration step using the Euler method.
        
        Parameters
        ----------
        dt : float, optional
            Time step in ms (default: 0.1).
        Iapp : float or np.ndarray, optional
            Input current. A scalar is applied to all neurons.
        """
        Iapp = np.full(self.N, Iapp) if np.isscalar(Iapp) else Iapp
        
        dVdt, dUdt = self.run_state(self.U_prev, self.V_prev, self.I_syn, Iapp)
        
        # Spike detection
        spiked = self.V_prev >= self.V_peak
        
        # Update states with reset
        self.V = np.where(spiked, self.c, self.V_prev + dt * dVdt)
        self.U = np.where(spiked, self.U_prev + self.d, self.U_prev + dt * dUdt)
        
        # Synaptic dynamics: dI_syn/dt = -I_syn/tau + W * output
        dI_syn = dt * (-self.I_syn * self.tau_syn + self.W * self.output)
        self.I_syn += dI_syn
        
        # Form output vector
        self.V_prev = np.where(self.V >= self.V_peak, self.V_peak + 1.0, self.V)
        self.U_prev = self.U.copy()
        self.output = np.where(spiked, self.V_peak, 0.0)


def IO_Network_decorator(cls):
    """
    Decorator that adds input (Q_app, Q_aff) and output (P) matrices to a network.
    
    Enables conversion of external signals to internal currents and collection
    of output spikes into a lower-dimensional vector.
    """
    class IO_Network(cls):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.input_size = kwargs.get('input_size', self.N)
            self.afferent_size = kwargs.get('afferent_size', self.N)
            self.output_size = kwargs.get('output_size', self.N)
            
            self.Q_app = np.asarray(kwargs.get('Q_app', np.ones((self.N, self.input_size))))
            self.Q_aff = np.asarray(kwargs.get('Q_aff', np.ones((self.N, self.afferent_size))))
            self.P = np.asarray(kwargs.get('P', np.ones((self.output_size, self.N))))
            
            # Validate shapes
            if self.Q_app.shape != (self.N, self.input_size):
                raise ValueError(f"Q_app: expected ({self.N}, {self.input_size}), got {self.Q_app.shape}")
            if self.Q_aff.shape != (self.N, self.afferent_size):
                raise ValueError(f"Q_aff: expected ({self.N}, {self.afferent_size}), got {self.Q_aff.shape}")
            if self.P.shape != (self.output_size, self.N):
                raise ValueError(f"P: expected ({self.output_size}, {self.N}), got {self.P.shape}")
            
            self.V_out = np.zeros(self.output_size)
        
        def step(self, dt: float = 0.1, Iapp: np.ndarray = 0.0, Iaff: np.ndarray = 0.0) -> None:
            """
            Step with external and afferent inputs.
            
            Parameters
            ----------
            Iapp : np.ndarray
                External stimulus of shape (input_size,).
            Iaff : np.ndarray
                Afferent signal of shape (afferent_size,).
            """
            I_total = self.Q_app @ Iapp + self.Q_aff @ Iaff
            super().step(dt=dt, Iapp=I_total)
            self.V_out = self.P @ self.output
            
        def __str__(self) -> str:
            return f"Wrapped {cls.__name__} (IO:{self.input_size}->{self.output_size})"

    return IO_Network


@IO_Network_decorator
class Izhikevich_IO_Network(Izhikevich_Network):
    """
    Izhikevich network with input and output projection matrices.
    
    Inherits all dynamics from Izhikevich_Network but overrides the step method
    to handle vector inputs and collect output spikes via matrix P.
    """
    pass