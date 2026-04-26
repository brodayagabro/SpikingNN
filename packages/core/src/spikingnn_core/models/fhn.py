"""
FitzHugh-Nagumo neuron network implementation.

The FitzHugh-Nagumo model is a simplified version of the Hodgkin-Huxley model,
capturing the essential dynamics of excitable systems with only two variables.

Reference:
    FitzHugh, R. (1961). Impulses and physiological states in theoretical 
    models of nerve membrane. Biophysical Journal, 1(6), 445-466.
"""

import numpy as np
from ..network.connectivity import NameNetwork


class FizhugNagumoNetwork(NameNetwork):
    """
    Network of FitzHugh-Nagumo neurons.
    
    The FitzHugh-Nagumo model is described by the following equations:
        dV/dt = V * (a - V) * (V - 1) - U + I_app + Σ(I_syn)
        dU/dt = b * V - c * U
    
    where:
        V : membrane potential (dimensionless)
        U : recovery variable (dimensionless)
        a, b, c : model parameters
        ts : time scaling parameter
    
    This model captures the essential dynamics of excitable systems with
    spike generation and recovery, making it computationally efficient
    for large-scale network simulations.
    
    Parameters
    ----------
    a : np.ndarray or float, optional
        Excitability parameter (default: 0.1). Controls the shape of 
        the cubic nullcline. Typical range: -0.1 to 0.7.
    b : np.ndarray or float, optional
        Recovery rate parameter (default: 0.01). Controls the speed 
        of the recovery variable. Typical range: 0.01 to 0.1.
    c : np.ndarray or float, optional
        Recovery decay parameter (default: 0.02). Controls the decay 
        rate of the recovery variable. Typical range: 0.01 to 0.1.
    ts : np.ndarray or float, optional
        Time scaling parameter (default: 1.0). Allows adjustment of 
        the timescale of dynamics.
    V_th : np.ndarray or float, optional
        Threshold potential for spike detection (default: 0.0).
    k : np.ndarray or float, optional
        Steepness parameter for output function (default: 8.0).
    V1_2 : np.ndarray or float, optional
        Half-activation potential for output function (default: 0.1).
        
    Attributes
    ----------
    V : np.ndarray
        Current membrane potential of shape (N,).
    U : np.ndarray
        Current recovery variable of shape (N,).
    V_prev : np.ndarray
        Membrane potential from previous time step.
    U_prev : np.ndarray
        Recovery variable from previous time step.
    I_syn : np.ndarray
        Synaptic current matrix of shape (N, N).
    output : np.ndarray
        Output signal (smoothed spike indicator) of shape (N,).
    V_peak : float
        Peak voltage for spike detection (fixed at 1.0).
        
    Examples
    --------
    >>> from spikingnn_core import FizhugNagumoNetwork
    >>> net = FizhugNagumoNetwork(N=10, a=0.1, b=0.01, c=0.02)
    >>> net.set_init_conditions()
    >>> for t in range(1000):
    ...     net.step(dt=0.1, Iapp=0.5)
    ...     if t % 100 == 0:
    ...         print(f"t={t}: V_mean={np.mean(net.V):.3f}")
    """
    
    def __init__(self, **kwargs):
        """Initialize FitzHugh-Nagumo network."""
        super().__init__(**kwargs)
        N = self.N
        
        # Model parameters (broadcast scalars to arrays if needed)
        def _broadcast_param(val, default, size):
            arr = np.asarray(val) if val is not None else np.full(size, default)
            return arr if arr.ndim > 0 else np.full(size, arr.item())
        
        self.a = _broadcast_param(kwargs.get('a'), 0.1, N)
        self.b = _broadcast_param(kwargs.get('b'), 0.01, N)
        self.c = _broadcast_param(kwargs.get('c'), 0.02, N)
        self.ts = _broadcast_param(kwargs.get('ts'), 1.0, N)
        self.V_th = _broadcast_param(kwargs.get('V_th'), 0.0, N)
        self.k = _broadcast_param(kwargs.get('k'), 8.0, N)
        self.V1_2 = _broadcast_param(kwargs.get('V1_2'), 0.1, N)
        
        # State variables
        self.V = np.zeros(N)
        self.U = np.zeros(N)
        self.V_prev = np.zeros(N)
        self.U_prev = np.zeros(N)
        self.I_syn = np.zeros((N, N))
        self.output = np.zeros(N)
        
        self.set_init_conditions()
    
    def syn_output(self) -> np.ndarray:
        """
        Compute smooth output signal based on membrane potential.
        
        The output is a sigmoidal function of V that approximates spike
        generation. It provides a continuous output signal suitable for
        synaptic transmission.
        
        Returns
        -------
        np.ndarray
            Output signal of shape (N,) with values in [0, 1].
        """
        return np.where(
            self.V_prev > self.V_th,
            1.0 / (1.0 + np.exp(-(self.V_prev - self.V1_2) / self.k)),
            0.0
        )
    
    def set_init_conditions(self) -> None:
        """
        Reset all state variables to initial conditions.
        
        Sets membrane potential V and recovery variable U to zero,
        and clears synaptic currents and output signals.
        """
        self.V = np.zeros(self.N)
        self.U = np.zeros(self.N)
        self.V_prev = np.zeros(self.N)
        self.U_prev = np.zeros(self.N)
        self.I_syn = np.zeros((self.N, self.N))
        self.output = np.zeros(self.N)
    
    def step(self, dt: float = 0.1, Iapp: float | np.ndarray = 0.0, 
             Iaff: float | np.ndarray = 0.0) -> None:
        """
        Perform one integration step using Euler method.
        
        Updates the membrane potential V and recovery variable U
        according to the FitzHugh-Nagumo equations with synaptic input.
        
        Parameters
        ----------
        dt : float, optional
            Time step (default: 0.1).
        Iapp : float or np.ndarray, optional
            Applied external current (default: 0.0). If scalar, applied
            to all neurons. If array, must have shape (N,).
        Iaff : float or np.ndarray, optional
            Afferent input current (default: 0.0). Added to Iapp.
            
        Notes
        -----
        The integration uses forward Euler method:
            V(t+dt) = V(t) + dt * dV/dt
            U(t+dt) = U(t) + dt * dU/dt
        
        where:
            dV/dt = V * (a - V) * (V - 1) - U + I_app + Σ(I_syn)
            dU/dt = b * V - c * U
        """
        # Handle scalar inputs
        if np.isscalar(Iapp):
            Iapp = np.full(self.N, Iapp)
        if np.isscalar(Iaff):
            Iaff = np.full(self.N, Iaff)
        
        # Total input current
        I_total = Iapp + Iaff
        
        # Compute derivatives
        # dV/dt = V * (a - V) * (V - 1) - U + I_total + Σ(I_syn)
        dVdt = self.V_prev * (self.a - self.V_prev) * (self.V_prev - 1.0) - self.U_prev
        dUdt = self.b * self.V_prev - self.c * self.U_prev
        
        # Euler integration with time scaling
        self.V += dt * (dVdt + I_total + np.sum(self.I_syn, axis=1)) * self.ts
        self.U += dUdt * dt * self.ts
        
        # Update previous states
        self.V_prev = self.V.copy()
        self.U_prev = self.U.copy()
        
        # Update synaptic currents
        # dI_syn/dt = -I_syn / tau_syn + W * output
        dI_syn = dt * (-self.I_syn * self.tau_syn + self.W * self.syn_output())
        self.I_syn += dI_syn
        
        # Update output
        self.output = self.syn_output()