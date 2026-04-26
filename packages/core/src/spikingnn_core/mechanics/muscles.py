"""
Muscle dynamics models for neuromechanical simulations.

Based on simplified muscle activation dynamics with adaptation.
"""

import numpy as np


class SimpleAdaptedMuscle:
    """
    Simplified model of an adapted muscle with activation dynamics.
    
    This model implements a first-order activation dynamics with
    nonlinear force-length-velocity relationships.
    
    Equations:
    ---------
    Muscle activation:
        dCn/dt + Cn/tau_c = w * u(t)
    
    Activation-to-force conversion:
        x = Cn^m / (Cn^m + k^m)
    
    Force dynamics:
        dF/dt + F/tau_1 = A * N * x
    
    where:
        Cn : neural activation variable
        x  : normalized activation [0, 1]
        F  : muscle force output
        u  : neural input (spike rate or voltage)
        w  : neuron-muscle synaptic weight
        A  : force scaling factor
        N  : number of motor units
        tau_c, tau_1 : time constants (stored as 1/tau internally)
        m, k : Hill-type activation curve parameters
    
    Attributes
    ----------
    w : float
        Synaptic weight from neuron to muscle.
    A : float
        Force scaling constant.
    N : int
        Number of motor units.
    tau_c : float
        Inverse time constant for activation dynamics (1/ms).
    tau_1 : float
        Inverse time constant for force dynamics (1/ms).
    m : float
        Hill coefficient for activation curve.
    k : float
        Half-activation constant for activation curve.
    Cn : float
        Current neural activation state.
    Cn_prev : float
        Previous activation state (for Euler integration).
    F : float
        Current muscle force output.
    F_prev : float
        Previous force output.
    x : float
        Current normalized activation [0, 1].
    
    Examples
    --------
    >>> muscle = SimpleAdaptedMuscle(w=0.5, N=10)
    >>> muscle.set_init_conditions()
    >>> for t in range(100):
    ...     muscle.step(dt=0.1, u=5.0)  # u in arbitrary units
    ...     print(f"Force: {muscle.F:.3f}")
    """
    
    def __init__(self, **kwargs):
        """
        Initialize muscle model.
        
        Parameters
        ----------
        w : float, optional
            Neuron-muscle synaptic weight (default: 0.5).
        A : float, optional
            Force scaling factor (default: 0.0074).
        N : int, optional
            Number of motor units (default: 10).
        tau_c : float, optional
            Activation time constant in ms (default: 71).
        tau_1 : float, optional
            Force time constant in ms (default: 130).
        m : float, optional
            Hill coefficient (default: 2.5).
        k : float, optional
            Half-activation constant (default: 0.75).
        """
        self.w = kwargs.get('w', 0.5)
        self.A = kwargs.get('A', 0.0074)
        self.N = kwargs.get('N', 10)
        # Store as inverse time constants for computational efficiency
        self.tau_c = 1.0 / kwargs.get('tau_c', 71)  # 1/ms
        self.tau_1 = 1.0 / kwargs.get('tau_1', 130)  # 1/ms
        self.m = kwargs.get('m', 2.5)
        self.k = kwargs.get('k', 0.75)
        
        # State variables
        self.Cn = 0.0
        self.Cn_prev = 0.0
        self.F = 0.0
        self.F_prev = 0.0
        self.x = 0.0
    
    def set_init_conditions(self) -> None:
        """Reset all state variables to zero."""
        self.Cn = 0.0
        self.Cn_prev = 0.0
        self.F = 0.0
        self.F_prev = 0.0
        self.x = 0.0
    
    def set_params(self, **kwargs) -> None:
        """
        Update muscle parameters.
        
        Parameters
        ----------
        w, A, N, tau_c, tau_1, m, k : optional
            Any subset of parameters to update.
        """
        self.w = kwargs.get('w', self.w)
        self.A = kwargs.get('A', self.A)
        self.N = kwargs.get('N', self.N)
        self.tau_c = 1.0 / kwargs.get('tau_c', 1.0 / self.tau_c)
        self.tau_1 = 1.0 / kwargs.get('tau_1', 1.0 / self.tau_1)
        self.m = kwargs.get('m', self.m)
        self.k = kwargs.get('k', self.k)
    
    def step(self, dt: float = 0.1, u: float = 0.0) -> None:
        """
        Perform one integration step using forward Euler method.
        
        Parameters
        ----------
        dt : float, optional
            Time step in milliseconds (default: 0.1).
        u : float, optional
            Neural input signal (default: 0.0).
        """
        # Activation dynamics: dCn/dt = w*u - Cn/tau_c
        self.Cn = self.Cn_prev + dt * (self.w * u - self.Cn_prev * self.tau_c)
        
        # Nonlinear activation-to-force conversion (Hill-type)
        self.x = self.Cn**self.m / (self.Cn**self.m + self.k**self.m)
        
        # Force dynamics: dF/dt = A*N*x - F/tau_1
        self.F = self.F_prev + dt * (self.A * self.N * self.x - self.F_prev * self.tau_1)
        
        # Update previous states
        self.F_prev = self.F
        self.Cn_prev = self.Cn