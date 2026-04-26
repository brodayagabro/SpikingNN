"""
Limb mechanics models for neuromechanical simulations.

Implements single-degree-of-freedom limb dynamics with muscle
actuation and optional ground reaction forces.
"""

import numpy as np


class Pendulum:
    """
    Simple pendulum model in gravitational field with angular friction.
    
    Equations of motion:
    -------------------
    J * dw/dt = 0.5 * g * m * ls * cos(q) - b * w + M_applied
    dq/dt = w
    
    where:
        q  : angular position (rad), measured from horizontal
        w  : angular velocity (rad/s)
        J  : moment of inertia = m * ls^2 / 3
        m  : mass (kg)
        ls : length (m)
        b  : angular viscosity coefficient (kg·m²/(ms·rad))
        g  : gravitational acceleration (9.81 m/s²)
        M  : external torque (N·m)
    
    Note: Time is in milliseconds, so a factor of 0.001 is applied
    to convert Newton's equations to ms-scale integration.
    
    Attributes
    ----------
    g : class float = 9.81
        Gravitational acceleration (m/s²).
    m : float
        Limb mass (kg).
    ls : float
        Limb length (m).
    J : float
        Moment of inertia (kg·m²).
    b : float
        Angular viscosity coefficient.
    q, w : float
        Current angular position and velocity.
    q0, w0 : float
        Initial conditions.
    q_prev, w_prev : float
        Previous state (for Euler integration).
    own_T : float
        Natural period of small oscillations (ms).
    
    Examples
    --------
    >>> limb = Pendulum(m=0.3, ls=0.3, b=0.002, q0=np.pi/2)
    >>> limb.set_init_conditions()
    >>> for t in range(1000):
    ...     limb.step(dt=0.1, M=0.01)  # Apply small torque
    ...     print(f"Angle: {limb.q:.3f} rad")
    """
    
    g = 9.81  # m/s²
    
    def __init__(self, m: float = 0.3, ls: float = 0.3, b: float = 0.002, **kwargs):
        """
        Initialize pendulum model.
        
        Parameters
        ----------
        m : float, optional
            Mass in kg (default: 0.3).
        ls : float, optional
            Length in meters (default: 0.3).
        b : float, optional
            Angular viscosity in kg·m²/(ms·rad) (default: 0.002).
        q0 : float, optional
            Initial angle in radians (default: π/2).
        w0 : float, optional
            Initial angular velocity in rad/s (default: 0).
        """
        self.m = m
        self.ls = ls
        self.J = m * ls**2 / 3  # Moment of inertia for uniform rod
        self.b = b
        self.q = kwargs.get('q0', np.pi / 2)
        self.w = kwargs.get('w0', 0.0)
        self.q0 = self.q
        self.w0 = self.w
        self.q_prev = self.q
        self.w_prev = self.w
        # Natural period for small oscillations: T = 2π√(2ls/(3g))
        self.own_T = 2 * np.pi * np.sqrt(2 * ls / (3 * self.g))
    
    def set_init_conditions(self, **kwargs) -> None:
        """
        Reset limb to initial conditions.
        
        Parameters
        ----------
        q0, w0 : optional
            Override initial angle (rad) and velocity (rad/s).
        """
        self.q = kwargs.get('q0', self.q0)
        self.w = kwargs.get('w0', self.w0)
        self.q_prev = self.q
        self.w_prev = self.w
    
    def step(self, dt: float = 0.1, M: float = 0.0) -> tuple[float, float]:
        """
        Perform one integration step using forward Euler method.
        
        Parameters
        ----------
        dt : float, optional
            Time step in milliseconds (default: 0.1).
        M : float, optional
            Applied external torque in N·m (default: 0).
            
        Returns
        -------
        tuple[float, float]
            (angular_velocity, angular_position) after the step.
        """
        # Euler integration with ms-to-s conversion factor (0.001)
        # dw/dt = (0.5*g*m*ls*cos(q) - b*w + M) / J
        self.w = self.w_prev + 0.001 * dt * (
            0.5 * self.g * self.m * self.ls * np.cos(self.q_prev) 
            - self.b * self.w_prev + M
        ) / self.J
        
        # dq/dt = w
        self.q = self.q_prev + 0.001 * dt * self.w_prev
        
        # Update previous states
        self.q_prev = self.q
        self.w_prev = self.w
        
        return self.w, self.q


class OneDOFLimb(Pendulum):
    """
    Single-degree-of-freedom limb with antagonistic muscle actuation.
    
    Extends Pendulum by adding muscle force-to-torque conversion
    based on tendon geometry.
    
    Additional Parameters
    ---------------------
    a1, a2 : float
        Tendon attachment points defining muscle moment arms (m).
    
    Methods
    -------
    L(q) : Calculate muscle length as function of joint angle.
    h(L, q) : Calculate moment arm (lever arm) for force-to-torque conversion.
    
    Examples
    --------
    >>> limb = OneDOFLimb(m=0.3, ls=0.3, a1=0.06, a2=0.007)
    >>> # Apply flexor force 2N, extensor force 1N
    >>> limb.step(dt=0.1, F_flex=2.0, F_ext=1.0)
    """
    
    def __init__(self, **kwargs):
        """
        Initialize limb with muscle geometry.
        
        Parameters
        ----------
        a1, a2 : float, optional
            Tendon attachment geometry parameters in meters
            (default: a1=0.06, a2=0.007).
        **kwargs : optional
            Additional parameters passed to Pendulum.__init__.
        """
        super().__init__(**kwargs)
        self.a1 = kwargs.get('a1', 0.06)
        self.a2 = kwargs.get('a2', 0.007)
        self.M_tot = 0.0  # Total applied torque
    
    def L(self, q: float) -> float:
        """
        Calculate muscle length as function of joint angle.
        
        Based on law of cosines for tendon geometry.
        
        Parameters
        ----------
        q : float
            Joint angle in radians.
            
        Returns
        -------
        float
            Muscle length in meters.
        """
        return np.sqrt(self.a1**2 + self.a2**2 - 2 * self.a1 * self.a2 * np.cos(q))
    
    def h(self, L: float, q: float) -> float:
        """
        Calculate moment arm for force-to-torque conversion.
        
        Parameters
        ----------
        L : float
            Current muscle length (m).
        q : float
            Current joint angle (rad).
            
        Returns
        -------
        float
            Moment arm (lever arm) in meters.
        """
        return self.a1 * self.a2 * np.sin(q) / L
    
    def step(self, dt: float = 0.1, F_flex: float = 0.0, 
             F_ext: float = 0.0, M: float = 0.0) -> tuple[float, float]:
        """
        Step limb dynamics with muscle forces.
        
        Parameters
        ----------
        dt : float, optional
            Time step in ms (default: 0.1).
        F_flex : float, optional
            Flexor muscle force in Newtons (default: 0).
        F_ext : float, optional
            Extensor muscle force in Newtons (default: 0).
        M : float, optional
            Additional external torque in N·m (default: 0).
            
        Returns
        -------
        tuple[float, float]
            (angular_velocity, angular_position) after the step.
        """
        # Calculate muscle lengths and moment arms
        L_flex = self.L(self.q)
        L_ext = self.L(np.pi - self.q)  # Extensor on opposite side
        h_flex = self.h(L_flex, self.q)
        h_ext = self.h(L_ext, np.pi - self.q)
        
        # Convert forces to torques (factor of 10 for unit scaling)
        # Flexor produces positive torque, extensor negative
        self.M_tot = 10 * F_flex * h_flex - 10 * F_ext * h_ext + M
        
        return super().step(dt=dt, M=self.M_tot)


class OneDOFLimb_withGR(OneDOFLimb):
    """
    OneDOFLimb with ground reaction force during stance phase.
    
    Adds a simplified ground reaction torque that activates when
    angular velocity is positive (limb moving into stance).
    
    Additional Attributes
    ---------------------
    M_GRmax : class float = 0.585
        Maximum ground reaction torque magnitude (N·m).
    
    Examples
    --------
    >>> limb = OneDOFLimb_withGR(m=0.3, ls=0.3)
    >>> # Ground reaction automatically activates when w >= 0
    >>> limb.step(dt=0.1, F_flex=2.0, F_ext=1.0)
    """
    
    M_GRmax = 0.585  # N·m, maximum ground reaction torque
    
    def __init__(self, **kwargs):
        """Initialize limb with ground reaction capability."""
        super().__init__(**kwargs)
    
    def GR(self, w: float, q: float) -> float:
        """
        Calculate ground reaction torque.
        
        Activates only during stance phase (w >= 0).
        
        Parameters
        ----------
        w : float
            Angular velocity (rad/s).
        q : float
            Joint angle (rad).
            
        Returns
        -------
        float
            Ground reaction torque in N·m (0 during swing phase).
        """
        return np.where(
            w >= 0,  # Stance phase: limb moving into ground contact
            -self.M_GRmax * np.cos(q),  # Reaction torque
            0.0  # Swing phase: no ground contact
        )
    
    def step(self, dt: float = 0.1, F_flex: float = 0.0, 
             F_ext: float = 0.0) -> tuple[float, float]:
        """
        Step dynamics with automatic ground reaction.
        
        Parameters
        ----------
        dt : float, optional
            Time step in ms (default: 0.1).
        F_flex : float, optional
            Flexor force in N (default: 0).
        F_ext : float, optional
            Extensor force in N (default: 0).
            
        Returns
        -------
        tuple[float, float]
            (angular_velocity, angular_position) after the step.
        """
        # Add ground reaction torque to muscle torques
        gr_torque = self.GR(self.w, self.q)
        return super().step(dt=dt, F_flex=F_flex, F_ext=F_ext, M=gr_torque)