"""
Afferented limb: mechanical limb with sensory feedback.

Combines limb mechanics, muscle dynamics, and afferent feedback
into a unified component for closed-loop neuromechanical simulation.
"""

import numpy as np
from ..mechanics.muscles import SimpleAdaptedMuscle
from ..mechanics.limb import OneDOFLimb
from ..mechanics.afferents import Afferents, Simple_Afferents


class Afferented_Limb:
    """
    Limb with full Prochaska-type afferent feedback.
    
    This class integrates:
    - OneDOFLimb for mechanical dynamics
    - Two SimpleAdaptedMuscle instances (flexor/extensor)
    - Afferents for sensory feedback calculation
    
    The output is a 6-element vector of afferent activities:
    [Ia_flex, II_flex, Ib_flex, Ia_ext, II_ext, Ib_ext]
    
    Attributes
    ----------
    Afferents : Afferents
        Prochaska-type afferent model instance.
    Limb : OneDOFLimb
        Mechanical limb instance.
    Flexor, Extensor : SimpleAdaptedMuscle
        Muscle instances for antagonistic actuation.
    output : np.ndarray
        Current afferent output vector (shape: (6,)).
    F_flex, F_ext : float
        Current flexor/extensor muscle forces (N).
    
    Properties
    ----------
    q : float
        Current joint angle (rad) - delegated to Limb.
    w : float
        Current angular velocity (rad/s) - delegated to Limb.
    
    Examples
    --------
    >>> from spikingnn_core.mechanics import OneDOFLimb, SimpleAdaptedMuscle
    >>> limb = OneDOFLimb(q0=np.pi/2)
    >>> flexor = SimpleAdaptedMuscle(w=0.5)
    >>> extensor = SimpleAdaptedMuscle(w=0.4)
    >>> al = Afferented_Limb(Limb=limb, Flexor=flexor, Extensor=extensor)
    >>> al.set_init_conditions()
    >>> # Step with neural inputs to muscles
    >>> al.step(dt=0.1, uf=5.0, ue=3.0)
    >>> print(f"Afferents: {al.output}")
    """
    
    def __init__(self, 
                 Limb: OneDOFLimb = None,
                 Flexor: SimpleAdaptedMuscle = None,
                 Extensor: SimpleAdaptedMuscle = None):
        """
        Initialize afferented limb system.
        
        Parameters
        ----------
        Limb : OneDOFLimb, optional
            Mechanical limb instance (default: new OneDOFLimb()).
        Flexor : SimpleAdaptedMuscle, optional
            Flexor muscle instance (default: new SimpleAdaptedMuscle()).
        Extensor : SimpleAdaptedMuscle, optional
            Extensor muscle instance (default: new SimpleAdaptedMuscle()).
        """
        if Limb is None:
            Limb = OneDOFLimb()
        if Flexor is None:
            Flexor = SimpleAdaptedMuscle()
        if Extensor is None:
            Extensor = SimpleAdaptedMuscle()
            
        self.Afferents = Afferents()
        self.Limb = Limb
        # Set afferent length threshold based on limb geometry
        self.Afferents.L_th = np.sqrt(self.Limb.a1**2 + self.Limb.a2**2)
        self.Flexor = Flexor
        self.Extensor = Extensor
        
        # Output: [Ia_f, II_f, Ib_f, Ia_e, II_e, Ib_e]
        self.output = np.zeros(6)
        self.F_flex = 0.0
        self.F_ext = 0.0
    
    @property
    def q(self) -> float:
        """Current joint angle (rad)."""
        return self.Limb.q
    
    @property
    def w(self) -> float:
        """Current angular velocity (rad/s)."""
        return self.Limb.w
    
    def calc_afferents(self) -> None:
        """
        Calculate afferent feedback based on current limb state.
        
        This method computes all six afferent signals based on:
        - Muscle lengths and velocities (from limb kinematics)
        - Muscle forces (from muscle dynamics)
        - Intrinsic muscle activation (for spindle sensitivity)
        
        The results are stored in self.output.
        """
        q = self.Limb.q
        w = self.Limb.w
        
        # Flexor muscle kinematics
        L_flex = self.Limb.L(q)
        v_flex = self.Limb.h(L_flex, q) * w  # Velocity = moment_arm * angular_velocity
        
        # Extensor muscle kinematics (opposite side of joint)
        L_ext = self.Limb.L(np.pi - q)
        v_ext = -self.Limb.h(L_ext, np.pi - q) * w  # Negative due to opposite geometry
        
        # Flexor afferents
        self.output[0] = self.Afferents.Ia(v_flex, L_flex, self.Flexor.x)  # Ia
        self.output[1] = self.Afferents.II(L_flex, self.Flexor.x)           # II
        self.output[2] = self.Afferents.Ib(self.F_flex)                      # Ib
        
        # Extensor afferents
        self.output[3] = self.Afferents.Ia(v_ext, L_ext, self.Extensor.x)   # Ia
        self.output[4] = self.Afferents.II(L_ext, self.Extensor.x)          # II
        self.output[5] = self.Afferents.Ib(self.F_ext)                       # Ib
    
    def set_init_conditions(self, **kwargs) -> None:
        """
        Reset all components to initial conditions.
        
        Parameters
        ----------
        **kwargs : optional
            Passed to Limb.set_init_conditions() for angle/velocity reset.
        """
        self.Limb.set_init_conditions(**kwargs)
        self.Flexor.set_init_conditions()
        self.Extensor.set_init_conditions()
    
    def step(self, dt: float = 0.1, uf: float = 0.0, ue: float = 0.0) -> None:
        """
        Perform one integration step for the complete afferented limb.
        
        Execution order:
        1. Update muscle activation and force (Flexor, Extensor)
        2. Update limb mechanics with muscle forces
        3. Calculate afferent feedback based on new state
        
        Parameters
        ----------
        dt : float, optional
            Time step in ms (default: 0.1).
        uf : float, optional
            Neural input to flexor muscle (default: 0.0).
        ue : float, optional
            Neural input to extensor muscle (default: 0.0).
        """
        # Update muscle dynamics
        self.Flexor.step(dt=dt, u=uf)
        self.Extensor.step(dt=dt, u=ue)
        
        # Store current forces for afferent calculation
        self.F_flex = self.Flexor.F
        self.F_ext = self.Extensor.F
        
        # Update limb mechanics with muscle forces
        self.Limb.step(dt=dt, F_flex=self.F_flex, F_ext=self.F_ext)
        
        # Calculate sensory feedback
        self.calc_afferents()


class Simple_Afferented_Limb:
    """
    Afferented limb with simplified afferent model.
    
    Same structure as Afferented_Limb but uses Simple_Afferents
    for reduced computational complexity.
    
    See Afferented_Limb for detailed documentation.
    """
    
    def __init__(self,
                 Limb: OneDOFLimb = None,
                 Flexor: SimpleAdaptedMuscle = None,
                 Extensor: SimpleAdaptedMuscle = None):
        if Limb is None:
            Limb = OneDOFLimb()
        if Flexor is None:
            Flexor = SimpleAdaptedMuscle()
        if Extensor is None:
            Extensor = SimpleAdaptedMuscle()
            
        self.Afferents = Simple_Afferents()
        self.Limb = Limb
        self.Afferents.L_th = np.sqrt(self.Limb.a1**2 + self.Limb.a2**2)
        self.Flexor = Flexor
        self.Extensor = Extensor
        self.output = np.zeros(6)
        self.F_flex = 0.0
        self.F_ext = 0.0
    
    @property
    def q(self) -> float:
        return self.Limb.q
    
    @property
    def w(self) -> float:
        return self.Limb.w
    
    def calc_afferents(self) -> None:
        q = self.Limb.q
        w = self.Limb.w
        
        L_flex = self.Limb.L(q)
        v_flex = self.Limb.h(L_flex, q) * w
        L_ext = self.Limb.L(np.pi - q)
        v_ext = -self.Limb.h(L_ext, np.pi - q) * w
        
        self.output[0] = self.Afferents.Ia(v_flex, L_flex)
        self.output[1] = self.Afferents.II(L_flex)
        self.output[2] = self.Afferents.Ib(self.F_flex)
        
        self.output[3] = self.Afferents.Ia(v_ext, L_ext)
        self.output[4] = self.Afferents.II(L_ext)
        self.output[5] = self.Afferents.Ib(self.F_ext)
    
    def set_init_conditions(self, **kwargs) -> None:
        self.Limb.set_init_conditions(**kwargs)
        self.Flexor.set_init_conditions()
        self.Extensor.set_init_conditions()
    
    def step(self, dt: float = 0.1, uf: float = 0.0, ue: float = 0.0) -> None:
        self.Flexor.step(dt=dt, u=uf)
        self.Extensor.step(dt=dt, u=ue)
        self.F_flex = self.Flexor.F
        self.F_ext = self.Extensor.F
        self.Limb.step(dt=dt, F_flex=self.F_flex, F_ext=self.F_ext)
        self.calc_afferents()