"""
Afferent feedback models based on Prochaska equations.

Implements sensory feedback from muscle spindles (Ia, II) and
Golgi tendon organs (Ib) for closed-loop neuromechanical control.

Reference:
    Markin, S. N., et al. (2010). "Afferent control of locomotor CPG:
    insights from a simple neuromechanical model." Annals of the
    New York Academy of Sciences, 1198(1), 202-213.
    doi: 10.1111/j.1749-6632.2010.05435.x
"""

import numpy as np


class Afferents:
    """
    Full Prochaska-type afferent model with multiple receptor types.
    
    This class implements three types of sensory afferents:
    - Ia: Primary muscle spindle afferents (velocity + length sensitive)
    - II: Secondary muscle spindle afferents (length sensitive)
    - Ib: Golgi tendon organ afferents (force sensitive)
    
    Equations:
    ---------
    Ia afferent activity:
        Ia = k_v * v_norm^p_v + k_dI * d_norm + k_nI * input + const_I
    
    Ib afferent activity:
        Ib = k_f * F_norm
    
    II afferent activity:
        II = k_dII * d_norm + k_nII * input + const_II
    
    where normalized variables are:
        v_norm = max(0, v / L_th)          # normalized velocity
        d_norm = max(0, (L - L_th) / L_th) # normalized length change
        F_norm = max(0, (F - F_th) / F_th) # normalized force
    
    Class Attributes
    ----------------
    p_v : float = 0.6
        Exponent for velocity term in Ia response.
    k_v : float = 6.2
        Gain for velocity term in Ia response.
    k_dI : float = 2.0
        Gain for length term in Ia response.
    k_dII : float = 1.5
        Gain for length term in II response.
    k_nI : float = 0.06
        Gain for neural input in Ia response.
    k_nII : float = 0.06
        Gain for neural input in II response.
    k_f : float = 1.0
        Gain for force term in Ib response.
    L_th : float = 0.059
        Threshold muscle length for afferent activation (m).
    F_th : float = 3.38
        Threshold muscle force for Ib activation (N).
    const_I, const_II : float = 0
        Baseline activity constants.
    
    Examples
    --------
    >>> afferents = Afferents()
    >>> # Ia response to muscle velocity 0.1 m/s, length 0.07 m, input 0.5
    >>> ia = afferents.Ia(v=0.1, L=0.07, input=0.5)
    >>> # Ib response to muscle force 5.0 N
    >>> ib = afferents.Ib(F=5.0)
    >>> # II response to length 0.07 m, input 0.5
    >>> ii = afferents.II(L=0.07, input=0.5)
    """
    
    # Class constants (Prochaska parameters)
    p_v = 0.6
    k_v = 6.2
    k_dI = 2.0
    k_dII = 1.5
    k_nI = 0.06
    k_nII = 0.06
    k_f = 1.0
    L_th = 0.059  # m
    F_th = 3.38   # N
    const_I = 0.0
    const_II = 0.0
    
    def __init__(self) -> None:
        """Initialize afferent model (no instance-specific parameters)."""
        pass
    
    def Ia(self, v: float, L: float, input: float) -> float:
        """
        Calculate Ia afferent activity (primary spindle).
        
        Parameters
        ----------
        v : float
            Muscle velocity (m/s).
        L : float
            Muscle length (m).
        input : float
            Motoneuron activity (arbitrary units).
            
        Returns
        -------
        float
            Ia afferent output (arbitrary units).
        """
        v_norm = np.where(v >= 0, v / self.L_th, 0.0)
        d_norm = np.where(L >= self.L_th, (L - self.L_th) / self.L_th, 0.0)
        return (self.k_v * v_norm**self.p_v + 
                self.k_dI * d_norm + 
                self.k_nI * input + 
                self.const_I)
    
    def Ib(self, F: float) -> float:
        """
        Calculate Ib afferent activity (Golgi tendon organ).
        
        Parameters
        ----------
        F : float
            Muscle force (N).
            
        Returns
        -------
        float
            Ib afferent output (arbitrary units).
        """
        F_norm = np.where(F >= self.F_th, (F - self.F_th) / self.F_th, 0.0)
        return self.k_f * F_norm
    
    def II(self, L: float, input: float) -> float:
        """
        Calculate II afferent activity (secondary spindle).
        
        Parameters
        ----------
        L : float
            Muscle length (m).
        input : float
            Motoneuron activity (arbitrary units).
            
        Returns
        -------
        float
            II afferent output (arbitrary units).
        """
        d_norm = np.where(L >= self.L_th, (L - self.L_th) / self.L_th, 0.0)
        return self.k_dII * d_norm + self.k_nII * input + self.const_II


class Simple_Afferents:
    """
    Simplified afferent model with normalized linear responses.
    
    This is a reduced version of the full Afferents class, useful
    for educational purposes or when computational efficiency is
    prioritized over biological detail.
    
    Equations:
    ---------
    Ia = v_norm = max(0, v / L_th)
    Ib = F_norm = max(0, (F - F_th) / F_th)
    II = d_norm = max(0, (L - L_th) / L_th)
    
    Class Attributes
    ----------------
    L_th : float = 0.059
        Threshold muscle length (m).
    F_th : float = 3.38
        Threshold muscle force (N).
    
    Examples
    --------
    >>> afferents = Simple_Afferents()
    >>> ia = afferents.Ia(v=0.1, L=0.07)  # Returns ~1.69
    >>> ib = afferents.Ib(F=5.0)           # Returns ~0.48
    >>> ii = afferents.II(L=0.07)          # Returns ~0.19
    """
    
    L_th = 0.059  # m
    F_th = 3.38   # N
    
    def __init__(self) -> None:
        """Initialize simplified afferent model."""
        pass
    
    def Ia(self, v: float, L: float, *args) -> float:
        """
        Calculate normalized Ia response (velocity-based).
        
        Parameters
        ----------
        v : float
            Muscle velocity (m/s).
        L : float
            Muscle length (m) - unused in simplified model.
        *args : optional
            Additional arguments (ignored).
            
        Returns
        -------
        float
            Normalized Ia response [0, inf).
        """
        return np.where(v >= 0, v / self.L_th, 0.0)
    
    def Ib(self, F: float, *args) -> float:
        """
        Calculate normalized Ib response (force-based).
        
        Parameters
        ----------
        F : float
            Muscle force (N).
        *args : optional
            Additional arguments (ignored).
            
        Returns
        -------
        float
            Normalized Ib response [0, inf).
        """
        return np.where(F >= self.F_th, (F - self.F_th) / self.F_th, 0.0)
    
    def II(self, L: float, *args) -> float:
        """
        Calculate normalized II response (length-based).
        
        Parameters
        ----------
        L : float
            Muscle length (m).
        *args : optional
            Additional arguments (ignored).
            
        Returns
        -------
        float
            Normalized II response [0, inf).
        """
        return np.where(L >= self.L_th, (L - self.L_th) / self.L_th, 0.0)