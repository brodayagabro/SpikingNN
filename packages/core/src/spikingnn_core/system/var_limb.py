"""
Var_Limb: Integrated neural-mechanical system.

Combines spiking neural networks with afferented limb mechanics
to create closed-loop sensorimotor systems for locomotion research.
"""

import numpy as np
from ..models.izhikevich import Izhikevich_IO_Network
from .afferented_limb import Afferented_Limb, Simple_Afferented_Limb


class Net_Limb_connect:
    """
    Base class for connecting neural network to limb mechanics.
    
    This class establishes the interface between:
    - A spiking neural network with I/O matrices (Izhikevich_IO_Network)
    - An afferented limb with sensory feedback
    
    The connection pattern:
    1. Network outputs (2 channels) → Muscle inputs (uf, ue)
    2. Limb afferents (6 channels) → Network afferent inputs
    
    Attributes
    ----------
    net : Izhikevich_IO_Network
        The spiking neural network with I/O projection matrices.
    Limb : Afferented_Limb or Simple_Afferented_Limb
        The mechanical limb with sensory feedback.
    
    Properties
    ----------
    V, U : np.ndarray
        Network membrane potentials and recovery variables.
    I_syn : np.ndarray
        Network synaptic currents.
    F_flex, F_ext : float
        Current muscle forces (delegated to Limb).
    q, w : float
        Current limb angle and velocity (delegated to Limb).
    
    Examples
    --------
    >>> from spikingnn_core import Izhikevich_IO_Network
    >>> from spikingnn_core.system import Afferented_Limb
    >>> 
    >>> # Create network with 2 inputs, 2 outputs, 6 afferents
    >>> net = Izhikevich_IO_Network(
    ...     input_size=2, output_size=2, afferent_size=6, N=4
    ... )
    >>> limb = Afferented_Limb()
    >>> system = Net_Limb_connect(Network=net, Limb=limb)
    >>> 
    >>> # Run closed-loop simulation
    >>> for t in range(1000):
    ...     Iapp = np.array([5.0, 5.0])  # Constant input to network
    ...     system.step(dt=0.1, Iapp=Iapp)
    ...     print(f"Angle: {system.q:.3f}, Force_flex: {system.F_flex:.3f}")
    """
    
    def __init__(self,
                 Network: Izhikevich_IO_Network = None,
                 Limb: Afferented_Limb = None):
        """
        Initialize neural-mechanical system.
        
        Parameters
        ----------
        Network : Izhikevich_IO_Network, optional
            Neural network with I/O matrices (default: new 4-neuron network).
        Limb : Afferented_Limb, optional
            Afferented limb instance (default: new Afferented_Limb()).
        """
        if Network is None:
            Network = Izhikevich_IO_Network(
                input_size=2, output_size=2, afferent_size=6, N=4
            )
        if Limb is None:
            Limb = Afferented_Limb()
            
        self.net = Network
        # Initialize network with small random noise
        self.net.set_init_conditions(
            v_noise=np.random.normal(size=self.net.N, scale=0.5)
        )
        self.Limb = Limb
    
    @property
    def V(self) -> np.ndarray:
        """Network membrane potentials."""
        return self.net.V_prev
    
    @property
    def U(self) -> np.ndarray:
        """Network recovery variables."""
        return self.net.U_prev
    
    @property
    def I_syn(self) -> np.ndarray:
        """Network synaptic currents."""
        return self.net.I_syn
    
    @property
    def F_flex(self) -> float:
        """Flexor muscle force (N)."""
        return self.Limb.Flexor.F_prev
    
    @property
    def F_ext(self) -> float:
        """Extensor muscle force (N)."""
        return self.Limb.Extensor.F_prev
    
    @property
    def q(self) -> float:
        """Limb joint angle (rad)."""
        return self.Limb.q
    
    @property
    def w(self) -> float:
        """Limb angular velocity (rad/s)."""
        return self.Limb.w
    
    def set_init_conditions(self, **kwargs) -> None:
        """
        Reset both network and limb to initial conditions.
        
        Parameters
        ----------
        **kwargs : optional
            Passed to both net.set_init_conditions() and 
            Limb.set_init_conditions().
        """
        self.net.set_init_conditions(**kwargs)
        self.Limb.set_init_conditions(**kwargs)
    
    def step(self, dt: float = 0.1, Iapp: np.ndarray = 0.0) -> None:
        """
        Perform one closed-loop integration step.
        
        Execution order:
        1. Network step: receives afferent feedback, produces motor outputs
        2. Limb step: receives motor commands, produces new sensory feedback
        
        Parameters
        ----------
        dt : float, optional
            Time step in ms (default: 0.1).
        Iapp : np.ndarray or float, optional
            External input to network (shape: (input_size,) or scalar).
        """
        # Network receives afferent feedback from limb
        self.net.step(dt=dt, Iapp=Iapp, Iaff=self.Limb.output)
        
        # Limb receives motor commands from network outputs
        # V_out[0] → flexor, V_out[1] → extensor
        self.Limb.step(dt=dt, uf=self.net.V_out[0], ue=self.net.V_out[1])


class Var_Limb(Net_Limb_connect):
    """
    Extended neural-mechanical system with named components.
    
    Adds named access to neurons, muscles, and afferents for
    easier configuration and debugging.
    
    Additional Attributes
    ---------------------
    names : dict
        Dictionary mapping component types to name lists:
        - 'neurons': list of neuron names
        - 'muscles': ['Flexor', 'Extensor']
        - 'afferents': ['Ia_Flex', 'II_Flex', 'Ib_Flex', 
                       'Ia_Ext', 'II_Ext', 'Ib_Ext']
    
    Examples
    --------
    >>> sys = Var_Limb(Network=net, Limb=limb)
    >>> # Set connection by name instead of index
    >>> sys.set_weights_by_names('CPG1', 'MN1', new_weight=0.4)
    >>> # Set afferent weight by name
    >>> sys.set_afferents_by_names('Ia_Flex', 'CPG1', new_weight=1.0)
    >>> # Adjust muscle parameters by name
    >>> sys.set_muscle_params('Flexor', A=0.4)
    """
    
    def __init__(self, **kwargs):
        """Initialize with named component registry."""
        super().__init__(**kwargs)
        self.names = {
            "neurons": self.net.names,
            "muscles": ["Flexor", "Extensor"],
            "afferents": [
                # Flexor sensors
                "Ia_Flex", "II_Flex", "Ib_Flex",
                # Extensor sensors
                "Ia_Ext", "II_Ext", "Ib_Ext"
            ]
        }
    
    def set_weights_by_names(self, *args, **kwargs) -> None:
        """
        Set synaptic weight by neuron names (delegates to network).
        
        Parameters
        ----------
        *args, **kwargs
            Passed to self.net.set_weights_by_names().
        """
        try:
            self.net.set_weights_by_names(*args, **kwargs)
        except AttributeError:
            print("Error: set_weights_by_names not available in network")
    
    @property
    def SynapticWeights(self) -> np.ndarray:
        """Network synaptic weight matrix."""
        return self.net.W
    
    @property
    def SynapticRelaxation(self) -> np.ndarray:
        """Network synaptic time constants."""
        return self.net.tau_syn
    
    @property
    def AfferentWeights(self) -> np.ndarray:
        """Network afferent projection matrix (Q_aff)."""
        return self.net.Q_aff
    
    def set_afferents_by_names(self, 
                               afferent_name: str,
                               neuron_name: str,
                               new_weight: float) -> bool:
        """
        Set afferent-to-neuron weight by component names.
        
        Parameters
        ----------
        afferent_name : str
            Name from self.names['afferents'].
        neuron_name : str
            Name from self.names['neurons'].
        new_weight : float
            New weight value for Q_aff matrix.
            
        Returns
        -------
        bool
            True if weight was set, False if names not found.
        """
        try:
            afferent_idx = self.names["afferents"].index(afferent_name)
            neuron_idx = self.names["neurons"].index(neuron_name)
            self.net.Q_aff[neuron_idx, afferent_idx] = new_weight
            return True
        except ValueError:
            return False
    
    def set_muscle_params(self, muscle_name: str, **params) -> None:
        """
        Set muscle parameters by muscle name.
        
        Parameters
        ----------
        muscle_name : str
            'Flexor' or 'Extensor'.
        **params : optional
            Parameters to pass to muscle.set_params().
        """
        if muscle_name == "Flexor":
            self.Limb.Flexor.set_params(**params)
        elif muscle_name == "Extensor":
            self.Limb.Extensor.set_params(**params)
        else:
            print(f"Muscle '{muscle_name}' not found")


def run(net, flexor, extensor, Limb, T: np.ndarray, Iapp) -> tuple:
    """
    Run open-loop simulation of limb controlled by two muscles.
    
    This function performs a forward simulation of the neuromechanical
    system without afferent feedback (open-loop). It is useful for
    testing muscle/limb dynamics independently of sensory feedback.
    
    Parameters
    ----------
    net : Izhikevich_Network or similar
        Neural network providing motor commands.
    flexor, extensor : SimpleAdaptedMuscle
        Muscle instances for antagonistic actuation.
    Limb : OneDOFLimb or similar
        Mechanical limb instance.
    T : np.ndarray
        Time vector in ms.
    Iapp : callable
        Function Iapp(t) returning external input to network.
        
    Returns
    -------
    tuple of np.ndarray
        (U, V, Cn_f, X_f, F_f, Cn_e, X_e, F_e, W, Q) where:
        - U, V : network states (shape: (len(T), N))
        - Cn_f, X_f, F_f : flexor activation, normalized activation, force
        - Cn_e, X_e, F_e : extensor equivalents
        - W, Q : limb angular velocity and position
    """
    dt = T[1] - T[0]
    N = len(net)
    
    # Pre-allocate output arrays
    U = np.zeros((len(T), N))
    V = np.zeros((len(T), N))
    Cn_f = np.zeros(len(T))
    X_f = np.zeros(len(T))
    F_f = np.zeros(len(T))
    Cn_e = np.zeros(len(T))
    X_e = np.zeros(len(T))
    F_e = np.zeros(len(T))
    W = np.zeros(len(T))  # Angular velocity
    Q = np.zeros(len(T))  # Angular position
    
    # Mixing coefficients for motoneuron outputs
    alpha_f = 1.0
    alpha_e = 1.0
    
    for i, t in enumerate(T):
        # Record current states
        U[i] = net.U_prev
        V[i] = net.V_prev
        Cn_f[i] = flexor.Cn_prev
        X_f[i] = flexor.x
        F_f[i] = flexor.F_prev
        Cn_e[i] = extensor.Cn_prev
        X_e[i] = extensor.x
        F_e[i] = extensor.F_prev
        Q[i] = Limb.q
        W[i] = Limb.w
        
        # Network step (open-loop: no afferent feedback)
        net.step(dt=dt, Iapp=Iapp(t))
        
        # Combine network outputs for muscle control
        # (alpha parameters allow mixing of different output channels)
        uf = alpha_f * net.output[0] + (1 - alpha_f) * net.output[2]
        ue = alpha_f * net.output[2] + (1 - alpha_e) * net.output[3]
        
        # Muscle steps
        flexor.step(dt=dt, u=uf)
        extensor.step(dt=dt, u=ue)
        
        # Limb step with muscle forces
        Limb.step(dt=dt, F_flex=flexor.F, F_ext=extensor.F)
    
    return U, V, Cn_f, X_f, F_f, Cn_e, X_e, F_e, W, Q