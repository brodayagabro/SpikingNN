import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm import tqdm

# Import framework components
from SpikingNN.core.Izh_net import (
    Izhikevich_IO_Network,
    Afferented_Limb,
    SimpleAdaptedMuscle,
    OneDOFLimb,
    types2params
)
from SpikingNN.core.multi_limb import MultiLimbSystem

# --- 1. Network Configuration (Single Limb HCO) ---

N = 4
names = ["CPG_Flex", "MN_Flex", "CPG_Ext", "MN_Ext"]
types = ['CH', 'FS', 'CH', 'FS']
a, b, c, d = types2params(types)

# Connectivity Mask M[i,j]: connection from j to i
M = np.zeros((4, 4))
M[1, 0] = 1      # CPG_F -> MN_F (Excitatory)
M[3, 2] = 1      # CPG_E -> MN_E (Excitatory)
M[0, 2] = -1     # CPG_E inhibits CPG_F
M[2, 0] = -1     # CPG_F inhibits CPG_E

# Synaptic Weights
W = np.zeros((4, 4))
W[M == 1] = 1.5
W[M == -1] = -1.5

# Synaptic Time Constants
tau_syn = np.ones((4, 4)) * 10
tau_syn[M == -1] = 20

# Input Mapping Q_app: Input 0 -> CPG_F (Neuron 0), Input 1 -> CPG_E (Neuron 2)
Q_app = np.zeros((4, 2))
Q_app[0, 0] = 1.0
Q_app[2, 1] = 1.0

# Output Mapping P: Output 0 <- MN_F (Neuron 1), Output 1 <- MN_E (Neuron 3)
P = np.zeros((2, 4))
P[0, 1] = 1.0
P[1, 3] = 1.0

# Afferent Mapping Q_aff (6 afferents per limb)
# Order: [Ia_F, II_F, Ib_F, Ia_E, II_E, Ib_E]
Q_aff = np.array([
    [-0.1, -0.1, 0, 0, 0, 0],   # CPG_Flex
    [0.1, 0.1, -0.1, 0, 0, 0],  # MN_Flex
    [0, 0, 0, -0.1, -0.1, 0],   # CPG_Ext
    [0, 0, 0, 0.1, 0.1, -0.1]   # MN_Ext
])

# Create Network Object
net = Izhikevich_IO_Network(
    N=N, M=M, a=a, b=b, c=c, d=d, names=names,
    input_size=2, output_size=2, afferent_size=6,
    Q_app=Q_app, Q_aff=Q_aff, P=P, W=W, tau_syn=tau_syn
)

# --- 2. Limb Configuration ---

flexor = SimpleAdaptedMuscle(w=0.5, N=30, tau_c=39, tau_1=21)
extensor = SimpleAdaptedMuscle(w=0.5, N=30, tau_c=39, tau_1=21)
limb_mech = OneDOFLimb(m=0.3, ls=0.3, b=0.01, q0=np.pi/2, a1=0.4, a2=0.05)
limb = Afferented_Limb(Limb=limb_mech, Flexor=flexor, Extensor=extensor)

system = MultiLimbSystem(network=net, limbs=[limb], names=["Leg"])

# --- 3. Simulation Loop ---

print("Initializing simulation...")
np.random.seed(42)

T_sim = 2000  # ms
dt = 0.1      # ms
steps = int(T_sim / dt)
time_vec = np.arange(steps) * dt

# Buffers
V_hist = np.zeros((steps, net.N))
F_flex_hist = np.zeros(steps)
F_ext_hist = np.zeros(steps)
q_hist = np.zeros(steps)
afferents_hist = np.zeros((steps, 6))

Iapp_base = np.array([5.0, 4.9])  # Constant drive

for i in tqdm(range(steps)):
    V_hist[i] = system.net.V_prev
    F_flex_hist[i] = system.limbs[0].F_flex
    F_ext_hist[i] = system.limbs[0].F_ext
    q_hist[i] = system.limbs[0].q
    afferents_hist[i, :] = system.limbs[0].output
    
    noise = np.random.normal(0, 0.05, size=2)
    system.step(dt=dt, Iapp=Iapp_base + noise)

print("Simulation finished.")

# --- 4. Simple Visualization ---

fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)

# Plot 1: Spike Raster
ax = axes[0]
for i in range(net.N):
    spikes = np.where(V_hist[:, i] > 20)[0]
    if len(spikes) > 0:
        ax.scatter(time_vec[spikes], [i]*len(spikes), s=2, color='black')
ax.set_yticks(range(net.N))
ax.set_yticklabels(net.names)
ax.set_title("Spike Raster")
ax.grid(True, alpha=0.3)

# Plot 2: Muscle Forces
ax = axes[1]
ax.plot(time_vec, F_flex_hist, label='Flexor', color='blue')
ax.plot(time_vec, F_ext_hist, label='Extensor', color='red')
ax.set_title("Muscle Forces")
ax.set_ylabel("F (N)")
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 3: Joint Angle
ax = axes[2]
ax.plot(time_vec, q_hist, color='black')
ax.set_title("Joint Angle")
ax.set_ylabel("q (rad)")
ax.grid(True, alpha=0.3)

# Plot 4: AFFERENT ACTIVITY 
ax = axes[3]
for side in ['top', 'right', 'bottom', 'left']:
    ax.spines[side].set_visible(False)
ax.set_xticks([])
ax.set_yticks([])

# Define groups for the three subplots
aff_groups = {
    'D1: Type Ia': [0, 3],  # Indices for Ia Flex, Ia Ext
    'D2: Type II': [1, 4],  # Indices for II Flex, II Ext
    'D3: Type Ib': [2, 5]   # Indices for Ib Flex, Ib Ext
}

# Create a 1x3 grid within the bottom row of the main grid
# Use .get_subplotspec() if 'ax' is an Axes object
gs_aff = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=ax.get_subplotspec(), hspace=0.4, wspace=0.3)

# Iterate through each group to create a subplot
for col_idx, (title, indices) in enumerate(aff_groups.items()):
    ax = fig.add_subplot(gs_aff[0, col_idx])
    
    # Plot Flexor (solid line) and Extensor (dashed line) for the current type
    # Index 0,1,2 are Flexors; Index 3,4,5 are Extensors
    ax.plot(time_vec, afferents_hist[:, indices[0]], label='Flex', color='black', linestyle='-')
    ax.plot(time_vec, afferents_hist[:, indices[1]], label='Ext', color='black', linestyle='--')
    
    # Set titles and labels
    ax.set_title(title)
    ax.set_xlabel("Time (ms)")
    
    # Only show Y-label on the first plot to save space
    if col_idx == 0:
        ax.set_ylabel("Activity (u.e.)")
        
    # Add legend and grid
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Remove top and right spines for a cleaner look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


plt.tight_layout()
plt.show()
