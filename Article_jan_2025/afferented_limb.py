from Izh_net import *
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

# Set the random seed for reproducibility
np.random.seed(2000)

Q_app = np.array([
            [0, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 0]
        ])

Q_aff = 0.05*np.random.rand(4, 6)

P = np.array([
            [0, 1, 0, 0],
            [0, 0, 1, 0]
        ])

# Create params' array from neurons
a = 0.001
b = 0.45
c = -45
d = 2

N=4
A = a * np.ones(N) #+ 0.0001 * np.random.randn(N)  # Adding fixed noise to a
A[0] = 0.1; A[3] = 0.1
B = b * np.ones(N)
B[0] = 0.2; B[3] = 0.2
C = c * np.ones(N)
C[0] = -65; C[3] = -65
D = d * np.ones(N)
D[0] = 0.05; D[3] = 0.05

print(A, B, C, D)

net = Izhikevich_IO_Network(input_size = 4, output_size = 2, 
                                afferent_size = 6,
                                N=4, Q_app = Q_app,
                                Q_aff = Q_aff , P = P)
net.set_params(a=A, b=B, c=C, d=D) 
# Neuron_weigths
net.M = np.ones((N, N))
    
# Limb settings

flexor = SimpleAdaptedMuscle(w = 0.5)
extensor = SimpleAdaptedMuscle(w = 0.5)

Limb = OneDOFLimb(q0=np.pi/2+0.4, b=0.001, a1 = 0.4, a2= 0.05, m=0.3, l=0.3)
AL = Afferented_Limb(
                Limb = Limb,
                Flexor = flexor,
                Extensor = extensor
            )

# Creating all system
sys = Net_Limb_connect(Network=net,
                           Limb = AL)

    

def run_sys(sys, T, input):
    V = np.zeros((len(T), N))
    F_flex = np.zeros(len(T))
    F_ext = np.zeros(len(T))
    Afferents = np.zeros((len(T), 6))
    Q = np.zeros(len(T))
    W = np.zeros(len(T))
    dt = T[1] - T[0]
    for i, t in enumerate(T):
        V[i] = sys.net.V_prev
        F_flex[i] = sys.F_flex
        F_ext[i] = sys.F_ext
        Afferents[i] = sys.Limb.output
        Q[i] = sys.q
        sys.step(dt = dt, Iapp = input(t))
        
    return V, F_flex, F_ext, Q, Afferents

# Create the directory if it does not exist
exp_dir = 'exp_afferents'
os.makedirs(exp_dir, exist_ok=True)

Q_dir = f'./{exp_dir}/Q_dir/'     
F_dir = f'./{exp_dir}/F_dir/'
V_dir = f'./{exp_dir}/V_dir/'
Aff_dir=f'./{exp_dir}/Aff_dir'

os.makedirs(Q_dir, exist_ok=True)
os.makedirs(F_dir, exist_ok=True)
os.makedirs(V_dir, exist_ok=True)
os.makedirs(Aff_dir, exist_ok=True)

data = {
    'w1': [],
    'q': [],
    'tau': [],
    'F_path' : [],
    'Q_path' : [],
    'V_path' : [],
    'Aff_path': []
}

data = pd.DataFrame(data)

# prepare simulations
T_max = 10000
time_scale = 150
T = np.linspace(0, T_max, T_max*time_scale)

I = np.zeros(N)
I[1] = 0.1
I[2] = 0
input = lambda t: (I)
np.save(f'./{exp_dir}/T.npy', T)

F = np.zeros((2, len(T)))  

# Define the parameter sets for each Q
w1_values =  np.arange(0.1, 1.1, 0.5)
q_values = np.arange(1, 11, 5)
tau_values = np.arange(1, 101, 15)

Max_proc = len(tau_values)*len(q_values)*len(w1_values)
with tqdm(total=Max_proc) as pbar:
    for II, w1 in enumerate(w1_values):
        for J, q in enumerate(q_values):
            for K, tau in enumerate(tau_values):
                sys.set_init_conditions()
                sys.net.set_weights(np.array([
                    [0, w1*q, 0, -w1],
                    [w1*q, 0, -w1, 0],
                    [0, -w1, 0, w1*q],
                    [-w1, 0, w1*q, 0]
                ])
                )
            
                sys.net.set_synaptic_relax_constant(
                    np.array(
                        [
                            [1, tau, 1, tau],
                            [tau, 1, tau, 1],
                            [1, tau, 1, tau],
                            [tau, 1, tau, 1]
                        ]
                    )
                )
            
                V, F_f, F_e, Q, Aff = run_sys(sys, T, input)
                Q_path = os.path.join(Q_dir, f'{II}_{J}_{K}.npy')
                F_path = os.path.join(F_dir, f'{II}_{J}_{K}.npy')
                V_path = os.path.join(V_dir, f'{II}_{J}_{K}.npy')
                Aff_path = os.path.join(Aff_dir, f'{II}_{J}_{K}.npy')
                np.save(V_path, V)
                np.save(Q_path, Q)
                np.save(Aff_path, Aff)
                F[0] = F_f
                F[1] = F_e
                np.save(F_path, F)
                data.loc[K + len(tau_values)*(J+II*len(q_values))] = [w1, q, tau, F_path, Q_path, V_path, Aff_path]
                # Update the progress bar
                pbar.update(1)
data.to_csv(f'./{exp_dir}/data.csv')
