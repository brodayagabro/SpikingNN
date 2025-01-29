from Izh_net import *
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
# Running procedure
def run(net, flexor, extensor, Limb, T, Iapp):
    net.set_init_conditions()
    flexor.set_init_conditions()
    extensor.set_init_conditions()
    Limb.set_init_conditions()
    dt = T[1] - T[0]
    N = len(net)
    #U = np.zeros((len(T), N))
    V = np.zeros((len(T), N))
    #Cn_f = np.zeros(len(T))
    #X_f = np.zeros(len(T))
    F_f = np.zeros(len(T))
    #Cn_e = np.zeros(len(T))
    #X_e = np.zeros(len(T))
    F_e = np.zeros(len(T))
    W = np.zeros(len(T))
    Q = np.zeros(len(T))
    for i, t in enumerate(T):
        #U[i] = net.U_prev
        V[i] = net.V_prev
        #Cn_f[i] = flexor.Cn_prev
        #X_f[i] = flexor.x
        F_f[i] = flexor.F_prev
        #Cn_e[i] = extensor.Cn_prev
        #X_e[i] = extensor.x
        F_e[i] = extensor.F_prev
        Q[i] = Limb.q
        W[i] = Limb.w
        net.step(dt=dt, Iapp=Iapp[i])
        uf = net.output[1]
        ue = net.output[2]
        flexor.step(dt=dt, u=uf)
        extensor.step(dt=dt, u=ue)
        Limb.step(dt=dt, F_flex=flexor.F, F_ext=extensor.F)
    return V, F_f, F_e, Q


# Creating network
N = 4
net = Izhikevich_Network(N=N)
net.set_init_conditions()
a = 0.001
b = 0.45
c = -45
d = 2
N1 = izhikevich_neuron(preset=None, a=a, b=b, c=c, d=d)
N2 = izhikevich_neuron(preset=None, a=a, b=b, c=c, d=d)

A = a * np.ones(N) #+ 0.0001 * np.random.randn(N)  # Adding fixed noise to a
A[0] = 0.1; A[3] = 0.1
B = b * np.ones(N)
B[0] = 0.2; B[3] = 0.2
C = c * np.ones(N)
C[0] = -65; C[3] = -65
D = d * np.ones(N)
D[0] = 0.05; D[3] = 0.05
net.set_params(a = A, b = B, c = C, d = D)

# Limb settings
flexor = SimpleAdaptedMuscle(w=1)
extensor = SimpleAdaptedMuscle(w=1)
Limb = OneDOFLimb(q0=np.pi/2, w0=0, a1=0.2, a2=0.07, m=0.3, ls=0.3, b=0.01,)


# Create the directory if it does not exist
exp_dir = 'exp_long_time_autoocil'
os.makedirs(exp_dir, exist_ok=True)

Q_dir = f'./{exp_dir}/Q_dir/'     
F_dir = f'./{exp_dir}/F_dir/'
V_dir = f'./{exp_dir}/V_dir'

os.makedirs(Q_dir, exist_ok=True)
os.makedirs(F_dir, exist_ok=True)
os.makedirs(V_dir, exist_ok=True)

data = {
    'w1': [],
    'q': [],
    'tau': [],
    'F_path' : [],
    'Q_path' : [],
    'V_path' : []
}

data = pd.DataFrame(data)

# Set the random seed for reproducibility
np.random.seed(2000)

# prepare simulations
T_max = 20000
time_scale = 100
T = np.linspace(0, T_max, T_max*time_scale)

input = np.zeros((len(T), N))#np.random.normal(size=(len(T), N), scale=3)
input[:, 1] = .1
input[:, 2] = 0
np.save(f'./{exp_dir}/T.npy', T)

F = np.zeros((2, len(T)))  

# Define the parameter sets for each Q
w1_values =  np.arange(0.1, 1.1, 0.5)
q_values = np.arange(1, 11, 5)
tau_values = np.arange(1, 101, 10)

Max_proc = len(tau_values)*len(q_values)*len(w1_values)
with tqdm(total=Max_proc) as pbar:
    for II, w1 in enumerate(w1_values):
        for J, q in enumerate(q_values):
            for K, tau in enumerate(tau_values):

                net.set_weights(np.array([
                    [0, w1*q, 0, -w1],
                    [w1*q, 0, -w1, 0],
                    [0, -w1, 0, w1*q],
                    [-w1, 0, w1*q, 0]
                ])
                )
            
                net.set_synaptic_relax_constant(
                    np.array(
                        [
                            [1, tau, 1, tau],
                            [tau, 1, tau, 1],
                            [1, tau, 1, tau],
                            [tau, 1, tau, 1]
                        ]
                    )
                )
            
                V, F_f, F_e, Q = run(net, flexor, extensor, Limb, T, input)
                Q_path = os.path.join(Q_dir, f'{II}_{J}_{K}.npy')
                F_path = os.path.join(F_dir, f'{II}_{J}_{K}.npy')
                V_path = os.path.join(V_dir, f'{II}_{J}_{K}.npy')
                np.save(V_path, V)
                np.save(Q_path, Q)
                F[0] = F_f
                F[1] = F_e
                np.save(F_path, F)
                data.loc[K + len(tau_values)*(J+II*len(q_values))] = [w1, q, tau, F_path, Q_path, V_path]
                # Update the progress bar
                pbar.update(1)
data.to_csv(f'./{exp_dir}/data.csv')
