from Izh_net import *
from net_preparation import *
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
# Running procedure
def run(net, flexor, extensor, Limb, T, Iapp):
    net.set_init_conditions(v_noise=np.random.normal(size=net.N, scale=3))
    flexor.set_init_conditions()
    extensor.set_init_conditions()
    Limb.set_init_conditions()
    dt = T[1] - T[0]
    N = len(net)
    V = np.zeros((len(T), N))
    F_f = np.zeros(len(T))
    F_e = np.zeros(len(T))
    W = np.zeros(len(T))
    Q = np.zeros(len(T))
    for i, t in enumerate(T):
        V[i] = net.V_prev
        F_f[i] = flexor.F_prev
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

def save_img(T, V, F_f, F_e, Q, hf, he, path, net, title):
    plt.figure(figsize=(10, 6))
    mask=T>0
    plt.subplot(4, 2, 1)
    plt.title('neurons')
    for i in range(net.N):
        plt.plot(T[mask], V[:, i][mask], label=net.names[i])
    plt.legend()
    plt.xlabel('t, ms')
    plt.ylabel('V, mV')
    #plt.minorticks_on()
    plt.subplot(4, 2, 3)
    firing_rastr = create_firing_rastr(V, T, 29)
    plt.scatter(firing_rastr[0], firing_rastr[1], s=0.1)
    plt.yticks(list(range(net.N)), net.names)


    plt.subplot(2, 2, 2)
    plt.title(title)
    plt.plot(T[mask], F_f[mask], label='F_flexor')
    plt.plot(T[mask], F_e[mask], label="F_extensor")
    plt.legend()
    plt.xlabel('t, ms')
    plt.ylabel('F, N')
    #plt.minorticks_on()

    plt.subplot(2, 2, 3)
    plt.plot(T[mask], Q[mask])
    plt.xlabel('t, ms')
    plt.ylabel('q, radians')
    #plt.minorticks_on()

    plt.subplot(2, 2, 4)
    M_tot = F_f*hf - F_e*he
    plt.plot(T, M_tot)
    plt.xlabel('t, ms')
    plt.ylabel('Total modent, N*m')
    plt.savefig(path)

def calc_arms(Q, Limb):
    Lf = Limb.L(Q)
    Le = Limb.L(np.pi-Q)
    hf = Limb.h(Lf, Q)
    he = Limb.h(Le, np.pi-Q)
    return hf, he

def var_synaptic_properties(exp_dir, noise_std=1):
    # Creating network
    N = 4
    A = np.array([0.001, 0.001, 0.001, 0.001])
    B = np.array([0.46, 0.46, 0.46, 0.46])
    C = np.array([-50, -50, -50, -50])
    D = np.array([2, 2, 2, 2])
    net = Izhikevich_Network(N=N, a=A, b=B, c=C, d=D)
    net.names=["In_f", "RG_f", "RG_e", "In_e"]
    net.M = np.array([
            [0, 0, 0, 1],
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [1, 0, 0, 0]
        ])

    # Limb settings
    flexor = SimpleAdaptedMuscle(N=5)
    extensor = SimpleAdaptedMuscle(N=5)
    Limb = OneDOFLimb(q0=np.pi/2, w0=0, a1=0.2, a2=0.07, m=0.3, ls=0.3, b=0.01,)


    # Create the directory if it does not exist
    os.makedirs(exp_dir, exist_ok=True)

    Q_dir = f'./{exp_dir}/Q_dir/'     
    F_dir = f'./{exp_dir}/F_dir/'
    Img_dir = f'./{exp_dir}/Img_dir/'
    V_dir = f'./{exp_dir}/V_dir'

    os.makedirs(Q_dir, exist_ok=True)
    os.makedirs(F_dir, exist_ok=True)
    os.makedirs(V_dir, exist_ok=True)
    os.makedirs(Img_dir, exist_ok=True)

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
    T_max = 14000
    time_scale = 3
    T = np.linspace(0, T_max, T_max*time_scale)

    input = 0*np.random.normal(size=(len(T), N), scale=noise_std)
    np.save(f'./{exp_dir}/T.npy', T)

    F = np.zeros((2, len(T)))  

    # Define the parameter sets for each Q
    w1_values =  np.linspace(0.2, 2, 5)
    q_values = np.linspace(1, 5, 5)
    tau_values = np.arange(1, 25, 10)

    Max_proc = len(tau_values)*len(q_values)*len(w1_values)
    with tqdm(total=Max_proc) as pbar:
        for II, w1 in enumerate(w1_values):
            for J, q in enumerate(q_values):
                for K, tau in enumerate(tau_values):

                    net.set_weights(np.array([
                        [0, 0, 0, -q*w1],
                        [-q*w1, 0, -w1, 0],
                        [0, -w1, 0, -q*w1],
                        [-q*w1, 0, 0, 0]
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
                    hf, he = calc_arms(Q, Limb)

                    Q_path = os.path.join(Q_dir, f'{II}_{J}_{K}.npy')
                    F_path = os.path.join(F_dir, f'{II}_{J}_{K}.npy')
                    V_path = os.path.join(V_dir, f'{II}_{J}_{K}.npy')
                    Img_path = os.path.join(Img_dir, f'{II}_{J}_{K}.svg')
                    title=f'w={w1.round(2)}, q={q.round(2)}, tau={tau.round(2)}'
                    save_img(T, V, F_f, F_e, Q, he, hf, Img_path, net, title)
                    np.save(V_path, V)
                    np.save(Q_path, Q)
                    F[0] = F_f
                    F[1] = F_e
                    np.save(F_path, F)
                    data.loc[K + len(tau_values)*(J+II*len(q_values))] = [w1, q, tau, F_path, Q_path, V_path]
                    # Update the progress bar
                    pbar.update(1)
    # saving info about data
    data.to_csv(f'./{exp_dir}/data.csv')

if __name__=="__main__":
    exp_dir = "example_exp"
    var_synaptic_properties(exp_dir, noise_std=1)
