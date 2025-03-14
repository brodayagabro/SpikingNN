import numpy as np


# Лрокализация пачек спайков
def find_bursts(firings_t, Tmax):
    # интенсивность спайковой активности на электроде
    fc = len(firings_t)/Tmax
    # определние порогового времени для малого пачечного события
    tau_c = min(2/(fc), 10)
    #print(tau_c)
    brusts = []
    brust = [firings_t[0]]
    for t in firings_t[1:]:
        if abs(t-brust[-1]) <= tau_c:
            brust.append(t)
        else:
            brusts.append(brust)
            brust=[t]
    brusts.append(brust)
    return brusts

     
def get_bursts_regions(bursts):
    bursts1 = np.zeros((len(bursts), 2))
    for i, burst in enumerate(bursts):
        bursts1[i] = np.array([burst[0], burst[-1]])
    return bursts1  
    
# Характеристики пачек
def get_brusts_duration(brusts):
    brusts = np.array(brusts)
    durations = []
    for brust in brusts:
        duration = np.max(brust) - np.min(brust)
        durations.append(duration)
    durations_mean = np.mean(durations)
    durations_std = np.std(durations)
    return(durations_mean, durations_std)

def get_brust_frequency(brusts):
    brusts = np.array(brusts)
    freqs = np.array([])
    for brust in brusts:
        brust = np.array(brust)
        #print(brust)
        f = 1/(brust[1:] - brust[:-1])*1000
        #print(f)
        freqs = np.concatenate((freqs, f))
    freqs_mean = np.mean(freqs)
    freqs_std = np.std(freqs)
    return (freqs_mean, freqs_std)

def create_firing_rastr(V, T, V_peak):
    """
Create Spikes from time dependency for all neurons
    """
    firing_idx = np.where(V>V_peak)
    return (T[firing_idx[0]], firing_idx[1])