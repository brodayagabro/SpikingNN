import numpy as np
def add_in_place(target, syllable, x=None, y=None):
    target[y:y+syllable.shape[0], x:x+syllable.shape[1]] += syllable
    return target
   
def weigths():
    """
      Function to create weigth's matrix of connections between Rybak neuromodules
      Return type: numpy.ndarray, shape = (N, N), N=12
    """
    # Interconnections
    Interlinks = np.array([
      [0, -0.3, 0, 0, 0, 0],
      [0, 0, 1, 1, 0, 2],
      [0, 0, 0, 0, 0, -0.6],
      [0, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0, -0.6],
      [0, 0, 0, 0, 1, 0]
    ])

    Externlinks = np.array([
      [0, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0, 0], # deleted CPG-N ingibitory synaps
      [0, 0, -0.5, 0, 0, 0],
      [2, 0, 0, -0.5, 0, -0.3],
      [0, 0, 0, 0, -0.5, 0],
      [0, 0, 0, 0, 0, -0.5]
    
    ])

    N = 12
    # Create matrix for all the network
    Links = np.zeros((N, N), dtype=float) 
    Links = add_in_place(Links, Interlinks, x=0, y=0)
    Links = add_in_place(Links, Interlinks, x=6, y=6)
    Links = add_in_place(Links, Externlinks, x=6, y=0)
    Links = add_in_place(Links, Externlinks, x=0, y=6)
    return N, Links

# Поиск пачек
# Лрокализация пачек спайков
def find_brusts(firings_t, Tmax):
    # интенсивность спайковой активности на электроде
    fc = len(firings_t)/Tmax
    # определние порогового времени для малого пачечного события
    tau_c = min(2/(fc), 100)
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

def determ_spikes(firings_t, firings_n, N):
    firings=[]
    for n in range(N):
        firing = []
        for i in range(len(firings_t)):
            if firings_n[i] == n:
                firing.append(firings_t[i])
        
        firings.append(firing)
    return firings
    
def draw_brusts(brusts):
    #plt.figure(figsize=(15, 5))
    for x in brusts:
        plt.axvline(x = min(x), ymin=0, ymax=1, color='red')
        plt.axvline(x = max(x), ymin=0, ymax=1, color='blue')
    #plt.plot(T, v)
    return None

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

if __name__ == "__main__":
  L = weigths()
  print(L)
