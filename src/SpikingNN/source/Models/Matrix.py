import numpy as np
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
step = int(N/2)
# Create matrix for all the network
Links = np.zeros((N, N), dtype=float)
def add_in_place(target, syllable, x=None, y=None):
    target[y:y+syllable.shape[0], x:x+syllable.shape[1]] += syllable
    return target
    
Links = add_in_place(Links, Interlinks, x=0, y=0)
Links = add_in_place(Links, Interlinks, x=6, y=6)
Links = add_in_place(Links, Externlinks, x=6, y=0)
Links = add_in_place(Links, Externlinks, x=0, y=6)
#print(links)
links = np.asarray(Links).reshape(-1)

if __name__ == "__main__":
    print(links)

