from brian2 import *
import numpy as np

# function to display network synapses
def visualise_connectivity(S, tit):
    Ns = len(S.source)
    size = int(len(S.w)**(1/2))
    #print(size)
    Nt = len(S.target)
    fig = figure(figsize=(7, 4))
    title(tit)
    plot(zeros(Ns), arange(Ns), 'ok', ms=15)
    plot(ones(Nt), arange(Nt), 'ok', ms=15)
    for i, j in zip(S.i, S.j):
        if S.w[size*i + j] < 0:
            arrow(x=0, y=i, dx=1, dy=j-i, color='b',
                   linestyle='-.')
        elif S.w[size*i + j] > 0:
            arrow(x=0, y=i, dx=1, dy=j-i, color='r',
                  linestyle='-')
    xticks([0, 1], ['Source', 'Target'])
    ylabel('Neuron index')
    ylim(-1, max(Ns, Nt))
    xlabel('Source neuron index')
    ylabel('Target neuron index')
    show()
    return fig

def draw_sym(Monitor, title):
    l = len(Monitor.v)
    n = 2
    m = l//n
    print(l)
    #fig, axs = subplots(m, n, figsize=(10, l))
    #fig.suptitle(title, fontsize=15)
    for j in range(l):
        plt.plot(Monitor.t/ms, Monitor.v[j], label="v")
        plt.plot(Monitor.t/ms, Monitor.u[j], label="u")
        plt.plot(Monitor.t/ms, Monitor.I[j], "k-", label = "I")
        plt.xlabel(f"t, ms")
        plt.ylabel(f'neuron_{j+1}')
        if ((j+1)%6 == 0):
            plt.title("Motoneuron")
        plt.grid()
        plt.legend()
        plt.show()
    return None