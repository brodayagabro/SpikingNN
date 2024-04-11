import numpy as np
from solve_izh import Izhikevich_Network, solve_izh
from matplotlib import pyplot as plt
def main():
    W = np.array([
            [0, 0],
            [0.5, 0]   
        ])
    T = np.array([
            [0.1, 0],
            [0, 0.1]
        ])
    a = [0.02, 0.02];
    b = [0.2, 0.2];
    c = [-65, -65];
    d = [8, 8];
    v_peak = [30, 30];

    net = Izhikevich_Network(size=2, W=W, T=T, a=a, b=b, c=c, d=d, v_peak=v_peak)
    T = np.linspace(0, 500, 5000)
    I = np.array([10, 0])*np.ones((len(T), 2))
    v, u, I_syn = solve_izh(T, I, net)
    plt.subplot(121)
    plt.plot(T, v[:, 0])
    plt.plot(T, u[:, 0])
    plt.plot(T, I_syn[:, 0])
    plt.subplot(122)
    plt.plot(T, v[:, 1])
    plt.plot(T, u[:, 1])
    plt.plot(T, I_syn[:, 1])
    plt.show()

if __name__ == "__main__":
    main()
