import numpy as np
from Izh_net import OneDOFLimb 


class lsim:

    def __init__(self, system):
        self.sys = system;

    def simulate(self, y0, input, T):        
        dt = T[1]-T[0]
        self.sys.set_init_conditions(q0=y0[0], w0=y0[1])
        Y = np.zeros((len(T), len(y0)))
        Y[0] = y0
        for i in range(1, len(T)):
            Y[i] = self.sys.step(dt=dt, 
                    F_flex=input[i, 0], F_ext=input[i, 1])
        return Y




if __name__=="__main__":
    import matplotlib.pyplot as plt
    Limb = OneDOFLimb(m=0.3, l=0.3, a1=0.2, a2=0.07, q0=np.pi/2)
    LSIM = lsim(Limb)
    time =50*1000
    sample_rate=4
    T = np.linspace(0, time, time*sample_rate)
    y0 =[0, np.pi/4]
    INPUT = np.zeros((len(T), len(y0)))
    A = 0.1
    w = np.pi*0.001
    INPUT[:, 0] = A*(1+np.sin(w*T))
    INPUT[:, 1] = A*(1+np.sin(w*T + np.pi))
    Y = LSIM.simulate(y0, INPUT, T)
    plt.figure()
    plt.plot(T, Y[:, 0])
    plt.plot(T, Y[:, 1])
    
    plt.figure()
    plt.plot(T, INPUT[:, 0])
    plt.plot(T, INPUT[:, 1])
    plt.show()
