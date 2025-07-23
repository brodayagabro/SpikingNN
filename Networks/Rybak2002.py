import numpy as np
from Izh_net import *
def Rybak2002_Mask():
    # Setting network connections' Mask like scheme(fig. 2)
    M = np.array([
    # source neuron number
    #  0     1     2     3     4     5     6     7     8     9    10    11     target neuron
     [ 0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   1.,   0.,   0.],  # CPG_IN_Flex
     [-1.,   0.,   0.,   0.,   0.,   0.,   0.,  -1.,   0.,   0.,   0.,   0.],  # CPG_N_Flex
     [ 0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,  -1.,   0.,   0.,   0.],  # Ib_IN_Flex
     [ 0.,   1.,   0.,   0.,   0.,  -1.,   0.,   0.,   0.,  -1.,   0.,   0.],  # Ia_IN_Flex
     [ 0.,   1.,  -1.,   0.,   0.,  -1.,   0.,   0.,   0.,  -1.,   0.,   0.],  # MN_Flex
     [ 0.,   0.,   0.,   0.,   1.,   0.,   0.,   0.,   0.,   0.,   0.,  -1.],  # R_Flex
    
    # Extensor module
     [ 0.,   0.,   0.,   1.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.],  # CPG_IN_Ext
     [ 0.,  -1.,   0.,   0.,   0.,   0.,  -1.,   0.,   0.,   0.,   0.,   0.],  # CPG_N_Ext
     [ 0.,   0.,  -1.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.],  # Ib_IN_Ext
     [ 0.,   0.,   0.,  -1.,   0.,   0.,   0.,   1.,   0.,   0.,   0.,  -1.],  # Ia_IN_Ext
     [ 0.,   0.,   0.,  -1.,   0.,   0.,   0.,   1.,  -1.,   0.,   0.,  -1.],  # MN_Ext
     [ 0.,   0.,   0.,   0.,   0.,  -1.,   0.,   0.,   0.,   0.,   1.,   0.]   # R_Ext
    ])
    return M



def Rybak2002_TAU(**kwargs):
    # Settings of synaptic relaxation constants
    ex_tau = kwargs.get('exitatory_tau', 10)
    in_tau = kwargs.get('inhibitory_tau', 20)
    tau_syn = np.array([
    # source neuron number
    # 0      1      2      3      4      5      6      7       8       9      10      11        Target neuron
     [1.,    1.,    1.,    1.,    1.,    1.,    1.,    1      ,1.,     ex_tau,     1.,     1.],  # CPG_IN_Flex
     [in_tau,1.,    1.,    1.,    1.,    1.,    1.,    in_tau,1.,     1.,     1.,     1.],      # CPG_N_Flex
     [1.,    1.,    1.,    1.,    1.,    1.,    1.,    1.,    in_tau, 1.,     1.,     1.],      # Ib_IN_Flex
     [1.,    ex_tau,1.,    1.,    1.,    in_tau,1.,    1.,    1.,     in_tau, 1.,     1.],      # Ia_IN_Flex
     [1.,    ex_tau,in_tau,1.,    1.,    in_tau,1.,    1.,    1.,     in_tau, 1.,     1.],      # MN_Flex
     [1.,    1.,    1.,    1.,    ex_tau,1.,    1.,    1.,    1.,     1.,     1.,     in_tau],  # R_Flex
    
    # Extensor module
     [1.,    1.,    1.,    ex_tau,1.,    1.,    1.,    1.,    1.,    1.,     1.,     1.],       # CPG_IN_Ext
     [1.,    in_tau,1.,    1.,    1.,    1.,    in_tau,1.,    1.,    1.,     1.,     1.],       # CPG_N_Ext
     [1.,    1.,    in_tau,1.,    1.,    1.,    1.,    1.,    1.,    1.,     1.,     1.],       # Ib_IN_Ext
     [1.,    1.,    1.,    in_tau,1.,    1.,    1.,    ex_tau,1.,    1.,     1.,     in_tau],   # Ia_IN_Ext
     [1.,    1.,    1.,    in_tau,1.,    1.,    1.,    ex_tau,in_tau,1.,     1.,     in_tau],   # MN_Ext
     [1.,    1.,    1.,    1.,    1.,    in_tau,1.,    1.,    1.,    1.,     ex_tau, 1.]        # R_Ext
    ])
    return tau_syn

def Rybak2002_Weights(**kwargs):
    # Settings of synaptic weights
    ex_w = kwargs.get('exitatory_w', 0.1)
    in_w = kwargs.get('inhibitory_w', -0.1)
    W = np.array([
    # source neuron number
    #  0      1      2      3      4      5      6      7      8      9     10     11      Target neuron
     [ 0.,    0.,    0.,    0.,    0.,    0.,    0.,    0,   0.,    ex_w,   0.,    0.],   # CPG_IN_Flex
     [in_w,   0.,    0.,    0.,    0.,    0.,    0.,  in_w,   0.,    0.,    0.,    0.],    # CPG_N_Flex
     [ 0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.,  in_w,   0.,    0.,    0.],    # Ib_IN_Flex
     [ 0.,  ex_w,    0.,    0.,    0.,  in_w,    0.,    0.,    0.,  in_w,   0.,    0.],    # Ia_IN_Flex
     [ 0.,  ex_w,  in_w,    0.,    0.,  in_w,    0.,    0.,    0.,  in_w,   0.,    0.],    # MN_Flex
     [ 0.,    0.,    0.,    0.,  ex_w,    0.,    0.,    0.,    0.,    0.,    0.,  in_w],   # R_Flex
    
    # Extensor module
     [ 0.,    0.,    0.,  ex_w,    0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.],    # CPG_IN_Ext
     [ 0.,  in_w,    0.,    0.,    0.,    0.,  in_w,    0.,    0.,    0.,    0.,    0.],    # CPG_N_Ext
     [ 0.,    0.,  in_w,    0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.],    # Ib_IN_Ext
     [ 0.,    0.,    0.,  in_w,    0.,    0.,    0.,  ex_w,    0.,    0.,    0.,  in_w],    # Ia_IN_Ext
     [ 0.,    0.,    0.,  in_w,    0.,    0.,    0.,  ex_w,  in_w,    0.,    0.,  in_w],    # MN_Ext
     [ 0.,    0.,    0.,    0.,    0.,  in_w,    0.,    0.,    0.,    0.,  ex_w,    0.]     # R_Ext
    ])
    return W
    
def Rybak2002Afferents(*args, **kwargs):
    # In this article as I think all afferent weigth ar greate or equal to 0
    # creating default afferent input matrix
    Q_aff=np.array([
        # afferet type                                      # target neuron
        # Ia_Flex, II_Flex, Ib_Flex, Ia_Ext, II_Ext, Ib_Ext
                                                            # Flexor module
         [ .1,     .1,      .1,      0.,     0.,     0. ],          # CPG_IN_Flex
         [ 0.,     .1,      .1,      0.,     0.,     0. ],          # CPG_N_Flex
         [ 0.,     0.,      1.,      0.,     0.,     0. ],          # Ib_IN_Flex
         [ 1.,     0.,      0.,      0.,     0.,     0. ],          # Ia_IN_Flex
         [ .1,     .1,      0.,      0.,     0.,     0. ],          # MN_Flex
         [ 0.,     0.,      0.,      0.,     0.,     0. ],          # R_Flex
    
                                                            # Extensor module
         [ 0.,     0.,      0.,      .1,     .1,     .1 ],          # CPG_IN_Ext
         [ 0.,     0.,      0.,      0.,     .1,     .1 ],          # CPG_N_Ext
         [ 0.,     0.,      0.,      0.,     0.,     1. ],          # Ib_IN_Ext
         [ 0.,     0.,      0.,      1.,     0.,     0. ],          # Ia_IN_Ext
         [ 0.,     0.,      0.,      .1,     .1,     0. ],          # MN_Ext
         [ 0.,     0.,      0.,      0.,     0.,     0. ]           # R_Ext
        
    ])
    return Q_aff

def Rybak2002P(*args, **kwargs):
    P = np.array([
        # 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11
        [ 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,  0],
        [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,  0]
    ])
    return P


def Rybak_2002_names_types():
    # Neuron Names
    names = [
        # Flexor NeuroModule names    # Number
        'CPG_IN_Flex',                # 0 
        'CPG_N_Flex',                 # 1
        'Ib_IN_Flex',                 # 2
        'Ia_IN_Flex',                 # 3
        'MN_Flex',                    # 4
        'R_Flex',                     # 5
        # Extensor NeuroModule names
        'CPG_IN_Ext',                 # 6
        'CPG_N_Ext',                  # 7
        'Ib_IN_Ext',                  # 8
        'Ia_IN_Ext',                  # 9
        'MN_Ext',                     # 10
        'R_Ext',                      # 11
    ]

    # Default types of Network neurons
    types = [ 
        # Flexor NeuroModule types     # Neuron Name
        'CH',                          # CPG_IN_Flex
        'CH',                          # CPG_N_Flex
        'RS',                          # Ib_IN_Flex
        'RS',                          # Ia_IN_Flex
        'RS',                          # MN_Flex
        'RS',                          # R_Flex
        # Extensor NeuroModule types
        'CH',                          # CPG_IN_Ext
        'CH',                          # CPG_N_Ext
        'RS',                          # Ib_IN_Ext
        'RS',                          # Ia_IN_Ext
        'RS',                          # MN_Ext
        'RS'                           # R_Ext
         
    ]
    return names, types
    

def Rybak_2002_network(*args, **kwargs):

    """
        Function creates a IO_Network object, describing dynamycs of Rybak system
    """
    # Quantitaty of neurons
    N = 12

    names,types =  Rybak_2002_names_types()
    # Converting types to Izhikevich neurons params as Default
    A_def, B_def, C_def, D_def = types2params(types)
    
    # Get Izhikevich's neuron params from kwargs
    a = kwargs.get('a', A_def)
    a[1] = 0.002
    a[7] = 0.002
    b = kwargs.get('b', B_def)
    c = kwargs.get('c', C_def)
    d = kwargs.get('d', D_def)

    ## Synaptic settings
    # I
    M = Rybak2002_Mask()
    #print(np.count_nonzero(M))
    
    # II
    tau_syn = Rybak2002_TAU(**kwargs)
    
    # III
    W = Rybak2002_Weights(**kwargs)
    

    # Getting applicated current matrix from kwargs
    # Default current for each neuron
    input_size = kwargs.get('input_size', N)
    Qapp = kwargs.get('Qapp', np.ones((N, input_size)))
    
    # Getting output matrix from kwargs
    # Default current for each neuron
    output_size = kwargs.get('output_size', N)
    P = kwargs.get('P', Rybak2002P(*args, **kwargs))

    
    
    # Getting afferents' weights matrix from kwargs
    # Default current for each neuron
    afferent_size = 6
    Qaff = Rybak2002Afferents(*args, **kwargs)

    
    # Creating an Izhickevich_IO_Network object with Rybak neural network configuration
    Net = Izhikevich_IO_Network(
        N=N,
        M=M,
        a=a, b=b, c=c, d=d,
        names=names,
        input_size=input_size,
        output_size=output_size,
        afferent_size=afferent_size,
        Q_app = Qapp,
        Q_aff = Qaff,
        P = P,
        W = W,
        tau_syn = tau_syn
    )
    return Net

if __name__=="__main__":
    Rybak_2002_network()