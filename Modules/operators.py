import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
import constants as const
import lattice as lat

# Descritizing $k_x$ and $k_y$
def k_x(coor, ax, ay):
    N = coor.shape[0]
    k_x = np.zeros((N,N), dtype = "complex")
    NN = lat.NN_Arr(coor)
    for i in range(N):
        for j in range(N):
            if NN[j,0] == i:
                k_x[j,i] = -1j/(2*ax)
            if NN[j, 2] == i:
                k_x[j,i] = 1j/(2*ax)
    return k_x
def k_y(coor, ax, ay):
    N = coor.shape[0]
    k_y = np.zeros((N,N), dtype = "complex")
    NN = lat.NN_Arr(coor)
    for i in range(N):
        for j in range(N):
            if NN[j,1] == i:
                k_y[j,i] = 1j/(2*ax)
            if NN[j, 3] == i:
                k_y[j,i] = -1j/(2*ax)
    return k_x

#Descritizing kx^2 and ky^2
def k_x2(coor, ax, ay):
    N = coor.shape[0]
    k_x2 = np.zeros((N,N), dtype='complex')
    NN = lat.NN_Arr(coor)
    for i in range(N):
        for j in range(N):
            if NN[j,0] == i:
                k_x2[j,i] = -1/ax**2
            if NN[j, 2] == i:
                k_x2[j,i] = -1/ax**2
            if i == j:
                k_x2[j,i] = 2/ax**2
    return k_x2

def k_y2(coor, ax, ay):
    N = coor.shape[0]
    k_y2 = np.zeros((N,N), dtype='complex')
    NN = lat.NN_Arr(coor)
    for i in range(N):
        for j in range(N):
            if NN[j,1] == i:
                k_y2[j,i] = -1/ax**2
            if NN[j, 3] == i:
                k_y2[j,i] = -1/ax**2
            if i == j:
                k_y2[j,i] = 2/ax**2
    return k_y2

#Defining Hamiltonian for simple free particle in the lattice
def H0(coor, ax, ay):
    N = coor.shape[0]
    H = np.zeros((N,N), dtype = 'complex')
    H = const.hbar**2/(2*const.m0)*(k_x2(coor, ax, ay) + k_y2(coor, ax, ay))
    return H

def V_barrier(size, xi, xf, coor):
    N = coor.shape[0]
    V = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i == j and coor[i,0] < xf and coor[i,0] > xi:
                V[i,j] = size
    return V

#Spin orbit coupling, spin energy splitting, size 2Nx2N: 0->N spin up states, N -> 2N spin down states
def H_SOC(coor, ax, ay, V, gamma, alpha):
    H_0 = H0(coor, ax, ay)
    N = H_0.shape[0]
    kx = k_x(coor, ax, ay)
    ky = k_y(coor, ax, ay)
    H = np.zeros((2*N, 2*N))
    toplft = H_0 + gamma*np.eye(N,N) + V
    btmrt = H_0 - gamma*np.eye(N,N) + V
    btmlft = alpha*(1j*kx - ky)
    toprt = alpha*(-1j*kx - ky)
    H = np.block([[toplft, toprt],[btmrt, btmlft]])
    return H   

def state_cplot(coor, states):
    if coor.shape[0] < states.shape[0]:
        N = int(states.shape[0]/2)
        prob_dens = []
        for i in np.arange(0, int(states.shape[0]/2)):
            prob_dens.append(np.square(abs(states[i])) + np.square(abs(states[i+N])))
    else:
        prob_dens = np.square(abs(eigarr))
    print(sum(prob_dens))
    plt.scatter(coor[:,0], coor[:,1], c = prob_dens)
    plt.colorbar()
    plt.show()
