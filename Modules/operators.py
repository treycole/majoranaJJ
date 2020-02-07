import numpy as np
from numpy import linalg as LA
import constants as const
import lattice as lat

# Descritizing $k_x$ and $k_y$
def k_x(coor, ax, ay):
    n = coor.shape[0]
    k_x = np.zeros((N,N), dtype = "complex")
    NN = lat.NN_Arr(coor, ax, ay)
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
    NN = lat.NN_Arr(coor, ax, ay)
    for i in range(N):
        for j in range(N):
            if NN[j,1] == i:
                k_x[j,i] = 1j/(2*ax)
            if NN[j, 3] == i:
                k_x[j,i] = -1j/(2*ax)
    return k_x
#Descritizing kx^2 and ky^2
def k_x2(coor, ax, ay):
    N = coor.shape[0]
    k_x2 = np.zeros((N,N), dtype='complex')
    NN = lat.NN_Arr(coor, ax, ay)
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
    NN = lat.NN_Arr(coor, ax, ay)
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
#Getting energies
def E0(coor, ax, ay):
    N = coor.shape[0]
    H = H0(coor, ax, ay)
    print (H.shape)
    eigvals, eigvecs = LA.eigh(H)
    return np.sort(eigvals)
#Getting States
def eigstate(coor, ax, ay):
    N = coor.shape[0]
    H = H0(coor, ax, ay)
    eigvals, eigvecs = LA.eigh(H)
    return eigvecs