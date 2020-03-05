import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA

from .. import constants as const
from .. import lattice as lat

#################### Descritizing momomentum operators ##################################

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

#Descritizing kx^2 and ky^2
def k_y(coor, ax, ay):
    N = coor.shape[0]
    k_y = np.zeros((N,N), dtype = "complex")
    NN = lat.NN_Arr(coor)
    for i in range(N):
        for j in range(N):
            if NN[j,1] == i:
                k_y[j,i] = 1j/(2*ay)
            if NN[j, 3] == i:
                k_y[j,i] = -1j/(2*ay)
    return k_y

def k_y2(coor, ax, ay):
    N = coor.shape[0]
    k_y2 = np.zeros((N,N), dtype='complex')
    NN = lat.NN_Arr(coor)
    for i in range(N):
        for j in range(N):
            if NN[j,1] == i:
                k_y2[j,i] = -1/ay**2
            if NN[j, 3] == i:
                k_y2[j,i] = -1/ay**2
            if i == j:
                k_y2[j,i] = 2/ay**2
    return k_y2

#################### Periodic momentum operators ##################################


def kp_x(coor, ax, ay, qx):
    N = coor.shape[0]                           #Number of Lattice sites
    xmin = min(coor[:, 0])                      #To determine the factor in the phase shift for periodic sites
    xmax = max(coor[:, 0])
    NN = lat.NN_Arr(coor)                       #Nearest Neighbor Array
    NNb = lat.NN_Bound(NN, coor)                #Neighbor array for sites on boundary, periodic conditions

    kbx = np.zeros((N,N), dtype = "complex")    #Momentum operator at boundary, contributes phase addition
    for i in range(N):
        for j in range(N):
            if NN[j,0] == i:
                kbx[j,i] = -1j/(2*ax)           #Same as bulk k-operator
            if NN[j, 2] == i:
                kbx[j,i] = 1j/(2*ax)            #Same as bulk k-operator
            if NNb[j, 0] == i:
                kbx[j, i] = (-1j/2*ax)*np.exp(-1j*qx*(xmax-xmin+1)*ax)       #Hopping to next unit cell, e^ik(Lx)
            if NNb[j, 2] == i:
                kbx[j,i] = (1j/2*ax)*np.exp(1j*qx*(xmax-xmin+1)*ax)
    return kbx

def kp_x2(coor, ax, ay, qx):
    N = coor.shape[0]                           #Number of Lattice sites
    xmin = min(coor[:, 0])                      #To determine the factor in the phase shift for periodic sites
    xmax = max(coor[:, 0])
    NN = lat.NN_Arr(coor)                       #Nearest Neighbor Array
    NNb = lat.NN_Bound(NN, coor)                #Neighbor array for sites on boundary, periodic conditions

    kbx = np.zeros((N,N), dtype = "complex")    #Momentum operator at boundary, contributes phase addition
    for i in range(N):
        for j in range(N):
            if NN[j,0] == i:
                kbx[j,i] = -1/ax**2             #Same as bulk k^2-operator
            if NN[j, 2] == i:
                kbx[j,i] = -1/ax**2             #Same as bulk k^2-operator
            if i == j:
                kbx[j,i] = 2/ax**2              #Same as bulk k^2-operator
            if NNb[j, 0] == i:
                kbx[j, i] = (-1/ax**2)*np.exp(-1j*qx*(xmax-xmin+1)*ax)
            if NNb[j, 2] == i:
                kbx[j,i] = (-1/ax**2)*np.exp(1j*qx*(xmax-xmin+1)*ax)

    return kbx

def kp_y(coor, ax, ay, qy):
    N = coor.shape[0]
    ymin = min(coor[:, 1])                      #To determine the factor in the phase shift for periodic sites
    ymax = max(coor[:, 1])
    NN = lat.NN_Arr(coor)
    NNb = lat.NN_Bound(NN, coor)                #Neighbor array for sites on boundary, periodic conditions

    k_y = np.zeros((N,N), dtype = "complex")
    for i in range(N):
        for j in range(N):
            if NN[j,1] == i:
                k_y[j,i] = 1j/(2*ay)
            if NN[j, 3] == i:
                k_y[j,i] = -1j/(2*ay)
            if NNb[j, 1] == i:
                k_y[j, i] = (1j/2*ay)*np.exp(1j*qy*(ymax-ymin+1)*ay)       #Hopping to next unit cell, e^ik(Ny)
            if NNb[j, 3] == i:
                k_y[j,i] = (-1j/2*ay)*np.exp(-1j*qy*(ymax-ymin+1)*ay)
    return k_y

def kp_y2(coor, ax, ay, qy):
    N = coor.shape[0]
    ymin = min(coor[:, 1])                      #To determine the factor in the phase shift for periodic sites
    ymax = max(coor[:, 1])
    NN = lat.NN_Arr(coor)
    NNb = lat.NN_Bound(NN, coor)                #Neighbor array for sites on boundary, periodic conditions

    k_y2 = np.zeros((N,N), dtype='complex')
    for i in range(N):
        for j in range(N):
            if NN[j,1] == i:
                k_y2[j,i] = -1/ay**2
            if NN[j, 3] == i:
                k_y2[j,i] = -1/ay**2
            if i == j:
                k_y2[j,i] = 2/ay**2
            if NNb[j, 1] == i:
                k_y2[j, i] = (-1/ay**2)*np.exp(1j*qy*(ymax-ymin+1)*ay)       #Hopping to next unit cell, e^ik(Nx)
            if NNb[j, 3] == i:
                k_y2[j,i] = (-1/ay**2)*np.exp(-1j*qy*(ymax-ymin+1)*ay)
    return k_y2

###################### Hamiltonians for single unit cell ################################

def H0(coor, ax, ay):
    N = coor.shape[0]
    H = np.zeros((N,N), dtype = 'complex')
    H = (const.xi/2)*(k_x2(coor, ax, ay) + k_y2(coor, ax, ay))
    return H

#Spin orbit coupling, spin energy splitting, size 2Nx2N: 0->N spin up states, N -> 2N spin down states
def H_SOC(coor, ax, ay, V, gamma, alpha):
    H_0 = H0(coor, ax, ay)
    N = H_0.shape[0]
    kx = k_x(coor, ax, ay)
    ky = k_y(coor, ax, ay)
    H = np.zeros((2*N, 2*N), dtype = 'complex')

    H00 = H_0 + gamma*np.eye(N,N) + V
    H11 = H_0 - gamma*np.eye(N,N) + V
    H10 = alpha*(1j*kx - ky)
    H01 = alpha*(-1j*kx - ky)
    H = np.block([[H00, H01],[H10, H11]])
    return H

###################### Hamiltonians with periodic boundary conditions ################################

def H0k(coor, ax, ay, qx = 0, qy = 0):
    N = coor.shape[0]
    H = np.zeros((N,N), dtype = 'complex')
    H = (const.xi/2)*(kp_x2(coor, ax, ay, qx) + kp_y2(coor, ax, ay, qy))
    return H

def H_SOk(coor, ax, ay, qx = 0, qy = 0, V = 0, gamma = 0, alpha = 0):
    H_0 = H0k(coor, ax, ay, qx , qy)
    N = H_0.shape[0]
    kx = kp_x(coor, ax, ay, qx)
    ky = kp_y(coor, ax, ay, qy)
    H = np.zeros((2*N, 2*N), dtype = 'complex')

    H00 = H_0 + gamma*np.eye(N,N) + V
    H11 = H_0 - gamma*np.eye(N,N) + V
    H10 = alpha*(1j*kx - ky)
    H01 = alpha*(-1j*kx - ky)
    H = np.block([[H00, H01], [H10, H11]])
    return H