import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
import constants as const
import lattice as lat

# Descritizing kx
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

#Descritizing kx^2
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

######################################################

#Periodic momentum operators, p is for periodic
def kp_x(qx, coor, ax, ay):
    N = coor.shape[0]                           #Number of Lattice sites
    xmin = min(coor[:, 0])                      #To determine the factor in the phase shift for periodic sites
    xmax = max(coor[:, 0])
    NN = lat.NN_Arr(coor)                       #Nearest Neighbor Array
    NNb = lat.NN_Bound(NN, coor)                #Neighbor array for sites on boundary, periodic conditions

    kbx = np.zeros((N,N), dtype = "complex")    #Momentum operator at boundary, contributes phase addition
    for i in range(N):
        for j in range(N):
            if NN[j,0] == i:
                kbx[j,i] = -1j/(2*ax)           #Same as bulk p-operator
            if NN[j, 2] == i:
                kbx[j,i] = 1j/(2*ax)            #Same as bulk p-operator
            if NNb[j, 0] == i:
                kbx[j, i] = (-1j/2*ax)*np.exp(1j*qx*(xmin-xmax)*ax)       #Hopping to next unit cell, e^ik(Nx)
            if NNb[j, 2] == i:
                kbx[j,i] = (1j/2*ax)*np.exp(1j*qx*(xmax-xmin)*ax)
    return kbx

def kp_x2(qx, coor, ax, ay):
    N = coor.shape[0]                           #Number of Lattice sites
    xmin = min(coor[:, 0])                      #To determine the factor in the phase shift for periodic sites
    xmax = max(coor[:, 0])
    NN = lat.NN_Arr(coor)                       #Nearest Neighbor Array
    NNb = lat.NN_Bound(NN, coor)                #Neighbor array for sites on boundary, periodic conditions

    kbx = np.zeros((N,N), dtype = "complex")    #Momentum operator at boundary, contributes phase addition
    for i in range(N):
        for j in range(N):
            if NN[j,0] == i:
                kbx[j,i] = -1/ax**2             #Same as bulk p^2-operator
            if NN[j, 2] == i:
                kbx[j,i] = -1/ax**2             #Same as bulk p^2-operator
            if i == j:
                kbx[j,i] = 2/ax**2              #Same as bulk p^2-operator
            if NNb[j, 0] == i:
                kbx[j, i] = (-1/ax**2)*np.exp(1j*qx*(xmin-xmax)*ax)
            if NNb[j, 2] == i:
                kbx[j,i] = (-1/ax**2)*np.exp(1j*qx*(xmax-xmin)*ax)

    return kbx

def kp_y(qy, coor, ax, ay):
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
                k_y[j, i] = (1j/2*ay)*np.exp(1j*qy*(ymax-ymin)*ay)       #Hopping to next unit cell, e^ik(Ny)
            if NNb[j, 3] == i:
                k_y[j,i] = (-1j/2*ay)*np.exp(1j*qy*(ymin-ymax)*ay)
    return k_y

def kp_y2(qy, coor, ax, ay):
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
                k_y2[j, i] = (1j/2*ay)*np.exp(1j*qy*(ymax-ymin)*ay)       #Hopping to next unit cell, e^ik(Nx)
            if NNb[j, 2] == i:
                k_y2[j,i] = (-1j/2*ay)*np.exp(1j*qy*(ymin-ymax)*ay)
    return k_y2


######################################################
#Potential Shapes

#Barrier in the range xi --> xf
def V_barrier(size, xi, xf, coor):
    N = coor.shape[0]
    V = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i == j and coor[i,0] < xf and coor[i,0] > xi:
                V[i,j] = size
    return V

#Some sinusoidal frequency oscillating with some defined frequency
def V_periodic(freq, coor):
    N = coor.shape[0]
    V = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            if i==j:
                V[i,j] = np.sin(2*np.pi*coor[i,0]/freq)
    return V

######################################################

#Defining Hamiltonian for simple free particle in the lattice
def H0(coor, ax, ay):
    N = coor.shape[0]
    H = np.zeros((N,N), dtype = 'complex')
    H = const.hbar**2/(2*const.m0)*(k_x2(coor, ax, ay) + k_y2(coor, ax, ay))
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

#e(i qx / Lx) phase acquired when the wf hops to the site off the edge of lattice, complex conjugate when left edge -> right edge
def H0k(qx, qy, coor, ax, ay):
    N = coor.shape[0]
    H = np.zeros((N,N), dtype = 'complex')
    H = const.hbar**2/(2*const.m0)*(kp_x2(qx, coor, ax, ay) + kp_y2(qy, coor, ax, ay))
    return H

def H_SOCk(qx, qy, coor, ax, ay):
    H_0 = H0k(qx, qy, coor, ax, ay)
    N = H_0.shape[0]
    kx = kp_x(qx, coor, ax, ay)
    ky = kp_y(qy, coor, ax, ay)
    H = np.zeros((2*N, 2*N), dtype = 'complex')

    H00 = H_0 + gamma*np.eye(N,N) + V
    H11 = H_0 - gamma*np.eye(N,N) + V
    H10 = alpha*(1j*kx - ky)
    H01 = alpha*(-1j*kx - ky)
    H = np.block([H00, H01], [H10, H11])
    return H

######################################################

def state_cplot(coor, states):
    if coor.shape[0] < states.shape[0]:
        N = int(states.shape[0]/2)
        prob_dens = []
        for i in np.arange(0, int(states.shape[0]/2)):
            prob_dens.append(np.square(abs(states[i])) + np.square(abs(states[i+N])))
    else:
        prob_dens = np.square(abs(states))
    print(sum(prob_dens))
    plt.scatter(coor[:,0], coor[:,1], c = prob_dens)
    plt.colorbar()
    plt.show()

def bands_FP(coor, ax, ay, Nx, Ny, nbands):
    steps = 40
    qx = np.linspace(-np.pi/Nx, np.pi/Nx, steps)
    qy = np.linspace(-np.pi/Ny, np.pi/Ny, steps)
    eigarr = np.zeros((steps, nbands))
    for i in range(steps):
        eigarr[i, :] = LA.eigh(H0k(qx[i], qy[i], coor, ax, ay))[0][:nbands]
    for j in range(eigarr.shape[1]):
        plt.plot(qx, eigarr[:, j], c ='b', linestyle = 'solid')
        plt.show()
