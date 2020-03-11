import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA

from . import constants as const
from . import lattice as lat

#################### Descritizing k operators ##################################

def kx(coor, ax, ay):
    N = coor.shape[0]
    k = np.zeros((N,N), dtype = "complex")
    NN = lat.NN_Arr(coor)
    for i in range(N):
        for j in range(N):
            if NN[j,0] == i:
                k[j,i] = -1j/(2*ax)
            if NN[j, 2] == i:
                k[j,i] = 1j/(2*ax)
    return k

def kx2(coor, ax, ay):
    N = coor.shape[0]
    k = np.zeros((N,N), dtype='complex')
    NN = lat.NN_Arr(coor)
    for i in range(N):
        for j in range(N):
            if NN[j,0] == i:
                k[j,i] = -1/ax**2
            if NN[j, 2] == i:
                k[j,i] = -1/ax**2
            if i == j:
                k[j,i] = 2/ax**2
    return k

def ky(coor, ax, ay):
    N = coor.shape[0]
    k = np.zeros((N,N), dtype = "complex")
    NN = lat.NN_Arr(coor)
    for i in range(N):
        for j in range(N):
            if NN[j,1] == i:
                k[j,i] = 1j/(2*ay)
            if NN[j, 3] == i:
                k[j,i] = -1j/(2*ay)
    return k

def ky2(coor, ax, ay):
    N = coor.shape[0]
    k = np.zeros((N,N), dtype='complex')
    NN = lat.NN_Arr(coor)
    for i in range(N):
        for j in range(N):
            if NN[j,1] == i:
                k[j,i] = -1/ay**2
            if NN[j, 3] == i:
                k[j,i] = -1/ay**2
            if i == j:
                k[j,i] = 2/ay**2
    return k


#################### Periodic k operators ##################################

def kpx(coor, ax, ay, qx = 0):
    N = coor.shape[0]   #Number of Lattice sites
    xmin = min(coor[:, 0])  #To determine the factor in the phase shift for periodic sites
    xmax = max(coor[:, 0])
    Lx = (xmax - xmin + 1)*ax

    NN = lat.NN_Arr(coor)    #Nearest Neighbor Array
    NNb = lat.NN_Bound(NN, coor)  #Neighbor array for sites on boundary, periodic conditions

    k = np.zeros((N,N), dtype = "complex")
    for i in range(N):
        for j in range(N):
            if NN[j,0] == i:
                k[j,i] = -1j/(2*ax)
            if NN[j, 2] == i:
                k[j,i] = 1j/(2*ax)
            if NNb[j, 0] == i:
                k[j, i] = (-1j/2*ax)*np.exp(-1j*qx*(Lx))       #Hopping to next unit cell, e^ik(Lx+1)
            if NNb[j, 2] == i:
                k[j,i] = (1j/2*ax)*np.exp(1j*qx*(Lx))
    return k

def kpx2(coor, ax, ay, qx = 0):
    N = coor.shape[0]
    xmin = min(coor[:, 0])
    xmax = max(coor[:, 0])
    Lx = (xmax - xmin + 1)*ax

    NN = lat.NN_Arr(coor)
    NNb = lat.NN_Bound(NN, coor)

    k = np.zeros((N,N), dtype = "complex")
    for i in range(N):
        for j in range(N):
            if NN[j,0] == i:
                k[j,i] = -1/ax**2
            if NN[j, 2] == i:
                k[j,i] = -1/ax**2
            if i == j:
                k[j,i] = 2/ax**2
            if NNb[j, 0] == i:
                k[j, i] = (-1/ax**2)*np.exp(-1j*qx*(Lx))
            if NNb[j, 2] == i:
                k[j,i] = (-1/ax**2)*np.exp(1j*qx*(Lx))
    return k

def kpy(coor, ax, ay, qy = 0):
    N = coor.shape[0]
    ymin = min(coor[:, 1])
    ymax = max(coor[:, 1])
    Ly = (ymax - ymin + 1)*ay

    NN = lat.NN_Arr(coor)
    NNb = lat.NN_Bound(NN, coor)

    k = np.zeros((N,N), dtype = "complex")
    for i in range(N):
        for j in range(N):
            if NN[j,1] == i:
                k[j,i] = 1j/(2*ay)
            if NN[j, 3] == i:
                k[j,i] = -1j/(2*ay)
            if NNb[j, 1] == i:
                k[j, i] = (1j/2*ay)*np.exp(1j*qy*(Ly))
            if NNb[j, 3] == i:
                k[j,i] = (-1j/2*ay)*np.exp(-1j*qy*(Ly))
    return k

def kpy2(coor, ax, ay, qy = 0):
    N = coor.shape[0]
    ymin = min(coor[:, 1])
    ymax = max(coor[:, 1])
    Ly = (ymax - ymin + 1)*ay

    NN = lat.NN_Arr(coor)
    NNb = lat.NN_Bound(NN, coor)

    k = np.zeros((N,N), dtype='complex')
    for i in range(N):
        for j in range(N):
            if NN[j,1] == i:
                k[j,i] = -1/ay**2
            if NN[j, 3] == i:
                k[j,i] = -1/ay**2
            if i == j:
                k[j,i] = 2/ay**2
            if NNb[j, 1] == i:
                k[j, i] = (-1/ay**2)*np.exp(1j*qy*(Ly))
            if NNb[j, 3] == i:
                k[j,i] = (-1/ay**2)*np.exp(-1j*qy*(Ly))
    return k

###################### Delta Operator ###############################
def delta(coor, Wsc, WJ, Sx = 0, Sy = 0, cutx = 0, cuty = 0):
    if (2*Wsc + WJ) != max(coor[:,1]):
        print("Need proper junction and superconducting widths")
        return
    N = coor.shape[0]
    Del = np.zeros(N, N, dtype = 'complex')
    for i in range(N):
        y = coor[i,1]
        x = coor[i,0]
        if y <= Wsc:
            Del[i,i] = delta*np.exp(-1j*phi/2)
        if y > Wsc and y <= (Wsc+Wj):
            Del[i,i] = 0
        if y >(Wsc+WJ):
            Del[i,j] = del*np.exp(1j*phi/2)
    #Delta = delta*np.eye(N, N, dtype = 'complex')
    D00 = np.zeros((N,N))
    D11 = np.zeros((N,N))
    D01 = Del
    D10 = -Del
    Delta = np.block([[D00, D01], [D10, D11]])
    return Delta

###################### Hamiltonians ################################
"""
This is the Hamiltonian with only Spin Orbit Coupling and no Superconductivity.
The parameter PERIODIC determines whether the function calls a constuction of SOC Hamiltonian
with boundary conditions or without boundary conditions. The way that the SOC Hamiltonian is
defined it shouldn't matter which constructor it calls.

"""

def H0(
    coor, ax, ay,
    potential = 0,
    mu = 0,
    gammax = 0, gammay = 0, gammaz = 0,
    alpha = 0,
    qx = 0, qy = 0,
    periodicx = 'no', periodicy = 'no'
    ):

    if periodicx.lower() == 'yes':
        k_x = kpx(coor, ax, ay, qx)
        k_x2 = kpx2(coor, ax, ay, qx)
    if periodicx.lower() == 'no':
        k_x = kx(coor, ax, ay)
        k_x2 = kx2(coor, ax, ay)
    if periodicy.lower() == 'yes':
        k_y = kpy(coor, ax, ay, qy)
        k_y2 = kpy2(coor, ax, ay, qy)
    if periodicy.lower() == 'no':
        k_y = ky(coor, ax, ay)
        k_y2 = ky2(coor, ax, ay)
        
    N = coor.shape[0]
    Hfree = np.zeros((N,N), dtype = 'complex') #free hamiltonian should be real, but just in case
    Hfree = (const.xi/2)*(k_x2 + k_y2)

    V = potential
    MU = mu*np.eye(N,N)

    H00 = Hfree + gammaz*np.eye(N,N) + V - MU
    H11 = Hfree - gammaz*np.eye(N,N) + V - MU
    H10 = alpha*(1j*k_x - k_y) + gammax*np.eye(N,N) + 1j*gammay*np.eye(N,N)
    H01 = alpha*(-1j*k_x - k_y) + gammax*np.eye(N,N) - 1j*gammay*np.eye(N,N)

    H = np.block([[H00, H01], [H10, H11]])
    return H

def HBDG(
    coor, ax, ay, Wsc, WJ,
    potential = 0,
    mu = 0,
    gammax = 0, gammay = 0, gammaz = 0,
    delta = 0, phi = 0,
    alpha = 0,
    qx = 0, qy = 0,
    periodicx = 'no', periodicy = 'no'
    ):

    Delta = delta(coor, Wsc, W)
    H00 =  H0(coor, ax, ay, potential = potential, mu = mu, gammax = gammax, gammay = gammay, gammaz = gammaz,
        alpha = alpha, qx = qx, qy = qy, periodic = periodic)
    H01 = Delta
    H10 = -np.conjugate(Delta)
    H11 = -np.conjugate( H0(coor, ax, ay, potential = potential, mu = mu, gammax = gammax, gammay = gammay,
        gammaz = gammaz, alpha = alpha, qx = -qx, qy = -qy, periodicx = periodicx, periodicy = periodicy) )

    HBDG = np.block([[H00, H01] , [H10, H11]])
    return HBDG



###################### Plotting states and energies ################################

def state_cplot(coor, states, title = 'Probability Density'):
    if coor.shape[0] < states.shape[0]:
        N = int(states.shape[0]/2)
        prob_dens = []
        for i in np.arange(0, int(states.shape[0]/2)):
            prob_dens.append(np.square(abs(states[i])) + np.square(abs(states[i+N])))
    else:
        prob_dens = np.square(abs(states))
    print(sum(prob_dens))
    plt.scatter(coor[:,0], coor[:,1], c = prob_dens)
    plt.title(title)
    plt.colorbar()
    plt.show()


def bands(eigarr, q, Lx, Ly, title = 'Band Structure'):
    for j in range(eigarr.shape[1]):
        plt.plot(q, eigarr[:, j], c ='b', linestyle = 'solid')
    plt.plot(np.linspace(min(q), max(q), 1000), 0*np.linspace(min(q), max(q), 1000), c='k', linestyle='solid', lw=1)
    plt.xticks(np.arange(-np.pi/Lx, np.pi/Lx+0.1*(np.pi/Lx), (np.pi/Lx)), ('-π/Lx', '0', 'π/Lx'))
    plt.xlabel('k [1/A]')
    plt.ylabel('Energy [eV]')
    plt.title(title)
    plt.show()


######################## Potential shapes ##############################

def V_barrier(V0, xi, xf, coor): #(Amplitude, starting point of barrier, ending pt of barrier, coordinate array)
    N = coor.shape[0]
    V = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i == j and coor[i,0] < xf and coor[i,0] > xi: #Making V operator, onsite energy contribution, if site is between xi and xf
                V[i,j] = size
    return V

def V_periodic(V0, coor):
    N = coor.shape[0]
    V = np.zeros((N,N))
    Lx = (max(coor[:, 0]) - min(coor[:, 0]))  #Unit cell size in x-direction, no +1 because has to match coor array
    Ly = (max(coor[:, 1]) - min(coor[:, 1])) #Unit cell size in y-direction
    for i in range(N):
        for j in range(N):
            if i==j:
                V[i,j] = V0*np.sin(np.pi*(coor[i,0])/Lx)*np.sin(np.pi*coor[i,1]/Ly)
    return V
