import numpy as np

import modules.constants as const

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
