import numpy as np

import majoranaJJ.modules.constants as const
import majoranaJJ.modules.operators as op

def H0(coor, ax, ay):
    N = coor.shape[0]
    H = np.zeros((N,N), dtype = 'complex')
    H = (const.xi/2)*(op.kx2(coor, ax, ay) + op.ky2(coor, ax, ay))
    return H

#Spin orbit coupling, spin energy splitting, size 2Nx2N: 0->N spin up states, N -> 2N spin down states
def H_SOC(coor, ax, ay, V, gamma, alpha):
    H_0 = H0(coor, ax, ay)
    N = H_0.shape[0]
    k_x = op.kx(coor, ax, ay)
    k_y = op.ky(coor, ax, ay)
    H = np.zeros((2*N, 2*N), dtype = 'complex')

    H00 = H_0 + gamma*np.eye(N,N) + V
    H11 = H_0 - gamma*np.eye(N,N) + V
    H10 = alpha*(1j*k_x - k_y)
    H01 = alpha*(-1j*k_x - k_y)
    H = np.block([[H00, H01],[H10, H11]])
    return H

###### Hamiltonians with periodic boundary conditions ######

def H0k(coor, ax, ay, qx = 0, qy = 0):
    N = coor.shape[0]
    H = np.zeros((N,N), dtype = 'complex')
    H = (const.xi/2)*(op.kpx2(coor, ax, ay, qx) + op.kpy2(coor, ax, ay, qy))
    return H

def H_SOk(coor, ax, ay, qx = 0, qy = 0, V = 0, gamma = 0, alpha = 0):
    H_0 = H0k(coor, ax, ay, qx , qy)
    N = H_0.shape[0]
    k_x = kpx(coor, ax, ay, qx)
    k_y = kpy(coor, ax, ay, qy)
    H = np.zeros((2*N, 2*N), dtype = 'complex')

    H00 = H_0 + gamma*np.eye(N,N) + V
    H11 = H_0 - gamma*np.eye(N,N) + V
    H10 = alpha*(1j*k_x - k_y)
    H01 = alpha*(-1j*k_x - k_y)
    H = np.block([[H00, H01], [H10, H11]])
    return H
