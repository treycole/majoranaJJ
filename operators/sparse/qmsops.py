import scipy.sparse as sparse
import scipy.sparse.linalg as spLA
from numpy import linalg as npLA
import numpy as np

import majoranaJJ.modules.constants as const

"""Descritized k-x operator"""
def kx(coor, ax, ay, NN, NNb = None, qx = 0):
    row = []; col = []; data = []
    N = coor.shape[0]

    xmax = max(coor[:, 0])
    xmin = min(coor[:, 0])
    Lx = (xmax - xmin + 1)*ax
    tx = 1j/(2*ax)

    for i in range(N):
        if NN[i,0] != -1:
            row.append( NN[i,0] ); col.append(i)
            data.append(-tx)

        if NN[i,2] != -1:
            row.append( NN[i,2] ); col.append(i)
            data.append(tx)

        if NNb is not None and NNb[i, 0] != -1:
            row.append(NNb[i,0]); col.append(i)
            data.append( -tx*np.exp(-1j*qx*Lx) )

        if NNb is not None and NNb[i, 2] != -1:
            row.append( NNb[i,2] ); col.append(i)
            data.append( tx*np.exp(1j*qx*Lx) )

    ksq = sparse.csc_matrix((data, (row,col)), shape = (N,N), dtype = 'complex')
    return ksq

"""Descritized k-x squared operator"""
def kx2(coor, ax, ay, NN, NNb = None, qx = 0):
    row = []; col = []; data = []
    N = coor.shape[0]

    xmax = max(coor[:, 0])
    xmin = min(coor[:, 0])
    Lx = (xmax - xmin + 1)*ax
    tx = 1/(ax**2)

    for i in range(N):
        row.append(i); col.append(i); data.append(2*tx)
        if NN[i,0] != -1:
            row.append( NN[i,0] ); col.append(i)
            data.append( -tx )

        if NN[i,2] != -1:
            row.append( NN[i,2] ); col.append(i)
            data.append( -tx )

        if NNb is not None and NNb[i, 0] != -1:
            row.append( NNb[i,0] ); col.append(i)
            data.append( -tx*np.exp(-1j*qx*Lx) )

        if NNb is not None and NNb[i, 2] != -1:
            row.append( NNb[i,2] ); col.append(i)
            data.append( -tx*np.exp(1j*qx*Lx) )

    ksq = sparse.csc_matrix((data, (row,col)), shape = (N,N), dtype = 'complex')
    return ksq

############ Descritizing ky operators ##############

"""Descritized k-y operator"""
def ky(coor, ax, ay, NN, NNb = None, qy = 0):
    row = []; col = []; data = []
    N = coor.shape[0]

    ymax = max(coor[:, 1])
    ymin = min(coor[:, 1])
    Ly = (ymax - ymin + 1)*ay
    ty = 1j/(2*ay)

    for i in range(N):
        if NN[i,1] != -1:
            row.append( NN[i,1] ); col.append(i)
            data.append( ty )

        if NN[i,3] != -1:
            row.append( NN[i,3] ); col.append(i)
            data.append( -ty )

        if NNb is not None and NNb[i, 1] != -1:
            row.append( NNb[i,1] ); col.append(i)
            data.append( ty*np.exp(1j*qy*Ly) )

        if NNb is not None and NNb[i, 3] != -1:
            row.append(NNb[i,3]); col.append(i)
            data.append( -ty*np.exp(-1j*qy*Ly) )

    ksq = sparse.csc_matrix((data, (row,col)), shape = (N,N), dtype = 'complex')
    return ksq

"""Descritized k-y squared operator"""
def ky2(coor, ax, ay, NN, NNb = None, qy = 0):
    row = []; col = []; data = []
    N = coor.shape[0]

    ymax = max(coor[:, 1])
    ymin = min(coor[:, 1])
    Ly = (ymax - ymin + 1)*ay
    ty = 1/(ay**2)

    for i in range(N):
        row.append(i); col.append(i); data.append(2*ty)
        if NN[i,1] != -1:
            row.append( NN[i,1] ); col.append(i)
            data.append(-ty)

        if NN[i,3] != -1:
            row.append( NN[i,3] ); col.append(i)
            data.append(-ty)

        if NNb is not None and NNb[i,1] != -1:
            row.append( NNb[i,1] ); col.append(i)
            data.append( -ty*np.exp(1j*qy*Ly) )

        if NNb is not None and NNb[i,3] != -1:
            row.append( NNb[i,3] ); col.append(i)
            data.append( -ty*np.exp(-1j*qy*Ly) )

    ksq = sparse.csc_matrix((data, (row,col)), shape= (N,N), dtype = 'complex')
    return ksq

########################################################

"""Delta Matrix: Particle hole coupling"""
def Delta(
    coor, Wj = 0,
    delta = 0, phi = 0,
    Sx = None, cutx = None, cuty = None
    ):

    N = coor.shape[0]
    Ny = (max(coor[: , 1]) - min(coor[:, 1])) + 1 #number of lattice sites in y-direction, perp to junction

    row = []; col = []; data = []

    if Wj == 0: #If no junction, every site is superconducting, no phase diff
        for i in range(N):
                row.append(i); col.append(i)
                data.append(delta)
        D = sparse.csc_matrix((data, (row, col)), shape = (N,N), dtype='complex')
        delta = sparse.bmat([[None, D], [-D, None]], format='csc', dtype='complex')
        return delta

    while Wj > Ny:
        Wj -= 1

    if (Ny-Wj)%2 != 0: #Cant have even Ny and odd Wj, odd number of lattice sites for proximty superconductors
        if (Wj + 1) <= Ny:
            Wj += 1
        else:
            Wj -=1

    Wsc = int((Ny - Wj)/2)
    for i in range(N):
        y = coor[i, 1]
        x = coor[i, 0]

        if y < Wsc: #if in bottom SC
            row.append(i); col.append(i)
            data.append(delta*np.exp(-1j*phi/2) )

        if y >= Wsc and y < (Wsc+Wj):
            if Sx is not None: #if there is a cut present
                if (2*Sx + cutx) != (max(coor[:, 0]) + 1):
                    print("Dimensions of nodule do not add up to lattice size along x direction")
                    return
                if (2*cuty) >= Wj:
                    print("Nodule extends across junction")
                    return
                if (x >= Sx and x < (Sx + cutx)) and (y < (Wsc + cuty) or y >= ((Wsc + Wj) - cuty)): #if in range of cut
                    row.append(i); col.append(i)
                    data.append(delta*np.exp(-1j*phi/2) )
                else: #lattice site is in junction
                    row.append(i); col.append(i)
                    data.append(0)
            else: #lattice site is in junction
                row.append(i); col.append(i)
                data.append(0)

        if y >= (Wsc+Wj):
            row.append(i); col.append(i)
            data.append( delta*np.exp( 1j*phi/2 ) )

    D = sparse.csc_matrix((data, (row, col)), shape = (N,N), dtype='complex')
    delta = sparse.bmat([[None, D], [-D, None]], format='csc', dtype='complex')

    return delta

########################################################

"""
This is the Hamiltonian with Spin Orbit Coupling and nearest neighbor hopping and no Superconductivity.

The parameter PERIODIC determines whether the function calls a construction of k-operators with or without
boundary conditions in x and y directions.

Basis: Two states per lattice site for spin up and down. Rows/Columns 1 ->
N correspond to spin up, rows/columns n -> 2N correspond to spin down
"""

def H0(
    coor, ax, ay, NN, NNb = None,
    V=0, mu=0,
    gammax=0, gammay=0, gammaz=0,
    alpha=0,
    qx=0, qy=0,
    periodicX = False, periodicY = False
    ):  # Hamiltonian with SOC

    N = coor.shape[0]

    if periodicX:
        k_x = kx(coor, ax, ay, NN, NNb = NNb, qx = qx)
        k_x2 = kx2(coor, ax, ay, NN, NNb = NNb, qx = qx)
    if periodicY:
        k_y = ky(coor, ax, ay, NN, NNb = NNb, qy = qy)
        k_y2 = ky2(coor, ax, ay, NN, NNb = NNb, qy = qy)
    if not periodicX:
        k_x = kx2(coor, ax, ay, NN)
        k_x2 = kx2(coor, ax, ay, NN)
    if not periodicY:
        k_y = ky(coor, ax, ay, NN)
        k_y2 = ky2(coor, ax, ay, NN)

    I = sparse.identity(N)

    H00 = (const.xi/2)*(k_x2 + k_y2) + V + gammaz*I - mu*I
    H11 = (const.xi/2)*(k_x2 + k_y2) + V - gammaz*I - mu*I
    H10 = alpha*(1j*k_x - k_y) + gammax*I + 1j*gammay*I
    H01 = alpha*(-1j*k_x - k_y) + gammax*I - 1j*gammay*I

    H = sparse.bmat([[H00, H01], [H10, H11]], format='csc', dtype = 'complex')
    return H

def HBDG(
    coor, ax, ay, NN, NNb = None, Wj = 0,
    Sx = None, cutx = None, cuty = None,
    V = 0, mu = 0,
    gammax = 0, gammay = 0, gammaz = 0,
    alpha = 0, delta = 0, phi = 0,
    qx = 0, qy = 0,
    periodicX = False, periodicY = False
    ): #BDG Hamiltonian for superconductivity and SOC

    N = coor.shape[0]

    D = Delta(coor, Wj = Wj, delta = delta, phi = phi, Sx = Sx, cutx = cutx, cuty = cuty)

    H00 = H0( coor, ax, ay, NN, NNb = NNb, V = V, mu = mu, gammax = gammax, gammay = gammay, gammaz = gammaz, alpha = alpha, qx = qx, qy = qy, periodicX = periodicX, periodicY = periodicY)

    H11 = -1*H0(coor, ax, ay, NN, NNb = NNb, V = V, mu = mu, gammax = gammax, gammay = gammay, gammaz = gammaz, alpha = alpha, qx = -qx, qy = -qy, periodicX = periodicX, periodicY = periodicY).conjugate()

    H10 = D

    H01 = D.conjugate().transpose()

    H = sparse.bmat([[H00, H01], [H10, H11]], format='csc', dtype = 'complex')
    return H

#######################################################

#Energy eigenvalues for BDG Hamilonian
def EBDG(
    coor, ax, ay, NN, NNb = None, Wj = 0,
    Sx = None, cutx = None, cuty = None,
    V = 0, mu = 0,
    gammax = 0, gammay = 0, gammaz = 0,
    alpha = 0, delta = 0, phi = 0,
    qx = 0, qy = 0,
    periodicX = False, periodicY = False,
    k = 8, sigma = 0, which = 'LM', tol = 0, maxiter = None
    ):

    H = HBDG(coor, ax, ay, NN, Wj=Wj, NNb=NNb, Sx=Sx, cutx=cutx, cuty=cuty, V=V, mu=mu, gammax=gammax, gammay=gammay, gammaz=gammaz, alpha=alpha, delta=delta, phi=phi, qx=qx, qy=qy, periodicX=periodicX, periodicY=periodicY)

    eigs, vecs = spLA.eigsh(H, k=k, sigma=sigma, which=which, tol=tol, maxiter=maxiter)
    idx_sort = np.argsort(eigs)
    eigs = eigs[idx_sort]

    return np.sort(eigs)

#Energy Eignencalues for SOC Hamiltonain, or H0
def ESOC(
    coor, ax, ay, NN, NNb = None,
    V = 0, mu = 0,
    gammax = 0, gammay = 0, gammaz = 0, alpha = 0,
    qx = 0, qy = 0,
    periodicX = False, periodicY = False,
    k = 4, sigma = 0, which = 'LM', tol = 0, maxiter = None
    ):

    H = H0(coor, ax, ay, NN, NNb=NNb,V=V, mu=mu, gammax=gammax, gammay=gammay, gammaz=gammaz,alpha=alpha,qx=qx, qy=qy, periodicX=periodicX, periodicY=periodicY)

    eigs, vecs = spLA.eigsh(H, k=k, sigma=sigma, which=which, tol=tol, maxiter=maxiter)
    idx_sort = np.argsort(eigs)
    eigs = eigs[idx_sort]

    return np.sort(eigs)
