import scipy.sparse as sparse
import scipy.sparse.linalg as spLA
from numpy import linalg as npLA
import numpy as np

import majoranaJJ.modules.constants as const

"""Descritized k-x operator"""
def kx(coor, ax, ay, NN, NNb = None):
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
            data.append(-tx)

        if NNb is not None and NNb[i, 2] != -1:
            row.append( NNb[i,2] ); col.append(i)
            data.append(tx)

    ksq = sparse.csc_matrix((data, (row,col)), shape = (N,N), dtype = 'complex')
    return ksq

"""Descritized k-x squared operator"""
def kx2(coor, ax, ay, NN, NNb = None):
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
            data.append(-tx)

        if NN[i,2] != -1:
            row.append( NN[i,2] ); col.append(i)
            data.append(-tx)

        if NNb is not None and NNb[i, 0] != -1:
            row.append( NNb[i,0] ); col.append(i)
            data.append(-tx)

        if NNb is not None and NNb[i, 2] != -1:
            row.append( NNb[i,2] ); col.append(i)
            data.append(-tx)

    ksq = sparse.csc_matrix((data, (row,col)), shape = (N,N), dtype = 'complex')
    return ksq

############ Descritizing ky operators ##############

"""Descritized k-y operator"""
def ky(coor, ax, ay, NN, NNb = None):
    row = []; col = []; data = []
    N = coor.shape[0]

    ymax = max(coor[:, 1])
    ymin = min(coor[:, 1])
    Ly = (ymax - ymin + 1)*ay
    ty = 1j/(2*ay)

    for i in range(N):
        if NN[i,1] != -1:
            row.append( NN[i,1] ); col.append(i)
            data.append(ty)

        if NN[i,3] != -1:
            row.append( NN[i,3] ); col.append(i)
            data.append(-ty)

        if NNb is not None and NNb[i, 1] != -1:
            row.append( NNb[i,1] ); col.append(i)
            data.append(ty)

        if NNb is not None and NNb[i, 3] != -1:
            row.append(NNb[i,3]); col.append(i)
            data.append(-ty)

    ksq = sparse.csc_matrix((data, (row,col)), shape = (N,N), dtype = 'complex')
    return ksq

"""Descritized k-y squared operator"""
def ky2(coor, ax, ay, NN, NNb = None):
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
            data.append(-ty)

        if NNb is not None and NNb[i,3] != -1:
            row.append( NNb[i,3] ); col.append(i)
            data.append(-ty )

    ksq = sparse.csc_matrix((data, (row,col)), shape= (N,N), dtype = 'complex')
    return ksq

########################################################
""" Delta Matrix: Particle hole coupling
Parameters:

coor = coordinate array, for a JJ the unit cell is square and numbered from bottom left to top right of unit cell.

Wj = Width of normal region in Josephson junction

delta = size of superconducting gap in eV

phi = superconducting phase difference across the normal region

Sx = "Side x" ~ the length of the interface region on either side of nodule.
So Sx = Nx/2 would mean no nodule, since (Nx - 2*Sx) = nodule Width

cutx = width of nodule

cuty = height of nodule

nodule = boolean value which determines whether there is a nodule or not. If True, then extra conditions must be met in order to determine whether a given lattice site is has a superconducting contribution or not
"""
def Delta(
    coor, Wj = 0,
    delta = 0, phi = 0,
    cutx = 0, cuty = 0
    ):

    N = coor.shape[0]
    Ny = (max(coor[: , 1]) - min(coor[:, 1])) + 1 #number of lattice sites in y-direction, perpendicular to junction
    Nx = (max(coor[: , 0]) - min(coor[:, 0])) + 1 #number of lattice sites in x-direction, parallel to junction
    row = []; col = []; data = []

    if Wj == 0: #If no junction, every site is superconducting, no phase diff
        for i in range(N):
                row.append(i); col.append(i)
                data.append(delta)
        D = sparse.csc_matrix((data, (row, col)), shape = (N,N), dtype='complex')
        delta = sparse.bmat([[None, D], [-D, None]], format='csc', dtype='complex')
        return delta

    if (Ny-Wj)%2 != 0 and Wj != 0: #Cant have even Ny and odd Wj, the top and bottom superconductors would then be of a different size
        if Wj - 1 > 0:
            Wj -= 1
        else:
            Wj +=1

    if (Nx-cutx)%2 != 0 and cutx != 0: #Sx must be equal lengths on both sides
        if cutx - 1 > 0:
            cutx -= 1
        else:
            cutx += 1

    while (2*cuty) >= Wj: #height of nodule cant be bigger than junction width
        cuty -= 1

    while Wj >= Ny: #if juntion width is larger than the total size of unit cell then we must decrease it until it is smaller
        Wj -= 1

    Sx = int((Nx - cutx)/2) #length of either side of nodule, leftover length after subtracted nodule length divided by two
    Wsc = int((Ny - Wj)/2) #width of single superconductor

    for i in range(N):
        y = coor[i, 1]
        x = coor[i, 0]

        if y < Wsc: #if in bottom SC
            row.append(i); col.append(i)
            data.append(delta*np.exp(-1j*phi/2) )

        if y >= (Wsc+Wj): #if in top SC
            row.append(i); col.append(i)
            data.append( delta*np.exp( 1j*phi/2 ) )

        if y >= Wsc and y < (Wsc+Wj): #if coordinates in junction region
            if cuty != 0 and cutx !=0: #if there is a nodule present
                if (x >= Sx and x < (Sx + cutx)): #in x range of cut
                    if y >= ((Wsc + Wj) - cuty): #if in y range of cut along bottom interface
                        row.append(i); col.append(i)
                        data.append(delta*np.exp(-1j*phi/2) )
                    if  y < (Wsc + cuty) :#if in y range of cut along top interface
                        row.append(i); col.append(i)
                        data.append(delta*np.exp(1j*phi/2) )
                    else: #site is in junction, out of y range
                        row.append(i); col.append(i)
                        data.append(0)
                else: #lattice site is in junction, out of x range
                    row.append(i); col.append(i)
                    data.append(0)
            else: #lattice site is in junction, no nodule
                row.append(i); col.append(i)
                data.append(0)

    D = sparse.csc_matrix((data, (row, col)), shape = (N,N), dtype='complex')
    delta = sparse.bmat([[None, D], [-D, None]], format='csc', dtype='complex')

    return delta

########################################################

def Hq(
    coor, ax, ay, NN, NNb = None,
    Wj = 0, cutx = 0, cuty = 0, #junction parameters
    V = 0, mu = 0, alpha = 0, delta = 0, phi = 0,
    periodicX = False, periodicY = False
    ):  # Hamiltonian with SOC and no superconductivity

    N = coor.shape[0] #number of lattice sites

    if periodicX: #if x-direction is periodic
        k_x = kx(coor, ax, ay, NN, NNb = NNb)
        k_x2 = kx2(coor, ax, ay, NN, NNb = NNb)
    if periodicY: #if y-direction is periodic
        k_y = ky(coor, ax, ay, NN, NNb = NNb)
        k_y2 = ky2(coor, ax, ay, NN, NNb = NNb)
    if not periodicX: #else
        k_x = kx(coor, ax, ay, NN)
        k_x2 = kx2(coor, ax, ay, NN)
    if not periodicY: #else
        k_y = ky(coor, ax, ay, NN)
        k_y2 = ky2(coor, ax, ay, NN)

    I = sparse.identity(N) #identity matrix of size NxN
    H0_00 = (const.xi/2)*(k_x2 + k_y2) + V - mu*I
    H0_11 = (const.xi/2)*(k_x2 + k_y2) + V - mu*I
    H0_10 = alpha*(1j*k_x - k_y)
    H0_01 = alpha*(-1j*k_x - k_y)

    Hq_00 = (const.xi/2)*(2*k_x)
    Hq_11 = (const.xi/2)*(2*k_x)
    Hq_10 = alpha*(1j*I)
    Hq_01 = alpha*(-1j*I)

    Hqq_00 = (const.xi/2)*I
    Hqq_11 = (const.xi/2)*I
    Hqq_10 = 0*I
    Hqq_01 = 0*I

    Hgam_00 = 0*I
    Hgam_11 = 0*I
    Hgam_10 = I
    Hgam_01 = I

    MU_00 = -1*I
    MU_11 = -1*I
    MU_10 = 0*I
    MU_01 = 0*I

    H0 = sparse.bmat([[H0_00, H0_01], [H0_10, H0_11]], format='csc', dtype = 'complex')
    Hq = sparse.bmat([[Hq_00, Hq_01], [Hq_10, Hq_11]], format='csc', dtype = 'complex')
    Hqq = sparse.bmat([[Hqq_00, Hqq_01], [Hqq_10, Hqq_11]], format='csc', dtype = 'complex')
    DELTA = Delta(coor, Wj = Wj, delta = delta, phi = phi, cutx = cutx, cuty = cuty)
    Hgam = sparse.bmat([[Hgam_00, Hgam_01], [Hgam_10, Hgam_11]], format='csc', dtype = 'complex')
    MU = sparse.bmat([[MU_00, MU_01], [MU_10, MU_11]], format='csc', dtype = 'complex')
    return H0, Hq, Hqq, DELTA, Hgam

def H0(H0, Hq, Hqq, Hgam, q, gx = 0):
    H = H0 + q*Hq + q**2*Hqq + gx*Hgam
    return H

def HBDG(H0, Hq, Hqq, DELTA, Hgam, q, gx=0):
    H00 = H0 + q*Hq + q**2*Hqq + gx*Hgam
    H11 = -(H0 - q*Hq + q**2*Hqq + gx*Hgam).conjugate()
    H01 = DELTA
    H10 = DELTA.conjugate().transpose()
    H = sparse.bmat([[H00, H01], [H10, H11]], format='csc', dtype = 'complex')
    return H

def HBDG_LE(H0, Hq, Hqq, DELTA, Hgam, MU, q, d_mu=0, gx=0):
    H00 = H0 + q*Hq + q**2*Hqq + gx*Hgam - d_mu*MU
    H11 = np.conjugate(-(H0 - q*Hq + q**2*Hqq + gx*Hgam - d_mu*MU))
    H01 = DELTA
    H10 = np.conjugate(np.transpose(DELTA))
    H = np.bmat([[H00, H01], [H10, H11]])
    return H

#######################################################

#Energy eigenvalues for BDG Hamilonian
def EBDG(H0, Hq, Hqq, DELTA, Hgam, q, gx = 0, k = 8, sigma = 0, which = 'LM', tol = 0, maxiter = None):

    H = HBDG(H0, Hq, Hqq, DELTA, q, Hgam, gx)

    eigs, vecs = spLA.eigsh(H, k=k, sigma=sigma, which=which, tol=tol, maxiter=maxiter)
    idx_sort = np.argsort(eigs)
    eigs = eigs[idx_sort]

    return np.sort(eigs)
