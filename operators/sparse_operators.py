import scipy.sparse as sparse
import scipy.sparse.linalg as spLA
from numpy import linalg as npLA
import numpy as np

import majoranaJJ.operators.potentials as potentials
import majoranaJJ.modules.constants as const
import majoranaJJ.modules.checkers as check

"""Descritized k-x operator"""
def kx(coor, ax, ay, NN, NNb = None, qx = None):
    row = []; col = []; data = []
    N = int(coor.shape[0])
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
        if NNb is not None and qx is not None and NNb[i, 0] != -1:
            row.append(NNb[i,0]); col.append(i)
            data.append(-tx*np.exp(-1j*qx*Lx))
        if NNb is not None and qx is not None and NNb[i, 2] != -1:
            row.append( NNb[i,2] ); col.append(i)
            data.append(tx*np.exp(1j*qx*Lx))
    ksq = sparse.csc_matrix((data, (row,col)), shape = (N,N), dtype = 'complex')
    return ksq

"""Descritized k-x squared operator"""
def kx2(coor, ax, ay, NN, NNb = None, qx = None):
    row = []; col = []; data = []
    N = int(coor.shape[0])
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
        if NNb is not None and qx is not None and NNb[i, 0] != -1:
            row.append( NNb[i,0] ); col.append(i)
            data.append( -tx*np.exp(-1j*qx*Lx) )
        if NNb is not None and qx is not None and NNb[i, 2] != -1:
            row.append( NNb[i,2] ); col.append(i)
            data.append( -tx*np.exp(1j*qx*Lx) )
    ksq = sparse.csc_matrix((data, (row,col)), shape = (N,N), dtype = 'complex')
    return ksq

############ Descritizing ky operators ##############

"""Descritized k-y operator"""
def ky(coor, ax, ay, NN, NNb = None, qy = None):
    row = []; col = []; data = []
    N = int(coor.shape[0])
    ymax = max(coor[:, 1])
    ymin = min(coor[:, 1])
    Ly = (ymax - ymin + 1)*ay
    ty = 1j/(2*ay)
    for i in range(N):
        if NN[i, 1] != -1:
            row.append( NN[i,1] ); col.append(i)
            data.append( ty )
        if NN[i, 3] != -1:
            row.append( NN[i, 3] ); col.append(i)
            data.append( -ty )
        if NNb is not None and qy is not None and NNb[i, 1] != -1:
            row.append( NNb[i, 1] ); col.append(i)
            data.append( ty*np.exp(1j*qy*Ly) )
        if NNb is not None and qy is not None and NNb[i, 3] != -1:
            row.append(NNb[i, 3]); col.append(i)
            data.append( -ty*np.exp(-1j*qy*Ly) )
    ksq = sparse.csc_matrix((data, (row,col)), shape = (N,N), dtype = 'complex')
    return ksq

"""Descritized k-y squared operator"""
def ky2(coor, ax, ay, NN, NNb = None, qy = None):
    row = []; col = []; data = []
    N = int(coor.shape[0])
    ymax = max(coor[:, 1])
    ymin = min(coor[:, 1])
    Ly = (ymax - ymin + 1)*ay
    ty = 1/(ay**2)
    for i in range(N):
        row.append(i); col.append(i); data.append(2*ty)
        if NN[i, 1] != -1:
            row.append( NN[i, 1] ); col.append(i)
            data.append(-ty)
        if NN[i, 3] != -1:
            row.append( NN[i, 3] ); col.append(i)
            data.append(-ty)
        if NNb is not None and qy is not None and NNb[i, 1] != -1:
            row.append( NNb[i, 1] ); col.append(i)
            data.append( -ty*np.exp(1j*qy*Ly) )
        if NNb is not None and qy is not None and NNb[i, 3] != -1:
            row.append( NNb[i, 3] ); col.append(i)
            data.append( -ty*np.exp(-1j*qy*Ly) )
    ksq = sparse.csc_matrix((data, (row,col)), shape= (N,N), dtype = 'complex')
    return ksq

########################################################

""" Delta Matrix: Particle hole coupling
Parameters:
coor =  coordinate array
Wj =    Width of normal region in Josephson junction
delta = size of superconducting gap in meV
phi =   superconducting phase difference across the normal region
cutx =   width of nodule
cuty =   height of nodule
"""
def Delta(
    coor, Wj = 0,
    delta = 0, phi = 0,
    cutx = 0, cuty = 0
    ):
    N = coor.shape[0]
    Nx = int((max(coor[: , 0]) - min(coor[:, 0])) + 1) #number of lattice sites in x-direction, parallel to junction
    Ny = int((max(coor[: , 1]) - min(coor[:, 1])) + 1) #number of lattice sites in y-direction, perpendicular to junction
    Sx = int((Nx - cutx)/2) #length of either side of nodule, leftover length after subtracted nodule length divided by two
    Wsc = int((Ny - Wj)/2) #width of single superconductor

    row = []; col = []; data = []
    if Wj == 0: #If no junction, every site is superconducting, no phase difference
        for i in range(N):
            row.append(i); col.append(i)
            data.append(delta)
        D = sparse.csc_matrix((data, (row, col)), shape = (N,N), dtype='complex')
        delta = sparse.bmat([[None, D], [-D, None]], format='csc', dtype='complex')
        return delta

    for i in range(N):
        row.append(i); col.append(i)
        bool_inSC, which = check.is_in_SC(i, coor, Wsc, Wj, Sx, cutx, cuty)
        if bool_inSC:
            if which == 'T':
                data.append(delta*np.exp(1j*phi/2))
            elif which == 'B':
                data.append(delta*np.exp(-1j*phi/2))
        else:
            data.append(0)

    D = sparse.csc_matrix((data, (row, col)), shape = (N,N), dtype='complex')
    delta = sparse.bmat([[None, D], [-D, None]], format='csc', dtype='complex')
    return delta
########################################################

"""
This is the Hamiltonian with Spin Orbit Coupling and nearest neighbor hopping and no Superconductivity.

Parameters:
V = potential type, must be a Matrix NxN
mu = chemical potential
alpha = Rashba SOC
gamx = Zeeman energy parallel to junction/superconductor interface
gamy = Zeeman energy perpindicular to Junction
gamz = Zeeman energy normal to device

Basis:
Two states per lattice site for spin up and down. Rows/Columns 1 ->
N correspond to spin up, rows/columns n -> 2N correspond to spin down
"""
def xi(meff):
    return ((const.hbar**2)*(const.e0)*(10**20)*(10**3))/(meff)
def H0(
    coor, ax, ay, NN, NNb = None,
    Wj = 0, cutx = 0, cuty = 0,
    V = 0, mu = 0, meff_normal = 0.026*const.m0, meff_sc = 0.026*const.m0,
    alpha = 0, gamx = 0, gamy = 0, gamz = 0, g_normal = 26, g_sc = 26,
    qx = None, qy = None,
    Tesla = False,
    diff_g_factors = True, Rfactor = 0, diff_alphas = False, diff_meff = False
    ):
    # Hamiltonian with SOC and no superconductivity
    N = coor.shape[0] #number of lattice sites
    I = sparse.identity(N) #identity matrix of size NxN
    Nx = int((max(coor[: , 0]) - min(coor[:, 0])) + 1) #number of lattice sites in x-direction, parallel to junction
    Ny = int((max(coor[: , 1]) - min(coor[:, 1])) + 1) #number of lattice sites in y-direction, perpendicular to junction
    Sx = int((Nx - cutx)/2) #length of either side of nodule, leftover length after subtracted nodule length divided by two
    Wsc = int((Ny - Wj)/2) #width of single superconductor

    k_x = kx(coor, ax, ay, NN, NNb = NNb, qx = qx)
    k_y = ky(coor, ax, ay, NN, NNb = NNb, qy = qy)
    k_x2 = kx2(coor, ax, ay, NN, NNb = NNb, qx = qx)
    k_y2 = ky2(coor, ax, ay, NN, NNb = NNb, qy = qy)

    if diff_g_factors and Tesla:
        row = []; col = []; data = []
        for i in range(N):
            row.append(i); col.append(i)
            inSC, which = check.is_in_SC(i, coor, Wsc, Wj, Sx, cutx, cuty)
            if inSC:
                data.append(g_SC)
            if not inSC:
                data.append(g_normal)
        g_factor = sparse.csc_matrix((data, (row, col)), shape = (N,N))
    elif diff_g_factors and not Tesla:
        row = []; col = []; data = []
        for i in range(N):
            row.append(i); col.append(i)
            inSC, which = check.is_in_SC(i, coor, Wsc, Wj, Sx, cutx, cuty)
            if inSC:
                data.append(Rfactor)
            if not inSC:
                data.append(1)
        R = sparse.csc_matrix((data, (row, col)), shape = (N,N))
    elif not diff_g_factors:
        g_factor = I*g_normal
        R = I

    if diff_meff:
        row = []; col = []; data = []
        for i in range(N):
            row.append(i); col.append(i)
            inSC_i = check.is_in_SC(i, coor, Wsc, Wj, Sx, cutx, cuty)[0]
            if inSC_i :
                data.append(xi(meff_sc)/2)
            else:
                data.append(xi(meff_normal)/2)
        meff = sparse.csc_matrix((data, (row, col)), shape = (N,N))
    elif not diff_meff:
        meff = xi(meff_normal)/2

    if Tesla:
        H00 = (k_x2 + k_y2).multiply(meff) + V + (1/2)*(const.muB*gamz)*g_factor - mu*I
        H11 = (k_x2 + k_y2).multiply(meff) + V - (1/2)*(const.muB*gamz)*g_factor - mu*I
        H10 = alpha*(1j*k_x - k_y)*I + (1/2)*(const.muB*gamx)*g_factor + 1j*(1/2)*(const.muB*gamy)*g_factor
        H01 = alpha*(-1j*k_x - k_y)*I + (1/2)*(const.muB*gamx)*g_factor - 1j*(1/2)*(const.muB*gamy)*g_factor
    elif not Tesla:
        #Then gamx is in units of meV and R is "Reduction factor"
        #R is a matrix multiplying SC sites by a fraction in proportion to
        #the ratio of g-factors
        H00 = (k_x2 + k_y2).multiply(meff) - mu*I + V + gamz*R
        H01 = alpha*(-1j*k_x - k_y)*I + gamx*R - 1j*gamy*R
        H10 = alpha*(1j*k_x - k_y)*I + gamx*R + 1j*gamy*R
        H11 = (k_x2 + k_y2).multiply(meff) - mu*I + V - gamz*R

    H = sparse.bmat([[H00, H01], [H10, H11]], format='csc', dtype = 'complex')
    return H

"""BDG Hamiltonian for superconductivity and SOC"""
def HBDG(
    coor, ax, ay, NN, NNb = None, #lattice parameters
    Wj = 0, cutx = 0, cuty = 0, #junction parameters
    V = 0, mu = 0, meff_normal = 0.026*const.m0, meff_sc = 0.026*const.m0,
    gamx = 0, gamy = 0, gamz = 0, g_normal = 26, g_sc = 26, #zeeman contributions
    alpha = 0, delta = 0, phi = 0, #SOC, SC, SC-phase difference
    qx = None, qy = None, #periodicity factors
    Tesla = False, diff_g_factors = True, Rfactor = 0, diff_alphas = False, diff_meff = False #booleans
    ):
    N = coor.shape[0] #number of lattice sites
    D = Delta(coor, Wj=Wj, delta=delta, phi=phi, cutx=cutx, cuty=cuty)

    QX11 = None
    QY11 = None
    if qx is not None:
        QX11 = -qx
    if qy is not None:
        QY11 = -qy

    H00 = H0(coor, ax, ay, NN, NNb=NNb, Wj=Wj, V=V, mu=mu, gamx=gamx, gamy=gamy, gamz=gamz, alpha=alpha, qx=qx, qy=qy, Tesla=Tesla, diff_g_factors=diff_g_factors, diff_alphas=diff_alphas, diff_meff=diff_meff)

    H01 = D

    H11 = -1*H0(coor, ax, ay, NN, NNb=NNb, Wj=Wj, V=V, mu=mu, gamx=gamx, gamy=gamy, gamz=gamz, alpha=alpha, qx=QX11, qy=QY11, Tesla=Tesla, diff_g_factors=diff_g_factors, diff_alphas=diff_alphas, diff_meff=diff_meff).conjugate()

    H10 = D.conjugate().transpose()

    H = sparse.bmat([[H00, H01], [H10, H11]], format='csc', dtype = 'complex')
    return H

#######################################################

#Energy eigenvalues for BDG Hamilonian
def EBDG(
    coor, ax, ay, NN, NNb = None, Wj = 0,
    cutx = 0, cuty = 0,
    V = 0, mu = 0,
    gamx = 0, gamy = 0, gamz = 0,
    alpha = 0, delta = 0, phi = 0,
    qx = 0, qy = 0,
    periodicX = False, periodicY = False,
    k = 8, sigma = 0, which = 'LM', tol = 0, maxiter = None
    ):
    H = HBDG(coor, ax, ay, NN, Wj=Wj, NNb=NNb, cutx=cutx, cuty=cuty, V=V, mu=mu, gamx=gamx, gamy=gamy, gamz=gamz, alpha=alpha, delta=delta, phi=phi, qx=qx, qy=qy, periodicX=periodicX, periodicY=periodicY)

    eigs, vecs = spLA.eigsh(H, k=k, sigma=sigma, which=which, tol=tol, maxiter=maxiter)
    idx_sort = np.argsort(eigs)
    eigs = eigs[idx_sort]
    return np.sort(eigs)

#Energy Eignencalues for SOC Hamiltonain, or H0
def ESOC(
    coor, ax, ay, NN, NNb = None,
    V = 0, mu = 0,
    gamx = 0, gamy = 0, gamz = 0, alpha = 0,
    qx = 0, qy = 0,
    periodicX = False, periodicY = False,
    k = 4, sigma = 0, which = 'LM', tol = 0, maxiter = None
    ):
    H = H0(coor, ax, ay, NN, NNb=NNb, V=V, mu=mu, gamx=gamx, gamy=gamy, gamz=gamz, alpha=alpha, qx=qx, qy=qy, periodicX=periodicX, periodicY=periodicY)

    eigs, vecs = spLA.eigsh(H, k=k, sigma=sigma, which=which, tol=tol, maxiter=maxiter)
    idx_sort = np.argsort(eigs)
    eigs = eigs[idx_sort]
    return np.sort(eigs)

#######################################################

def print_matrix(M):
    for r in M:
        for c in r:
            print(c, end = " ")
        print()
