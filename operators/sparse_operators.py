import scipy.sparse as sparse
import scipy.sparse.linalg as spLA
from numpy import linalg as npLA
import numpy as np

import majoranaJJ.operators.potentials as potentials
import majoranaJJ.modules.constants as const
import majoranaJJ.modules.checkers as check
import majoranaJJ.modules.plots as plots

"""Descritized k-x operator"""
def kx(coor, ax, ay, NN, NNb = None, qx = None):
    row = []; col = []; data = []
    N = int(coor.shape[0])
    xmax = max(coor[:, 0])
    xmin = min(coor[:, 0])
    Lx = (xmax - xmin + 1)*ax
    tx = -1j/(2*ax)
    #print("kxLx sparse", qx*Lx)
    for i in range(N):
        if NN[i,0] != -1:
            row.append( NN[i,0] ); col.append(i)
            data.append(-tx)
        if NN[i,2] != -1:
            row.append( NN[i,2] ); col.append(i)
            data.append(tx)
        if NNb is not None and qx is not None and NNb[i, 0] != -1:
            row.append(NNb[i,0]); col.append(i)
            data.append(-tx*np.exp(1j*qx*Lx))
        if NNb is not None and qx is not None and NNb[i, 2] != -1:
            row.append( NNb[i,2] ); col.append(i)
            data.append(tx*np.exp(-1j*qx*Lx))
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
            data.append( -tx*np.exp(1j*qx*Lx) )
        if NNb is not None and qx is not None and NNb[i, 2] != -1:
            row.append( NNb[i,2] ); col.append(i)
            data.append( -tx*np.exp(-1j*qx*Lx) )
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
    ty = -1j/(2*ay)
    for i in range(N):
        if NN[i, 1] != -1:
            row.append( NN[i,1] ); col.append(i)
            data.append( ty )
        if NN[i, 3] != -1:
            row.append( NN[i, 3] ); col.append(i)
            data.append( -ty )
        if NNb is not None and qy is not None and NNb[i, 1] != -1:
            row.append( NNb[i, 1] ); col.append(i)
            data.append( ty*np.exp(-1j*qy*Ly) )
        if NNb is not None and qy is not None and NNb[i, 3] != -1:
            row.append(NNb[i, 3]); col.append(i)
            data.append( -ty*np.exp(1j*qy*Ly) )
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
            data.append( -ty*np.exp(-1j*qy*Ly) )
        if NNb is not None and qy is not None and NNb[i, 3] != -1:
            row.append( NNb[i, 3] ); col.append(i)
            data.append( -ty*np.exp(1j*qy*Ly) )
    ksq = sparse.csc_matrix((data, (row,col)), shape= (N,N), dtype = 'complex')
    return ksq

########################################################
"""
Delta Matrix: Particle hole coupling
Parameters:
coor = coordinate array
Wj = Width of normal region in Josephson junction
delta = size of superconducting gap in meV
phi = superconducting phase difference across the normal region
cutxT = nodule width along x for top (integer)
cutxB = nodule width along x for bottom (integer)
cutyT = nodule width along y for top (integer)
cutyB = nodule width along y for bottom (integer)
"""
def Delta(
    coor, Wj = 0, delta = 0, phi = 0,
    cutxT = 0, cutyT = 0, cutxB = 0, cutyB = 0
    ):
    N = coor.shape[0]
    Nx = int((max(coor[: , 0]) - min(coor[:, 0])) + 1) #number of lattice sites in x-direction, parallel to junction
    Ny = int((max(coor[: , 1]) - min(coor[:, 1])) + 1) #number of lattice sites in y-direction, perpendicular to junction
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
        bool_inSC, which = check.is_in_SC(i, coor, Wsc, Wj, cutxT=cutxT, cutyT=cutyT, cutxB=cutxB, cutyB=cutyB)
        if bool_inSC:
            if which == 'T': #in top SC
                data.append(delta)
            elif which == 'B': #in bottom SC
                data.append(delta*np.exp(1j*phi))
        else:
            data.append(0)

    D = sparse.csc_matrix((data, (row, col)), shape = (N,N), dtype='complex')
    delta = sparse.bmat([[None, D], [-D, None]], format='csc', dtype='complex')
    return delta
########################################################
"""
Parameters:
coor = coordinate array
ax = descritization constant along x
ay = descritization constant along y
NN = nearest neighbor array
NNb = nearest neighbor array for periodic boundary conditions
Wj = Junction width (integer)
cutxT = nodule width along x for top (integer)
cutxB = nodule width along x for bottom (integer)
cutyT = nodule width along y for top (integer)
cutyB = nodule width along y for bottom (integer)
Vj = junction potential
Vsc = superconducting potential
mu = chemical potential
alpha = Rashba SOC
gamx = Zeeman energy parallel to junction/superconductor interface
gamy = Zeeman energy perpendicular to junction
gamz = Zeeman energy normal to device

Basis:
Two states per lattice site for spin up and down.
Rows/Columns 1 -> N correspond to spin up, rows/columns N -> 2N correspond to spin down
"""
def H0(
    coor, ax, ay, NN, NNb = None,
    Wj = 0, cutxT = 0, cutyT = 0, cutxB = 0, cutyB = 0,
    Vj = 0, Vsc = 0, mu = 0, meff_normal = 0.026, meff_sc = 0.026,
    alpha = 0, gamx = 0, gamy = 0, gamz = 0, g_normal = 26, g_sc = 0,
    qx = None, qy = None,
    Tesla = False,
    diff_g_factors = True, Rfactor = 0, diff_alphas = False, diff_meff = False
    ):
    N = coor.shape[0] #number of lattice sites
    Nx = int((max(coor[: , 0]) - min(coor[:, 0])) + 1) #number of lattice sites in x-direction, parallel to junction
    Ny = int((max(coor[: , 1]) - min(coor[:, 1])) + 1) #number of lattice sites in y-direction, perpendicular to junction
    I = sparse.identity(N) #identity matrix of size NxN
    V = potentials.Vjj(coor=coor, Wj=Wj, Vsc=Vsc, Vj=Vj, cutxT=cutxT, cutyT=cutyT, cutxB=cutxB, cutyB=cutyB) #potential matrix NxN

    Wsc = int((Ny - Wj)/2) #width of single superconductor

    k_x = kx(coor, ax, ay, NN, NNb=NNb, qx=qx)
    k_y = ky(coor, ax, ay, NN, NNb=NNb, qy=qy)
    k_x2 = kx2(coor, ax, ay, NN, NNb=NNb, qx=qx)
    k_y2 = ky2(coor, ax, ay, NN, NNb=NNb, qy=qy)

    if diff_g_factors and Tesla:
        row = []; col = []; data = []
        for i in range(N):
            row.append(i); col.append(i)
            inSC, which = check.is_in_SC(i, coor, Wsc, Wj, cutxT=cutxT, cutyT=cutyT, cutxB=cutxB, cutyB=cutyB)
            if inSC:
                data.append(g_sc)
            if not inSC:
                data.append(g_normal)
        g_factor = sparse.csc_matrix((data, (row, col)), shape = (N,N))
    elif diff_g_factors and not Tesla:
        row = []; col = []; data = []
        for i in range(N):
            row.append(i); col.append(i)
            inSC, which = check.is_in_SC(i, coor, Wsc, Wj, cutxT=cutxT, cutyT=cutyT, cutxB=cutxB, cutyB=cutyB)
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
            inSC_i = check.is_in_SC(i, coor, Wsc, Wj, cutxT=cutxT, cutyT=cutyT, cutxB=cutxB, cutyB=cutyB)[0]
            if inSC_i :
                data.append(const.hbsqr_m0/(2*meff_sc))
            else:
                data.append(const.hbsqr_m0/(2*meff_normal))
        meff = sparse.csc_matrix((data, (row, col)), shape = (N,N))
    elif not diff_meff:
        meff = const.hbsqr_m0/(2*meff_normal)

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
    coor, ax, ay, NN, NNb = None,
    Wj = 0, cutxT = 0, cutyT = 0, cutxB = 0, cutyB = 0,
    Vj = 0, Vsc = 0, mu = 0, gamx = 0, gamy = 0, gamz = 0, alpha = 0,
    delta = 0, phi = 0,
    meff_normal = 0.026, meff_sc = 0.026, g_normal = 26, g_sc = 26,
    qx = None, qy = None,
    Tesla = False, diff_g_factors = True, Rfactor = 0, diff_alphas = False, diff_meff = False
    ):
    N = coor.shape[0] #number of lattice sites
    Nx = int((max(coor[: , 0]) - min(coor[:, 0])) + 1) #number of lattice sites in x-direction, parallel to junction
    Ny = int((max(coor[: , 1]) - min(coor[:, 1])) + 1) #number of lattice sites in y-direction, perpendicular to junction
    D = Delta(coor=coor, Wj=Wj, delta=delta, phi=phi, cutxT=cutxT, cutyT=cutyT, cutxB=cutxB, cutyB=cutyB)

    #V = potentials.Vjj(coor=coor, Wj=Wj, Vsc=Vsc, Vj=Vj, cutxT=cutxT, cutyT=cutyT, cutxB=cutxB, cutyB=cutyB)
    #plots.potential_profile(coor, V)
    #plots.junction(coor, D)

    QX11 = None
    QY11 = None
    if qx is not None:
        QX11 = -qx
    if qy is not None:
        QY11 = -qy

    H00 = H0(coor, ax, ay, NN, NNb=NNb, Wj=Wj, cutxT=cutxT, cutyT=cutyT, cutxB=cutxB, cutyB=cutyB, Vj=Vj, Vsc=Vsc, mu=mu, gamx=gamx, gamy=gamy, gamz=gamz, alpha=alpha, qx=qx, qy=qy, g_normal=g_normal, g_sc=g_sc, Tesla=Tesla, Rfactor=Rfactor, diff_g_factors=diff_g_factors, diff_alphas=diff_alphas, diff_meff=diff_meff, meff_normal=meff_normal, meff_sc=meff_sc)
    H11 = -1*H0(coor, ax, ay, NN, NNb=NNb, Wj=Wj, cutxT=cutxT, cutyT=cutyT, cutxB=cutxB, cutyB=cutyB, Vj=Vj, Vsc=Vsc, mu=mu, gamx=gamx, gamy=gamy, gamz=gamz, alpha=alpha, qx=QX11, qy=QY11, g_normal=g_normal, g_sc=g_sc, Tesla=Tesla, Rfactor=Rfactor, diff_g_factors=diff_g_factors, diff_alphas=diff_alphas, diff_meff=diff_meff, meff_normal=meff_normal, meff_sc=meff_sc).conjugate()
    H01 = D
    H10 = -D.conjugate()

    H = sparse.bmat([[H00, H01], [H10, H11]], format='csc', dtype = 'complex')
    return H

#######################################################

def print_matrix(M):
    for r in M:
        for c in r:
            print(c, end = " ")
        print()
