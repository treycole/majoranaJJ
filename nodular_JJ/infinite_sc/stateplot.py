import sys
import time
import os
import gc

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.signal import argrelextrema
import scipy.sparse.linalg as spLA

import majoranaJJ.modules.SNRG as SNRG
import majoranaJJ.operators.sparse_operators as spop
import majoranaJJ.lattice.nbrs as nb #neighbor arrays
import majoranaJJ.lattice.shapes as shps #lattice shapes
import majoranaJJ.modules.plots as plots #plotting functions
import majoranaJJ.modules.finders as finders
import majoranaJJ.modules.checkers as check
import majoranaJJ.modules.constants as const
import scipy.interpolate as interp
###################################################
#Defining System
ax = 50 #lattice spacing in x-direction: [A]
ay = 50 #lattice spacing in y-direction: [A]
Nx = 12 #Number of lattice sites along x-direction
Ny = 300

Wj = int(1000/ay) #Junction region [A]
Wsc = int((Ny-Wj)/2)
cutx = 4 #width of nodule
cuty = 8 #height of nodule
cutxT = cutx
cutxB = cutx
cutyT = 2*cuty
cutyB = 0

coor = shps.square(Nx, Ny) #square lattice
NN = nb.NN_Arr(coor) #neighbor array
NNb = nb.Bound_Arr(coor) #boundary array]

Junc_width = Wj*.1*ay #nm
cutxT_width = cutxT*ax*.1 #nm
cutyT_width = cutyT*ax*.1 #nm
cutxB_width = cutxB*ax*.1 #nm
cutyB_width = cutyB*ax*.1 #nm
Lx = Nx*ax #Angstrom
Lx = (max(coor[:, 0]) - min(coor[:, 0]) + 1)*ax #Unit cell size in x-direction
Ly = (max(coor[:, 1]) - min(coor[:, 1]) + 1)*ay #Unit cell size in y-direction
print("Lx = ", Lx*.1, "(nm)" )
print("Top Nodule Width in x-direction = ", cutxT_width, "(nm)")
print("Bottom Nodule Width in x-direction = ", cutxB_width, "(nm)")
print("Top Nodule Width in y-direction = ", cutyT_width, "(nm)")
print("Bottom Nodule Width in y-direction = ", cutyB_width, "(nm)")
print("Junction Width = ", Junc_width, "(nm)")
#########################################
#Defining Hamiltonian parameters
m_eff = 0.026
alpha = 200 #Spin-Orbit Coupling constant: [meV*A]
delta = 0.3 #Superconducting Gap: [meV]
phi = np.pi #SC phase difference
Vj = -40 #junction potential: [meV]
mu = 5.9148
gx = 1

k = 4 #This is the number of eigenvalues and eigenvectors
steps = 100 #Number of kx values that are evaluated
VVJ = 0
if Vj < 0:
    VVJ = Vj
if mu < 1:
    muf = 5
qmax = np.sqrt(2*(mu-VVJ)*m_eff/const.hbsqr_m0)*1.25
if qmax >= np.pi/Lx or cutxT != 0 or cutxB != 0:
    qmax = np.pi/(Lx)
qx = np.linspace(0, qmax, steps)
###################################################
dirS = 'bands_data'
if not os.path.exists(dirS):
    os.makedirs(dirS)
try:
    PLOT = str(sys.argv[1])
except:
    PLOT = 'F'
if PLOT != 'P':
    bands = np.zeros((steps, k))
    for i in range(steps):
        print(steps - i)
        H = spop.HBDG(coor, ax, ay, NN, NNb=NNb, Wj=Wj, cutxT=cutxT, cutyT=cutyT, cutxB = cutxB, cutyB = cutyB, Vj=Vj, mu=mu, alpha=alpha, delta=delta, phi=phi, gamx=gx, qx=qx[i])
        eigs, vecs = spLA.eigsh(H, k=k, sigma=0, which='LM')
        idx_sort = np.argsort(eigs)
        eigs = eigs[idx_sort]
        bands[i, :] = eigs
        np.save("%s/bands Lx = %.1f Wj = %.1f cutxT = %.1f cutyT = %.1f cutxB = %.1f cutyB = %.1f Vj = %.1f phi = %.3f mu = %.1f gam = %.1f.npy" % (dirS, Lx*.1, Junc_width, cutxT_width, cutyT_width, cutxB_width, cutyB_width, Vj,  phi, mu, gx), bands)

    local_min_idx = np.array(argrelextrema(bands, np.less)[0])
    mins = []
    kx_of_mins = []
    for i in range(local_min_idx.shape[0]):
        print("i: ", local_min_idx.shape[0]-i)
        if bands[local_min_idx[i], int(k/2)] >= 1.1*min(bands[:, int(k/2)]):
            pass
        else:
            qx_lower = qx[local_min_idx[i]-1]
            qx_c = qx[local_min_idx[i]]
            qx_higher = qx[local_min_idx[i]+1]
            deltaq = qx_higher - qx_lower
            kx_finer = np.linspace(qx_lower, qx_higher, 20)
            bands_finer = np.zeros((kx_finer.size))
            for j in range(kx_finer.shape[0]):
                print(kx_finer.shape[0] - j)
                H = spop.HBDG(coor, ax, ay, NN, NNb=NNb, Wj=Wj, cutxT=cutxT, cutyT=cutyT, cutxB = cutxB, cutyB = cutyB, Vj=Vj, mu=mu, alpha=alpha, delta=delta, phi=phi, gamx=gx, qx=kx_finer[j])
                eigs, vecs = spLA.eigsh(H, k=k, sigma=0, which='LM')
                idx_sort = np.argsort(eigs)
                eigs = eigs[idx_sort]
                bands_finer[j] = eigs[int(k/2)]
            GAP, IDX = finders.minima(bands_finer)
            mins.append(GAP)
            kx_of_mins.append(kx_finer[IDX])

    mins = np.array(mins)
    gap, idx = finders.minima(mins)
    kx_of_gap = kx_of_mins[idx]
    np.save("%s/gap Lx = %.1f Wj = %.1f cutxT = %.1f cutyT = %.1f cutxB = %.1f cutyB = %.1f Vj = %.1f phi = %.3f mu = %.1f gam = %.1f.npy" % (dirS, Lx*.1, Junc_width, cutxT_width, cutyT_width, cutxB_width, cutyB_width, Vj,  phi, mu, gx), gap)
    np.save("%s/kxofgap Lx = %.1f Wj = %.1f cutxT = %.1f cutyT = %.1f cutxB = %.1f cutyB = %.1f Vj = %.1f phi = %.3f mu = %.1f gam = %.1f.npy" % (dirS, Lx*.1, Junc_width, cutxT_width, cutyT_width, cutxB_width, cutyB_width, Vj,  phi, mu, gx), kx_of_gap)
    gc.collect()
    sys.exit()
else:
    bands = np.load("%s/bands Lx = %.1f Wj = %.1f cutxT = %.1f cutyT = %.1f cutxB = %.1f cutyB = %.1f Vj = %.1f phi = %.3f mu = %.1f gam = %.1f.npy" % (dirS, Lx*.1, Junc_width, cutxT_width, cutyT_width, cutxB_width, cutyB_width, Vj,  phi, mu, gx))
    gap = np.load("%s/gap Lx = %.1f Wj = %.1f cutxT = %.1f cutyT = %.1f cutxB = %.1f cutyB = %.1f Vj = %.1f phi = %.3f mu = %.1f gam = %.1f.npy" % (dirS, Lx*.1, Junc_width, cutxT_width, cutyT_width, cutxB_width, cutyB_width, Vj,  phi, mu, gx))
    kx_of_gap = np.load("%s/kxofgap Lx = %.1f Wj = %.1f cutxT = %.1f cutyT = %.1f cutxB = %.1f cutyB = %.1f Vj = %.1f phi = %.3f mu = %.1f gam = %.1f.npy" % (dirS, Lx*.1, Junc_width, cutxT_width, cutyT_width, cutxB_width, cutyB_width, Vj,  phi, mu, gx))

    local_min_idx = np.array(argrelextrema(bands, np.less)[0])
    plt.scatter(qx[local_min_idx], bands[local_min_idx, int(k/2)], c='g', s=5, marker='X')
    plt.scatter(kx_of_gap, gap, c='r', s=5, marker='X')
    for i in range(int(k/2)):
        plt.plot(qx, bands[:, int(k/2)+i], c='b')
    plt.ylim(0, 0.5)
    plt.show()

    H = spop.HBDG(coor, ax, ay, NN, NNb=NNb, Wj=Wj, cutxT=cutxT, cutyT=cutyT, cutxB = cutxB, cutyB = cutyB, Vj=Vj, mu=mu, alpha=alpha, delta=delta, phi=phi, gamx=gx, qx=kx_of_gap)
    eigs, vecs = spLA.eigsh(H, k=k, sigma=0, which='LM')
    idx_sort = np.argsort(eigs)
    eigs = eigs[idx_sort]
    vecs = vecs[:, idx_sort]
    nth = 0
    n = int(k/2) + nth
    N = coor.shape[0]
    num_div = int(vecs.shape[0]/N)
    probdens = np.square(abs(vecs[:, n]))
    map = np.zeros(N)
    for i in range(num_div):
        map[:] = map[:] + probdens[i*N:(i+1)*N]

    wt_sc = 0
    wt_junc = 0
    for i in range(coor.shape[0]):
        bool_inSC, which = check.is_in_SC(i, coor, Wsc, Wj, cutxT=cutxT, cutyT=cutyT, cutxB=cutxB, cutyB=cutyB)
        if bool_inSC:
            wt_sc += map[i]
        else:
            wt_junc += map[i]
    print("Weight in SC: ", wt_sc)
    print("Weight in junction: ", wt_junc)
    print("Total weight: ", wt_sc+wt_junc)
    plots.state_cmap(coor, eigs, vecs, n=n)
