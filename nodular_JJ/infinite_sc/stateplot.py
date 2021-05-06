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
Nx = 3 #Number of lattice sites along x-direction
Ny = 500
Wj = int(1000/ay) #Junction region [A]
Wsc = int((Ny-Wj)/2)
cutx = 0 #width of nodule
cuty = 0 #height of nodule
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
N = coor.shape[0]
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
Vj = -40*0 #junction potential: [meV]
mu = 10
gx = 1

k = 4 #This is the number of eigenvalues and eigenvectors
steps = 100 #Number of kx values that are evaluated
VVJ = 0
if Vj < 0:
    VVJ = Vj
if mu < 1:
    muf = 5
else:
    muf = mu
qmax = np.sqrt(2*(muf-VVJ)*m_eff/const.hbsqr_m0)*1.25
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
    #plt.scatter(qx[local_min_idx], bands[local_min_idx, int(k/2)], c='g', s=5, marker='X')
    #plt.scatter(kx_of_gap, gap, c='r', s=5, marker='X')
    #for i in range(int(k/2)):
        #plt.plot(qx, bands[:, int(k/2)+i], c='b')
    #plt.ylim(0, 0.5)
    #plt.show()

    H = spop.HBDG(coor, ax, ay, NN, NNb=NNb, Wj=Wj, cutxT=cutxT, cutyT=cutyT, cutxB = cutxB, cutyB = cutyB, Vj=Vj, mu=mu, alpha=alpha, delta=delta, phi=phi, gamx=gx, qx=kx_of_gap, plot_junction=False)
    H2 = spop.HBDG(coor, ax, ay, NN, NNb=NNb, Wj=Wj, cutxT=cutxT, cutyT=cutyT, cutxB = cutxB, cutyB = cutyB, Vj=Vj, mu=mu, alpha=alpha, delta=delta, phi=phi, gamx=gx, qx=-kx_of_gap, plot_junction=False)
    eigs, vecs = spLA.eigsh(H, k=k, sigma=0, which='LM')
    eigs2, vecs2 = spLA.eigsh(H2, k=k, sigma=0, which='LM')
    idx_sort = np.argsort(eigs)
    eigs = eigs[idx_sort]
    vecs = vecs[:, idx_sort]
    idx_sort2 = np.argsort(eigs2)
    eigs2 = eigs2[idx_sort2]
    vecs2 = vecs2[:, idx_sort2]
    print(eigs, eigs2)
    nth = 0
    n = int(k/2) + nth
    n2 = int(k/2) - 1
    idx1 = 239*Nx-4*Nx+1
    idx2 = 260*Nx+4*Nx+1
    print(eigs[n], eigs2[n2])

    #plots.lattice(idx1, coor, NN = NN, NNb = NNb)
    #plots.lattice(idx2, coor, NN = NN, NNb = NNb)

    sigma_x_tau_x = np.array([[0,0,0,1], [0,0,1,0], [0,1,0,0], [1,0,0,0]])
    VEC = np.array([vecs[idx1, n], vecs[idx1+N, n], vecs[idx1+2*N, n], vecs[idx1+3*N, n]])
    VEC = np.matmul(sigma_x_tau_x, VEC)
    #phi=pi phase same
    #phi=0 complex conj (K) of vecs2 phase same
    #print(VEC[0], vecs2[idx2, n2])
    #print(VEC[1], vecs2[idx2+N, n2])
    #print(VEC[2], vecs2[idx2+2*N, n2])
    #print(VEC[3], vecs2[idx2+3*N, n2])
    #print(VEC[0]/vecs2[idx2, n2]) #phase
    #print(VEC[1]/vecs2[idx2+N, n2]) #phase
    #print(VEC[2]/vecs2[idx2+2*N, n2]) #phase
    #print(VEC[3]/vecs2[idx2+3*N, n2]) #phase
    #print(vecs[idx1, n], vecs2[idx2, n2])
    #print(vecs[idx1+N, n], vecs2[idx2+N, n2])
    #print(vecs[idx1+2*N, n], vecs2[idx2+2*N, n2])
    #print(vecs[idx1+3*N, n], vecs2[idx2+3*N, n2])
    #print(vecs[idx1, n]/vecs2[idx2, n2])
    #print(vecs[idx1+N, n]/vecs2[idx2+N, n2])
    #print(vecs[idx1+2*N, n]/vecs2[idx2+2*N, n2])
    #print(vecs[idx1+3*N, n]/vecs2[idx2+3*N, n2])
    #sys.exit()

    N = coor.shape[0]
    num_div = int(vecs.shape[0]/N)
    probdens = np.square(abs(vecs[:, n]))
    map = np.zeros(N)
    for i in range(num_div):
        map[:] = map[:] + probdens[i*N:(i+1)*N]

    wt_scT = 0
    wt_scB = 0
    wt_junc = 0
    for i in range(coor.shape[0]):
        bool_inSC, which = check.is_in_SC(i, coor, Wsc, Wj, cutxT=cutxT, cutyT=cutyT, cutxB=cutxB, cutyB=cutyB)
        if bool_inSC:
            if which == 'T':
                wt_scT += map[i]
            if which == 'B':
                wt_scB += map[i]
        else:
            wt_junc += map[i]

    print("Weight in Top SC: ", wt_scT)
    print("Weight in Bottom SC: ", wt_scB)
    print("Weight in junction: ", wt_junc)
    print("Total weight: ", wt_scT+wt_junc+wt_scB)
    plots.probdens_cmap(coor, Wj, cutxT=cutxT, cutxB=cutxB, cutyT=cutyT, cutyB=cutyB, eigs=eigs, states=vecs, n=n)
    plots.state_cmap(coor, Wj, cutxT=cutxT, cutxB=cutxB, cutyT=cutyT, cutyB=cutyB, eigs=eigs, states=vecs, n=n)
