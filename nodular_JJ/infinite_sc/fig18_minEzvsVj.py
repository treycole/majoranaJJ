import sys
import time
import os
import gc

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import ticker
from scipy.signal import argrelextrema
import scipy.sparse.linalg as spLA

import majoranaJJ.modules.SNRG as SNRG
import majoranaJJ.operators.sparse_operators as spop
import majoranaJJ.modules.checkers as check
import majoranaJJ.modules.constants as const
import majoranaJJ.modules.fig_params as params
import majoranaJJ.modules.finders as fndrs
import majoranaJJ.lattice.shapes as shps #lattice shapes
import majoranaJJ.lattice.nbrs as nb #neighbor arrays

###################################################
#Defining System
ax = 50 #lattice spacing in x-direction: [A]
ay = 50 #lattice spacing in y-direction: [A]
Nx = 3 #Number of lattice sites along x-direction
Wj = 1000 #Junction region [A]
cutx = 0 #width of nodule
cuty = 0 #height of nodule
cutxT = cutx
cutxB = cutx
cutyT = cuty
cutyB = cuty
Lx = Nx*ax #Angstrom

Junc_width = Wj*.1 #nm
cutxT_width = cutxT*ax*.1 #nm
cutyT_width = cutyT*ax*.1 #nm
cutxB_width = cutxB*ax*.1 #nm
cutyB_width = cutyB*ax*.1 #nm

Ny = 500
Wj_int = int(1000/ay) #Junction region [A]
Wsc = int((Ny-Wj_int)/2)
coor = shps.square(Nx, Ny) #square lattice
NN = nb.NN_Arr(coor) #neighbor array
NNb = nb.Bound_Arr(coor) #boundary array]

Lx = (max(coor[:, 0]) - min(coor[:, 0]) + 1)*ax #Unit cell size in x-direction
Ly = (max(coor[:, 1]) - min(coor[:, 1]) + 1)*ay #Unit cell size in y-direction

print("Lx = ", Lx*.1, "(nm)" )
print("Top Nodule Width in x-direction = ", cutxT_width, "(nm)")
print("Bottom Nodule Width in x-direction = ", cutxB_width, "(nm)")
print("Top Nodule Width in y-direction = ", cutyT_width, "(nm)")
print("Bottom Nodule Width in y-direction = ", cutyB_width, "(nm)")
print("Junction Width = ", Wj*.1, "(nm)")
print()
#########################################
#Defining Hamiltonian parameters
m_eff = 0.026
alpha = 200 #Spin-Orbit Coupling constant: [meV*A]
phi = 0 #SC phase difference
delta = 0.30 #Superconducting Gap: [meV]

Vj_i = 0 #Junction potential: [meV]
Vj_f = 40
Vj = np.linspace(Vj_i, Vj_f, 9)
#print(Vj)

gi = 0 #mev
gf = 10

min_Ez = np.zeros(Vj.shape[0])
min_mu = np.zeros(Vj.shape[0])
###################################################
dirS = 'boundary_data'
if not os.path.exists(dirS):
    os.makedirs(dirS)
try:
    PLOT = str(sys.argv[1])
except:
    PLOT = 'F'
if PLOT != 'P':
    min_Ez = np.load("%s/min_EZfxVj Lx = %.1f Wj = %.1f cutxT = %.1f cutyT = %.1f cutxB = %.1f cutyB = %.1f phi = %.3f Vj_i = %.1f Vj_f = %.1f.npy" % (dirS, Lx*.1, Junc_width, cutxT_width, cutyT_width, cutxB_width, cutyB_width, phi, Vj_i, Vj_f))
    min_mu = np.load("%s/min_mufxVj Lx = %.1f Wj = %.1f cutxT = %.1f cutyT = %.1f cutxB = %.1f cutyB = %.1f phi = %.3f Vj_i = %.1f Vj_f = %.1f.npy" % (dirS, Lx*.1, Junc_width, cutxT_width, cutyT_width, cutxB_width, cutyB_width, phi, Vj_i, Vj_f))
    WT_SC = np.load("%s/wt_scfxVj Lx = %.1f Wj = %.1f cutxT = %.1f cutyT = %.1f cutxB = %.1f cutyB = %.1f phi = %.3f Vj_i = %.1f Vj_f = %.1f.npy" % (dirS, Lx*.1, Junc_width, cutxT_width, cutyT_width, cutxB_width, cutyB_width, phi, Vj_i, Vj_f))

    res = 0.05
    num_bound = 4
    boundary = np.zeros((Vj.shape[0], int(10/res), num_bound))
    for i in range(0, Vj.shape[0]):
        mu_i = Vj[i] - 5
        mu_f = Vj[i] + 5
        delta_mu = mu_f - mu_i
        mu_steps = int(delta_mu/res)
        mu = np.linspace(mu_i, mu_f, mu_steps) #Chemical Potential: [meV]
        for j in range(0, mu.shape[0]):
            print(Vj.shape[0]-i, "| Vj =", Vj[i], "meV")
            print(mu.shape[0]-j, "| mu =", mu[j], "meV")
            gcrit = fndrs.SNRG_gam_finder(ax, ay, mu[j], gi, gf, Wj=Wj, Lx=Lx, cutxT=cutxT, cutyT=cutyT, cutxB=cutxB, cutyB=cutyB, Vj=Vj[i], alpha=alpha, delta=delta, phi=phi, k=44, tol = 1e-6, PLOT=False, plot_junction=False)

            for z in range(num_bound):
                if z >= gcrit.size:
                    boundary[i, j, z] = 10
                else:
                    boundary[i, j, z] = gcrit[z]

        min_Ez[i], idx = fndrs.minima(boundary[i, :, 0])
        min_mu[i] = mu[idx]
        np.save("%s/boundary Lx = %.1f Wj = %.1f cutxT = %.1f cutyT = %.1f cutxB = %.1f cutyB = %.1f Vj = %.1f phi = %.3f mu_i = %.1f mu_f = %.1f.npy" % (dirS, Lx*.1, Junc_width, cutxT_width, cutyT_width, cutxB_width, cutyB_width, Vj[i], phi, mu_i, mu_f), boundary)
        np.save("%s/min_EZfxVj Lx = %.1f Wj = %.1f cutxT = %.1f cutyT = %.1f cutxB = %.1f cutyB = %.1f phi = %.3f Vj_i = %.1f Vj_f = %.1f.npy" % (dirS, Lx*.1, Junc_width, cutxT_width, cutyT_width, cutxB_width, cutyB_width, phi, Vj_i, Vj_f), min_Ez)
        np.save("%s/min_mufxVj Lx = %.1f Wj = %.1f cutxT = %.1f cutyT = %.1f cutxB = %.1f cutyB = %.1f phi = %.3f Vj_i = %.1f Vj_f = %.1f.npy" % (dirS, Lx*.1, Junc_width, cutxT_width, cutyT_width, cutxB_width, cutyB_width, phi, Vj_i, Vj_f), min_mu)
        gc.collect()

    steps = 50
    k = 4
    WT_SC = np.zeros(Vj.shape[0])
    for i in range(Vj.shape[0]):
        print(Vj.shape[0] - i)
        VVJ = 0
        if Vj[i] < 0:
            VVJ = Vj[i]
        qmax = np.sqrt(2*(min_mu[i]-VVJ)*m_eff/const.hbsqr_m0)*1.25
        qx = np.linspace(0, qmax, steps)
        bands = np.zeros(steps)
        for j in range(steps):
            print(Vj.shape[0] - i)
            print(steps - j)
            H = spop.HBDG(coor, ax, ay, NN, NNb=NNb, Wj=Wj_int, cutxT=cutxT, cutyT=cutyT, cutxB=cutxB, cutyB=cutyB, Vj=Vj[i], mu=min_mu[i], alpha=alpha, delta=delta, phi=phi, gamx=min_Ez[i], qx=qx[j])
            eigs, vecs = spLA.eigsh(H, k=k, sigma=0, which='LM')
            idx_sort = np.argsort(eigs)
            eigs = eigs[idx_sort]
            bands[j] = eigs[int(k/2)]

        local_min_idx = np.array(argrelextrema(bands, np.less)[0])
        mins = []
        kx_of_mins = []
        mins.append(bands[0])
        kx_of_mins.append(0)
        mins.append(bands[-1])
        kx_of_mins.append(qx[-1])
        for l in range(local_min_idx.shape[0]):
            print("l: ", local_min_idx.shape[0]-l)
            if bands[local_min_idx[l]] >= 1.1*min(bands):
                pass
            else:
                qx_lower = qx[local_min_idx[l]-1]
                qx_c = qx[local_min_idx[l]]
                qx_higher = qx[local_min_idx[l]+1]
                deltaq = qx_higher - qx_lower
                kx_finer = np.linspace(qx_lower, qx_higher, 20)
                bands_finer = np.zeros((kx_finer.size))
                for m in range(kx_finer.shape[0]):
                    print(kx_finer.shape[0] - m)
                    H = spop.HBDG(coor, ax, ay, NN, NNb=NNb, Wj=Wj_int, cutxT=cutxT, cutyT=cutyT, cutxB=cutxB, cutyB=cutyB, Vj=Vj[i], mu=min_mu[i], alpha=alpha, delta=delta, phi=phi, gamx=min_Ez[i], qx=kx_finer[m])
                    eigs, vecs = spLA.eigsh(H, k=k, sigma=0, which='LM')
                    idx_sort = np.argsort(eigs)
                    eigs = eigs[idx_sort]
                    bands_finer[m] = eigs[int(k/2)]
                GAP, IDX = fndrs.minima(bands_finer)
                mins.append(GAP)
                kx_of_mins.append(kx_finer[IDX])

        mins = np.array(mins)
        gap, idx = fndrs.minima(mins)
        kx_of_gap = kx_of_mins[idx]
        H = spop.HBDG(coor, ax, ay, NN, NNb=NNb, Wj=Wj_int, cutxT=cutxT, cutyT=cutyT, cutxB=cutxB, cutyB=cutyB, Vj=Vj[i], mu=min_mu[i], alpha=alpha, delta=delta, phi=phi, gamx=min_Ez[i], qx=kx_of_gap)
        eigs, vecs = spLA.eigsh(H, k=k, sigma=0, which='LM')
        wt_junc, wt_scT, wt_scB = fndrs.weights(eigs, vecs, coor, Wsc, Wj_int, cutxT, cutyT, cutxB, cutyB, k)
        wt_sc = wt_scT+wt_scB
        WT_SC[i] = wt_sc
        np.save("%s/wt_scfxVj Lx = %.1f Wj = %.1f cutxT = %.1f cutyT = %.1f cutxB = %.1f cutyB = %.1f phi = %.3f Vj_i = %.1f Vj_f = %.1f.npy" % (dirS, Lx*.1, Junc_width, cutxT_width, cutyT_width, cutxB_width, cutyB_width, phi, Vj_i, Vj_f), WT_SC)
        gc.collect()
    sys.exit()
else:
    min_Ez = np.load("%s/min_EZfxVj Lx = %.1f Wj = %.1f cutxT = %.1f cutyT = %.1f cutxB = %.1f cutyB = %.1f phi = %.3f Vj_i = %.1f Vj_f = %.1f.npy" % (dirS, Lx*.1, Junc_width, cutxT_width, cutyT_width, cutxB_width, cutyB_width, phi, Vj_i, Vj_f))
    min_mu = np.load("%s/min_mufxVj Lx = %.1f Wj = %.1f cutxT = %.1f cutyT = %.1f cutxB = %.1f cutyB = %.1f phi = %.3f Vj_i = %.1f Vj_f = %.1f.npy" % (dirS, Lx*.1, Junc_width, cutxT_width, cutyT_width, cutxB_width, cutyB_width, phi, Vj_i, Vj_f))
    WT_SC = np.load("%s/wt_scfxVj Lx = %.1f Wj = %.1f cutxT = %.1f cutyT = %.1f cutxB = %.1f cutyB = %.1f phi = %.3f Vj_i = %.1f Vj_f = %.1f.npy" % (dirS, Lx*.1, Junc_width, cutxT_width, cutyT_width, cutxB_width, cutyB_width, phi, Vj_i, Vj_f))

    x = np.linspace(-5, 42, 1000)
    P = np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        P[i] = fndrs.Lagrange_interp(x[i], min_mu, Vj)
    plt.plot(x, P, c='b', ls='--', zorder=2)
    plt.scatter(min_mu, Vj, c='r', zorder=1, s =1)
    plt.xlabel('mu')
    plt.ylabel('Vj for min Ez')
    plt.grid()
    plt.show()
    sys.exit()
    #plt.plot(Vj, min_mu, c='r', zorder=1)
    #plt.show()
    for i in range(Vj.shape[0]):
        print(Vj[i], min_mu[i])
    fig, axs = plt.subplots(3, gridspec_kw={'hspace':0.2}, sharex=True)

    axs[0].plot(Vj, min_Ez, lw=1, zorder=1, c='b')
    axs[1].plot(Vj, min_mu, lw=1, zorder=1, c='b')
    axs[2].plot(Vj, WT_SC, lw=1, zorder=1, c='b')

    axs[2].set_xlabel(r'$V_J$ (meV)', size=9)
    axs[2].tick_params(axis='x', labelsize=9)
    axs[2].set_xticks([0, 10, 20, 30, 40])

    axs[0].set_ylabel(r'Minimum $E_Z$', size=9)
    axs[1].set_ylabel(r'$\mu$ of min($E_Z$)', size=9)
    axs[2].set_ylabel(r'$|\psi|^2$ in SC', size=9)

    axs[0].tick_params(axis='y', labelsize=9)
    axs[1].tick_params(axis='y', labelsize=9)
    axs[2].tick_params(axis='y', labelsize=9)

    axs[1].set_yticks([0, 10, 20, 30, 40])
    axs[2].set_yticks([0, 0.5, 1])

    axs[0].grid(True, zorder=2.5)
    axs[1].grid(True, zorder=2.5)
    axs[2].grid(True, zorder=2.5)
    #plt.legend(loc=1, prop={'size': 6})

    plt.subplots_adjust(top=0.95, left=0.18, bottom=0.15, right=0.98)
    plt.savefig('FIG18', dpi=700)
    plt.show()
    sys.exit()
