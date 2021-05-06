import sys
import os
import gc

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.signal import argrelextrema

import majoranaJJ.modules.SNRG as SNRG
import majoranaJJ.modules.fig_params as params
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

print("Lx = ", Lx*.1, "(nm)" )
print("Top Nodule Width in x-direction = ", cutxT_width, "(nm)")
print("Bottom Nodule Width in x-direction = ", cutxB_width, "(nm)")
print("Top Nodule Width in y-direction = ", cutyT_width, "(nm)")
print("Bottom Nodule Width in y-direction = ", cutyB_width, "(nm)")
print("Junction Width = ", Junc_width, "(nm)")
###################################################
#Defining Hamiltonian parameters
alpha = 200 #Spin-Orbit Coupling constant: [meV*A]
phi = 0 #SC phase difference
delta = 0.3 #Superconducting Gap: [meV]
Vsc = 0 #SC potential: [meV]
Vj = np.array([0, 5, 10, 15, 20, 25]) #Junction potential: [meV]

gi = 0
gf = 3
res = 0.05
steps_gam = int((gf - gi)/res)
gx = np.linspace(gi, gf, steps_gam)

k = 4
gap_gam = np.zeros((Vj.shape[0], gx.shape[0]))
kx_of_gap = np.zeros((Vj.shape[0], gx.shape[0]))
###################################################
dirS = 'boundary_data'
min_Ez = np.load("%s/min_EZfxVj Lx = %.1f Wj = %.1f cutxT = %.1f cutyT = %.1f cutxB = %.1f cutyB = %.1f phi = %.3f Vj_i = %.1f Vj_f = %.1f.npy" % (dirS, Lx*.1, Junc_width, cutxT_width, cutyT_width, cutxB_width, cutyB_width, phi, 0, 40))
min_mu = np.load("%s/min_mufxVj Lx = %.1f Wj = %.1f cutxT = %.1f cutyT = %.1f cutxB = %.1f cutyB = %.1f phi = %.3f Vj_i = %.1f Vj_f = %.1f.npy" % (dirS, Lx*.1, Junc_width, cutxT_width, cutyT_width, cutxB_width, cutyB_width, phi, 0, 40))
dirS = 'gap_data'
#gap = np.load("%s/gapfxgamATminmu Wj = %.1f Lx = %.1f cutxT = %.1f cutyT = %.1f cutxB = %.1f cutyB = %.1f phi = %.3f gam_i = %.1f gam_f = %.1f.npy" % (dirS, Junc_width, Lx*.1, cutxT_width,  cutyT_width, cutxB_width, cutyB_width, phi, gi, gf))
#gx = np.linspace(gi,gf, gap.shape[0])
#kx_of_gap = np.load("%s/kxofgapfxgamATminmu Wj = %.1f Lx = %.1f cutxT = %.1f cutyT = %.1f cutxB = %.1f cutyB = %.1f phi = %.3f gam_i = %.1f gam_f = %.1f.npy" % (dirS, Junc_width, Lx*.1, cutxT_width,  cutyT_width, cutxB_width, cutyB_width, phi, gi, gf))
if not os.path.exists(dirS):
    os.makedirs(dirS)
try:
    PLOT = str(sys.argv[1])
except:
    PLOT = 'F'
if PLOT != 'P':
    for i in range(0, Vj.shape[0]):
        for j in range(gx.shape[0]):
            print(Vj.shape[0]-i, "| Vj =", Vj[i])
            print(gx.shape[0]-j, "| gx =", gx[j])
            GAP, KX = SNRG.gap(Wj=Wj, Lx=Lx, cutxT=cutxT, cutyT=cutyT, cutxB=cutxB, cutyB=cutyB, ax=ax, ay=ay, gam=gx[j], mu=min_mu[i], Vj=Vj[i], alpha=alpha, delta=delta, phi=phi, targ_steps=5000, n_avg=3, muf=min_mu[i], PLOT=False, tol=1e-7)
            gap_gam[i,j] = GAP
            kx_of_gap[i,j] = KX
            np.save("%s/gapfxgamATminmu Wj = %.1f Lx = %.1f cutxT = %.1f cutyT = %.1f cutxB = %.1f cutyB = %.1f phi = %.3f gam_i = %.1f gam_f = %.1f.npy" % (dirS, Junc_width, Lx*.1, cutxT_width,  cutyT_width, cutxB_width, cutyB_width, phi, gi, gf), gap_gam)
            np.save("%s/kxofgapfxgamATminmu Wj = %.1f Lx = %.1f cutxT = %.1f cutyT = %.1f cutxB = %.1f cutyB = %.1f phi = %.3f gam_i = %.1f gam_f = %.1f.npy" % (dirS, Junc_width, Lx*.1, cutxT_width,  cutyT_width, cutxB_width, cutyB_width, phi, gi, gf), kx_of_gap)
            gc.collect()

    sys.exit()
else:
    gap = np.load("%s/gapfxgamATminmu Wj = %.1f Lx = %.1f cutxT = %.1f cutyT = %.1f cutxB = %.1f cutyB = %.1f phi = %.3f gam_i = %.1f gam_f = %.1f.npy" % (dirS, Junc_width, Lx*.1, cutxT_width,  cutyT_width, cutxB_width, cutyB_width, phi, gi, gf))
    #gx = np.linspace(gi,gf, gap.shape[0])
    kx_of_gap = np.load("%s/kxofgapfxgamATminmu Wj = %.1f Lx = %.1f cutxT = %.1f cutyT = %.1f cutxB = %.1f cutyB = %.1f phi = %.3f gam_i = %.1f gam_f = %.1f.npy" % (dirS, Junc_width, Lx*.1, cutxT_width,  cutyT_width, cutxB_width, cutyB_width, phi, gi, gf))

    """
    top_arr = np.zeros(gx.shape[0])
    num = 1
    local_min_idx = np.array(argrelextrema(gap, np.less)[0])
    lower_bound = 0
    top_arr[lower_bound:] = num
    for i in range(local_min_idx.shape[0]):
        lower_bound = local_min_idx[i]
        if gap[local_min_idx[i]]/delta < 0.02 and (Lx*kx_of_gap[local_min_idx[i]] <= 0.1 or abs(Lx*kx_of_gap[local_min_idx[i]] - np.pi) < .15):
            num=num*-1
            gap[local_min_idx[i]] = 0
        top_arr[lower_bound:] = num
    """
    fig, axs = plt.subplots(3, gridspec_kw={'hspace':0.2}, sharex=True)

    #art = axs[0].fill_between(gx, gap/delta, visible=True, alpha=1, color='lightcyan', where=top_arr[:]<0)
    #art.set_edgecolor('k')
    #art = axs[1].fill_between(gx, Lx*kx_of_gap[:], visible=True, alpha=1, color='lightcyan', where=top_arr[:]<0)
    #art.set_edgecolor('k')

    axs[0].grid()
    axs[1].grid()
    axs[2].grid()

    print(gx.shape, gap.shape)
    axs[0].plot(gx, gap[0, :]/delta, c='k', lw=2, zorder=1)
    axs[1].plot(gx, gap[1, :]/delta, c='k', lw=2, zorder=1)
    axs[2].plot(gx, gap[2, :]/delta, c='k', lw=2, zorder=1)

    axs[2].set_xlabel(r'$E_Z$ (meV)')

    axs[0].set_ylabel(r'$\Delta_{qp}/\Delta_{0}$')
    axs[1].set_ylabel(r'$\Delta_{qp}/\Delta_{0}$')
    axs[2].set_ylabel(r'$\Delta_{qp}/\Delta_{0}$')

    plt.subplots_adjust(top=0.95, left=0.18, bottom=0.15, right=0.98)
    plt.savefig('FIG19', dpi=700)
    plt.show()

    fig, axs = plt.subplots(3, gridspec_kw={'hspace':0.2}, sharex=True)

    #art = axs[0].fill_between(gx, gap/delta, visible=True, alpha=1, color='lightcyan', where=top_arr[:]<0)
    #art.set_edgecolor('k')
    #art = axs[1].fill_between(gx, Lx*kx_of_gap[:], visible=True, alpha=1, color='lightcyan', where=top_arr[:]<0)
    #art.set_edgecolor('k')

    axs[0].grid()
    axs[1].grid()
    axs[2].grid()

    print(gx.shape, gap.shape)
    axs[0].plot(gx, gap[3, :]/delta, c='k', lw=2, zorder=1)
    axs[1].plot(gx, gap[4, :]/delta, c='k', lw=2, zorder=1)
    axs[2].plot(gx, gap[5, :]/delta, c='k', lw=2, zorder=1)

    axs[2].set_xlabel(r'$E_Z$ (meV)')

    axs[0].set_ylabel(r'$\Delta_{qp}/\Delta_{0}$')
    axs[1].set_ylabel(r'$\Delta_{qp}/\Delta_{0}$')
    axs[2].set_ylabel(r'$\Delta_{qp}/\Delta_{0}$')

    plt.subplots_adjust(top=0.95, left=0.18, bottom=0.15, right=0.98)
    plt.savefig('FIG20', dpi=700)
    plt.show()
