import sys
import time
import os
import gc

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import ticker
from scipy.signal import argrelextrema

import majoranaJJ.modules.SNRG as SNRG
import majoranaJJ.modules.finders as finders
import majoranaJJ.modules.checkers as check
import majoranaJJ.modules.constants as const
import majoranaJJ.modules.fig_params as params
###################################################
#Defining System
ax = 50 #lattice spacing in x-direction: [A]
ay = 50 #lattice spacing in y-direction: [A]
Nx = 3 #Number of lattice sites along x-direction
Wj_i = 200
Wj_f = 2000
Wj = np.linspace(Wj_i, Wj_f, 19) #Junction region [A]
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
print()
#########################################
#Defining Hamiltonian parameters
alpha = 200 #Spin-Orbit Coupling constant: [meV*A]
phi = np.pi #SC phase difference
delta = 0.30 #Superconducting Gap: [meV]
Vj = 0 #Junction potential: [meV]
gx = 2.0 #mev
mu = 10
###################################################
dirS = 'gap_data'
if not os.path.exists(dirS):
    os.makedirs(dirS)
try:
    PLOT = str(sys.argv[1])
except:
    PLOT = 'F'
if PLOT != 'P':
    gap = np.zeros(Wj.shape[0])
    kx_of_gap = np.zeros(Wj.shape[0])
    for i in range(0, Wj.shape[0]):
        print(Wj.shape[0]-i, "| Wj =", Wj[i]*.1, "nm")
        GAP, KX = SNRG.gap(Wj=Wj[i], Lx=Lx, cutxT=cutxT, cutyT=cutyT, cutxB=cutxB, cutyB=cutyB, ax=ax, ay=ay, gam=gx, mu=mu, Vj=Vj, alpha=alpha, delta=delta, phi=phi, targ_steps=50000, n_avg=4, muf=mu, PLOT=False, tol=1e-7)
        gap[i] = GAP
        kx_of_gap[i] = KX
        np.save("%s/gapfxWj Wj_i = %.1f Wj_f = %.1f Lx = %.1f cutxT = %.1f cutyT = %.1f cutxB = %.1f cutyB = %.1f Vj = %.1f phi = %.2f mu = %.1f gx = %.2f.npy" % (dirS, Wj_i, Wj_f, Lx*.1, cutxT_width,  cutyT_width, cutxB_width, cutyB_width, Vj, phi, mu, gx), gap)
        np.save("%s/kxofgapfxWj Wj_i = %.1f Wj_f = %.1f Lx = %.1f cutxT = %.1f cutyT = %.1f cutxB = %.1f cutyB = %.1f Vj = %.1f phi = %.2f mu = %.1f gx = %.2f.npy" % (dirS, Wj_i, Wj_f, Lx*.1, cutxT_width,  cutyT_width, cutxB_width, cutyB_width, Vj, phi, mu, gx), kx_of_gap)
        gc.collect()
    sys.exit()
else:
    gap1 = np.load("%s/gapfxWj Wj_i = %.1f Wj_f = %.1f Lx = %.1f cutxT = %.1f cutyT = %.1f cutxB = %.1f cutyB = %.1f Vj = %.1f phi = %.2f mu = %.1f gx = %.2f.npy" % (dirS, Wj_i, Wj_f, Lx*.1, cutxT_width,  cutyT_width, cutxB_width, cutyB_width, Vj, phi, mu, 1))
    gap2 = np.load("%s/gapfxWj Wj_i = %.1f Wj_f = %.1f Lx = %.1f cutxT = %.1f cutyT = %.1f cutxB = %.1f cutyB = %.1f Vj = %.1f phi = %.2f mu = %.1f gx = %.2f.npy" % (dirS, Wj_i, Wj_f, Lx*.1, cutxT_width,  cutyT_width, cutxB_width, cutyB_width, Vj, phi, mu, 2))
    kx_of_gap = np.load("%s/kxofgapfxWj Wj_i = %.1f Wj_f = %.1f Lx = %.1f cutxT = %.1f cutyT = %.1f cutxB = %.1f cutyB = %.1f Vj = %.1f phi = %.2f mu = %.1f gx = %.2f.npy" % (dirS, Wj_i, Wj_f, Lx*.1, cutxT_width,  cutyT_width, cutxB_width, cutyB_width, Vj, phi, mu, gx))

    fig, axs = plt.subplots(1, gridspec_kw={'hspace':0.1}, sharex=True)

    axs.plot(Wj*.1, gap1/delta, lw=2, zorder=1, c='r', label=r'$E_Z $ = 1 meV')
    axs.plot(Wj*.1, gap2/delta, lw=2, zorder=1, c='b', label=r'$E_Z $ = 2 meV')

    axs.set_xlabel(r'$W_J$ (nm)', size=9)
    axs.set_ylabel(r'$\Delta_{qp}/\Delta_{0}$', size=9)
    axs.tick_params(axis='x', labelsize=9)
    axs.tick_params(axis='y', labelsize=9)
    axs.set_ylim(0, 0.23)
    axs.set_xticks([ 0, 50, 100, 150, 200])
    plt.legend(loc=1, prop={'size': 6})

    plt.subplots_adjust(top=0.95, left=0.15, bottom=0.26, right=0.98)
    plt.savefig('FIG16', dpi=700)
    plt.show()
    sys.exit()
