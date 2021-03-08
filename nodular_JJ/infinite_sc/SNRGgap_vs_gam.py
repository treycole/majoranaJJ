import sys
import time
import os
import gc

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import majoranaJJ.modules.SNRG as SNRG
from scipy.signal import argrelextrema
###################################################
#Defining System
ax = 50 #lattice spacing in x-direction: [A]
ay = 50 #lattice spacing in y-direction: [A]
Nx = 12 #Number of lattice sites along x-direction
Wj = 1000 #Junction region [A]
nodx = 4 #width of nodule
nody = 8 #height of nodule
Lx = Nx*ax

Junc_width = Wj*.1 #nm
Nod_widthx = nodx*ay*.1 #nm
Nod_widthy = nody*ay*.1 #nm
print("Nodule Width in x-direction = ", Nod_widthx, "(nm)")
print("Nodule Width in y-direction = ", Nod_widthy, "(nm)")
print("Junction Width = ", Junc_width, "(nm)")
#########################################
#Defining Hamiltonian parameters
alpha = 200 #Spin-Orbit Coupling constant: [meV*A]
phi = np.pi #SC phase difference
delta = 0.3 #Superconducting Gap: [meV]
Vsc = 0 #SC potential: [meV]
Vj = -40 #Junction potential: [meV]
mu = [2.37, 6.28, 10.30] #meV

gi = 0
gf = 3
res = 0.01
steps_gam = int((gf - gi)/(0.5*res)) + 1
gx = np.linspace(gi, gf, steps_gam)

k = 4
gap_gam = np.zeros(gx.shape[0])
kx_of_gap = np.zeros(gx.shape[0])
###################################################
dirS = 'gap_data'
if not os.path.exists(dirS):
    os.makedirs(dirS)
try:
    PLOT = str(sys.argv[1])
except:
    PLOT = 'F'
if PLOT != 'P':
    np.save("%s/gamx Wj = %.1f Lx = %.1f nodx = %.1f nody = %.1f Vj = %.1f alpha = %.1f delta = %.2f phi = %.3f gam_i = %.1f gam_f = %.1f mu = %.2f.npy" % (dirS, Junc_width, Lx, Nod_widthx,  Nod_widthy, Vj,  alpha, delta, phi, gi, gf, mu[2]), gx)
    for i in range(gx.shape[0]):
        print(gx.shape[0]-i)
        GAP, KX = SNRG.gap(Wj=Wj, Lx=Lx, nodx=nodx, nody=nody, ax=ax, ay=ay, gam=gx[i], mu=mu[2], Vj=Vj, alpha=alpha, delta=delta, phi=phi, targ_steps=2000, muf=mu[2], n_avg=4, PLOT=False, tol=1e-8)
        gap_gam[i] = GAP
        kx_of_gap[i] = KX
        np.save("%s/gapfxgam Wj = %.1f Lx = %.1f nodx = %.1f nody = %.1f Vj = %.1f alpha = %.1f delta = %.2f phi = %.3f gam_i = %.1f gam_f = %.1f mu = %.2f.npy" % (dirS, Junc_width, Lx, Nod_widthx,  Nod_widthy, Vj,  alpha, delta, phi, gi, gf, mu[2]), gap_gam)
        np.save("%s/kxofgapfxgam Wj = %.1f Lx = %.1f nodx = %.1f nody = %.1f Vj = %.1f alpha = %.1f delta = %.2f phi = %.3f gam_i = %.1f gam_f = %.1f mu = %.2f.npy" % (dirS, Junc_width, Lx, Nod_widthx,  Nod_widthy, Vj,  alpha, delta, phi, gi, gf, mu[2]), kx_of_gap)
        gc.collect()

    sys.exit()
else:
    gap = np.load("%s/gapfxgam Wj = %.1f Lx = %.1f nodx = %.1f nody = %.1f Vj = %.1f alpha = %.1f delta = %.2f phi = %.3f gam_i = %.1f gam_f = %.1f mu = %.2f.npy" % (dirS, Junc_width, Lx, Nod_widthx,  Nod_widthy, Vj,  alpha, delta, phi, gi, gf, mu[2]))
    gx = np.load("%s/gamx Wj = %.1f Lx = %.1f nodx = %.1f nody = %.1f Vj = %.1f alpha = %.1f delta = %.2f phi = %.3f gam_i = %.1f gam_f = %.1f mu = %.2f.npy" % (dirS, Junc_width, Lx, Nod_widthx,  Nod_widthy, Vj,  alpha, delta, phi, gi, gf, mu[2]))
    kx_of_gap = np.load("%s/kxofgapfxgam Wj = %.1f Lx = %.1f nodx = %.1f nody = %.1f Vj = %.1f alpha = %.1f delta = %.2f phi = %.3f gam_i = %.1f gam_f = %.1f mu = %.2f.npy" % (dirS, Junc_width, Lx, Nod_widthx,  Nod_widthy, Vj,  alpha, delta, phi, gi, gf, mu[2]))

    top_arr = np.zeros(gx.shape[0])
    num = 1
    local_min_idx = np.array(argrelextrema(gap, np.less)[0])
    lower_bound = 0
    top_arr[lower_bound:] = num
    for i in range(local_min_idx.shape[0]):
        lower_bound = local_min_idx[i]
        if gap[local_min_idx[i]]/delta < 0.02 and (Lx*kx_of_gap[local_min_idx[i]] <= 0.1 or abs(Lx*kx_of_gap[local_min_idx[i]] - np.pi) < .15):
            num=num*-1
        top_arr[lower_bound:] = num

    fig, axs = plt.subplots(2, 1, gridspec_kw={'hspace':0.1}, sharex=True)

    art = axs[0].fill_between(gx, gap/delta, visible=True, alpha=1, color='lightcyan', where=top_arr[:]<0)
    art.set_edgecolor('k')
    art = axs[1].fill_between(gx, Lx*kx_of_gap[:], visible=True, alpha=1, color='lightcyan', where=top_arr[:]<0)
    art.set_edgecolor('k')

    axs[0].grid()
    axs[1].grid()

    axs[0].scatter(gx[local_min_idx], (1/delta)*gap[local_min_idx], marker='X', c=(1/delta)*gap[local_min_idx], cmap='plasma', vmax=0.05)
    axs[1].scatter(gx[local_min_idx], Lx*kx_of_gap[local_min_idx], marker='X', c=(1/delta)*gap[local_min_idx], cmap='plasma', vmax=0.05)

    #axs[0].scatter(mu, gap/delta, c='r', zorder=2, s=2)
    axs[0].plot(gx, gap/delta, c='k', lw=2, zorder=1)
    axs[1].plot(gx, kx_of_gap*Lx, c='k', lw=2)

    for ax in axs.flat:
        ax.set_xlabel(r'$\Gamma_x$ (meV)')

    for ax in fig.get_axes():
        ax.label_outer()

    axs[0].set_ylabel(r'$\Delta_{qp}/\Delta_{0}$')
    axs[1].set_ylabel(r'$k_{x}*L_{x}$')
    axs[1].set_yticks([0, np.pi/8, 2*np.pi/8, 3*np.pi/8, 4*np.pi/8, 5*np.pi/8, 6*np.pi/8, 7*np.pi/8, 8*np.pi/8])
    #plt.xlim(gi, gf)
    #plt.ylim(0, 1.0)
    #title = r"$\mu$ = %.1f, $W_j$ = %.1f nm, $nodule_x$ = %.1f nm, $nodule_y$ = %.1f nm, $V_j$ = %.1f meV, $V_{SC}$ = %.1f meV, $\phi$ = %.2f " % (mu[1], Junc_width, Nod_widthx, Nod_widthy, Vj, Vsc, phi)
    #plt.title(title, loc = 'center', wrap = True)
    #plt.subplots_adjust(top=0.85)
    plt.show()
