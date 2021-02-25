import sys
import time
import os
import gc

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.signal import argrelextrema
import majoranaJJ.modules.SNRG as SNRG
import majoranaJJ.modules.finders as finders
import majoranaJJ.modules.checkers as check
import majoranaJJ.modules.constants as const
import scipy.interpolate as interp
###################################################
#Defining System
ax = 50 #lattice spacing in x-direction: [A]
ay = 50 #lattice spacing in y-direction: [A]
Nx = 12 #Number of lattice sites along x-direction
Wj = 1000 #Junction region [A]
nodx = 4 #width of nodule
nody = 8 #height of nodule
#new_params  = check.junction_geometry_check(Nx, 1000, nodx, nody, int(Wj/ay))
#Nx = new_params[0]
#nodx = new_params[2]
print(Nx, nodx)
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
phi = 0*np.pi #SC phase difference
delta = 0.30 #Superconducting Gap: [meV]
Vj = -40 #Junction potential: [meV]
gx = 1 #mev

mu_i = -2
mu_f = 12
delta_mu = mu_f - mu_i
res = 0.01
steps = int(abs(delta_mu/res))+1
mu = np.linspace(mu_i, mu_f, steps) #meV

print("alpha = ", alpha)
print("Mu_i = ", mu_i)
print("Mu_f = ", mu_f)
print("Gamma_x = ", gx)
print("Vj = ", Vj)
print()
###################################################
dirS = 'gap_data'
if not os.path.exists(dirS):
    os.makedirs(dirS)
try:
    PLOT = str(sys.argv[1])
except:
    PLOT = 'F'
if PLOT != 'P':
    np.save("%s/mu Wj = %.1f nm Lx = %.1f nm nodx = %.1f nm nody = %.1f nm Vj = %.1f meV alpha = %.1f meVA delta = %.2f meV phi = %.2f mu_i = %.1f meV mu_f = %.1f meV gx = %.2f meV.npy" % (dirS, Junc_width, Lx*.1, Nod_widthx,  Nod_widthy, Vj,  alpha, delta, phi, mu_i, mu_f, gx), mu)
    gap = np.zeros(mu.shape[0])
    kx_of_gap = np.zeros(mu.shape[0])
    for i in range(mu.shape[0]):
        print(steps-i, "| mu =", mu[i])
        GAP, KX = SNRG.gap(Wj=Wj, Lx=Lx, nodx=nodx, nody=nody, ax=ax, ay=ay, gam=gx, mu=mu[i], Vj=Vj, alpha=alpha, delta=delta, phi=phi, targ_steps=2000, n_avg=4, muf=mu[i], PLOT=False, tol=1e-8)
        gap[i] = GAP
        kx_of_gap[i] = KX
        np.save("%s/gapfxmu Wj = %.1f nm Lx = %.1f nm nodx = %.1f nm nody = %.1f nm Vj = %.1f meV alpha = %.1f meVA delta = %.2f meV phi = %.2f mu_i = %.1f meV mu_f = %.1f meV gx = %.2f meV.npy" % (dirS, Junc_width, Lx*.1, Nod_widthx,  Nod_widthy, Vj,  alpha, delta, phi, mu_i, mu_f, gx), gap)
        np.save("%s/kxofgapfxmu Wj = %.1f nm Lx = %.1f nm nodx = %.1f nm nody = %.1f nm Vj = %.1f meV alpha = %.1f meVA delta = %.2f meV phi = %.2f mu_i = %.1f meV mu_f = %.1f meV gx = %.2f meV.npy" % (dirS, Junc_width, Lx*.1, Nod_widthx,  Nod_widthy, Vj,  alpha, delta, phi, mu_i, mu_f, gx), kx_of_gap)
        gc.collect()

    sys.exit()
else:
    gap = np.load("%s/gapfxmu Wj = %.1f nm Lx = %.1f nm nodx = %.1f nm nody = %.1f nm Vj = %.1f meV alpha = %.1f meVA delta = %.2f meV phi = %.2f mu_i = %.1f meV mu_f = %.1f meV gx = %.2f meV.npy" % (dirS, Junc_width, Lx*.1, Nod_widthx,  Nod_widthy, Vj, alpha, delta, phi, mu_i, mu_f, gx))
    kx_of_gap = np.load("%s/kxofgapfxmu Wj = %.1f nm Lx = %.1f nm nodx = %.1f nm nody = %.1f nm Vj = %.1f meV alpha = %.1f meVA delta = %.2f meV phi = %.2f mu_i = %.1f meV mu_f = %.1f meV gx = %.2f meV.npy" % (dirS, Junc_width, Lx*.1, Nod_widthx,  Nod_widthy, Vj, alpha, delta, phi, mu_i, mu_f, gx))
    mu = np.load("%s/mu Wj = %.1f nm Lx = %.1f nm nodx = %.1f nm nody = %.1f nm Vj = %.1f meV alpha = %.1f meVA delta = %.2f meV phi = %.2f mu_i = %.1f meV mu_f = %.1f meV gx = %.2f meV.npy" % (dirS, Junc_width, Lx*.1, Nod_widthx,  Nod_widthy, Vj, alpha, delta, phi, mu_i, mu_f, gx))

    local_min_idx = np.array(argrelextrema(gap, np.less)[0])

    fig, axs = plt.subplots(2, 1, gridspec_kw={'hspace':0.1}, sharex=True)
    axs[0].grid()
    axs[1].grid()
    axs[0].scatter(mu[local_min_idx], (1/delta)*gap[local_min_idx], marker='X', c=(1/delta)*gap[local_min_idx], cmap='plasma', vmax=0.05)
    axs[1].scatter(mu[local_min_idx], Lx*kx_of_gap[local_min_idx], marker='X', c=(1/delta)*gap[local_min_idx], cmap='plasma', vmax=0.05)
    #axs[0].scatter(mu, gap/delta, c='r', zorder=2, s=2)
    axs[0].plot(mu, gap/delta, c='k', lw=2, zorder=1)
    axs[1].plot(mu, kx_of_gap*Lx, c='k', lw=2)
    #plt.scatter(mu, gap/delta, c='r', zorder=2, s=2)
    #plt.plot(mu_new, gap_new, c='b', lw=2, zorder=1)
    #plt.plot(mu, gap/delta, c='b', lw=2, zorder=1)

    for ax in axs.flat:
        ax.set_xlabel(r'$\mu$ (meV)')

    for ax in fig.get_axes():
        ax.label_outer()

    axs[0].set_ylabel(r'$\Delta_{qp}/\Delta_{0}$')
    axs[1].set_ylabel(r'$k_{x}*L_{x}$')
    axs[1].set_yticks([0, np.pi/8, 2*np.pi/8, 3*np.pi/8, 4*np.pi/8, 5*np.pi/8, 6*np.pi/8, 7*np.pi/8, 8*np.pi/8])
    axs[1].set_yticklabels(['0', r'$\pi/8$', r'$2\pi/8$', r'$3\pi/8$', r'$4\pi/8$', r'$5\pi/8$', r'6$\pi/8$', r'7$\pi/8$', r'$\pi$'])
    #plt.xlabel(r'$\mu$ (meV)')
    #plt.ylabel(r'$\Delta_{qp}/\Delta_{0}$')
    #plt.ylim(0,0.3)

    #plt.subplots_adjust(top=0.75, left=0.25, bottom=0.25)
    #title = r'gx = {} Lx = {} nm, lx = {} nm, W1 = {} nm, W2 = {} nm, Vj = {} meV'.format(gx, Lx*.1, Nod_widthx, Junc_width, Junc_width-2*Nod_widthy, Vj)
    #plt.title(title, loc = 'center', wrap = True)
    #plt.savefig('gapfxmu juncwidth = {} nodwidthx = {} nodwidthy = {} alpha = {} phi = {} Vj = {}.png'.format(Junc_width, Nod_widthx, Nod_widthy, alpha, phi, Vj), dpi=700)
    plt.show()

    sys.exit()
