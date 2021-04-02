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
cutx = 4 #width of nodule
cuty = 8 #height of nodule
cutxT = cutx
cutxB = cutx
cutyT = 2*cuty
cutyB = 0
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
Vj = -40 #Junction potential: [meV]
gx = 1.0 #mev

mu_i = -2.5
mu_f = 14.0
delta_mu = mu_f - mu_i
res = 0.015
steps = int(abs(delta_mu/res))+1
mu = np.linspace(mu_i, mu_f, steps) #meV
###################################################
dirS = 'gap_data'
if not os.path.exists(dirS):
    os.makedirs(dirS)
try:
    PLOT = str(sys.argv[1])
except:
    PLOT = 'F'
if PLOT != 'P':
    gap = np.zeros(mu.shape[0])
    kx_of_gap = np.zeros(mu.shape[0])
    gap = np.load("%s/gapfxmu Wj = %.1f Lx = %.1f cutxT = %.1f cutyT = %.1f cutxB = %.1f cutyB = %.1f Vj = %.1f phi = %.2f mu_i = %.1f mu_f = %.1f gx = %.2f.npy" % (dirS, Junc_width, Lx*.1, cutxT_width,  cutyT_width, cutxB_width, cutyB_width, Vj, phi, mu_i, mu_f, gx))
    kx_of_gap = np.load("%s/kxofgapfxmu Wj = %.1f Lx = %.1f cutxT = %.1f cutyT = %.1f cutxB = %.1f cutyB = %.1f Vj = %.1f phi = %.2f mu_i = %.1f mu_f = %.1f gx = %.2f.npy" % (dirS, Junc_width, Lx*.1, cutxT_width,  cutyT_width, cutxB_width, cutyB_width, Vj, phi, mu_i, mu_f, gx))
    mu = np.load("%s/mu Wj = %.1f Lx = %.1f cutxT = %.1f cutyT = %.1f cutxB = %.1f cutyB = %.1f Vj = %.1f phi = %.2f mu_i = %.1f mu_f = %.1f gx = %.2f.npy" % (dirS, Junc_width, Lx*.1, cutxT_width,  cutyT_width, cutxB_width, cutyB_width, Vj, phi, mu_i, mu_f, gx))
    #b = 224
    #e = 298
    #print(mu[b], mu[e])
    #print(gap[895], mu[895])
    #mu1=mu[0:103]
    #gap1 = gap[0:103]
    #kx_of_gap1 = kx_of_gap[0:103]
    #mu2 = np.linspace(mu[103], mu[103+1], 10)
    #gap2 = np.zeros(10)
    #kx_of_gap2 = np.zeros(10)
    #mu3 = mu[105:]
    #gap3 = gap[105:]
    #kx_of_gap3 = kx_of_gap[105:]

    #mu = np.concatenate((mu1, mu2, mu3), axis=None)
    #print(mu[0:160])
    #sys.exit()
    np.save("%s/mu Wj = %.1f Lx = %.1f cutxT = %.1f cutyT = %.1f cutxB = %.1f cutyB = %.1f Vj = %.1f phi = %.2f mu_i = %.1f mu_f = %.1f gx = %.2f.npy" % (dirS, Junc_width, Lx*.1, cutxT_width,  cutyT_width, cutxB_width, cutyB_width, Vj, phi, mu_i, mu_f, gx), mu)
    for i in range(mu.shape[0]-270, mu.shape[0]):
        print(mu.shape[0]-i, "| mu =", mu[i])
        GAP, KX = SNRG.gap(Wj=Wj, Lx=Lx, cutxT=cutxT, cutyT=cutyT, cutxB=cutxB, cutyB=cutyB, ax=ax, ay=ay, gam=gx, mu=mu[i], Vj=Vj, alpha=alpha, delta=delta, phi=phi, targ_steps=50000, n_avg=4, muf=mu[i], PLOT=False, tol=1e-7)
        gap[i] = GAP
        kx_of_gap[i] = KX
        np.save("%s/gapfxmu Wj = %.1f Lx = %.1f cutxT = %.1f cutyT = %.1f cutxB = %.1f cutyB = %.1f Vj = %.1f phi = %.2f mu_i = %.1f mu_f = %.1f gx = %.2f.npy" % (dirS, Junc_width, Lx*.1, cutxT_width,  cutyT_width, cutxB_width, cutyB_width, Vj, phi, mu_i, mu_f, gx), gap)
        np.save("%s/kxofgapfxmu Wj = %.1f Lx = %.1f cutxT = %.1f cutyT = %.1f cutxB = %.1f cutyB = %.1f Vj = %.1f phi = %.2f mu_i = %.1f mu_f = %.1f gx = %.2f.npy" % (dirS, Junc_width, Lx*.1, cutxT_width,  cutyT_width, cutxB_width, cutyB_width, Vj, phi, mu_i, mu_f, gx), kx_of_gap)
        gc.collect()

    #gap = np.concatenate((gap1, gap2, gap3), axis=None)
    #kx_of_gap = np.concatenate((kx_of_gap1, kx_of_gap2, kx_of_gap3), axis=None)
    #np.save("%s/gapfxmu Wj = %.1f Lx = %.1f cutxT = %.1f cutyT = %.1f cutxB = %.1f cutyB = %.1f Vj = %.1f phi = %.2f mu_i = %.1f mu_f = %.1f gx = %.2f.npy" % (dirS, Junc_width, Lx*.1, cutxT_width,  cutyT_width, cutxB_width, cutyB_width, Vj, phi, mu_i, mu_f, gx), gap)
    #np.save("%s/kxofgapfxmu Wj = %.1f Lx = %.1f cutxT = %.1f cutyT = %.1f cutxB = %.1f cutyB = %.1f Vj = %.1f phi = %.2f mu_i = %.1f mu_f = %.1f gx = %.2f.npy" % (dirS, Junc_width, Lx*.1, cutxT_width,  cutyT_width, cutxB_width, cutyB_width, Vj, phi, mu_i, mu_f, gx), kx_of_gap)
    #gc.collect()

    sys.exit()
else:
    np.save("%s/mu Wj = %.1f Lx = %.1f cutxT = %.1f cutyT = %.1f cutxB = %.1f cutyB = %.1f Vj = %.1f phi = %.2f mu_i = %.1f mu_f = %.1f gx = %.2f.npy" % (dirS, Junc_width, Lx*.1, cutxT_width,  cutyT_width, cutxB_width, cutyB_width, Vj, phi, mu_i, mu_f, gx), mu)
    gap = np.load("%s/gapfxmu Wj = %.1f Lx = %.1f cutxT = %.1f cutyT = %.1f cutxB = %.1f cutyB = %.1f Vj = %.1f phi = %.2f mu_i = %.1f mu_f = %.1f gx = %.2f.npy" % (dirS, Junc_width, Lx*.1, cutxT_width,  cutyT_width, cutxB_width, cutyB_width, Vj, phi, mu_i, mu_f, gx))
    kx_of_gap = np.load("%s/kxofgapfxmu Wj = %.1f Lx = %.1f cutxT = %.1f cutyT = %.1f cutxB = %.1f cutyB = %.1f Vj = %.1f phi = %.2f mu_i = %.1f mu_f = %.1f gx = %.2f.npy" % (dirS, Junc_width, Lx*.1, cutxT_width,  cutyT_width, cutxB_width, cutyB_width, Vj, phi, mu_i, mu_f, gx))
    mu = np.load("%s/mu Wj = %.1f Lx = %.1f cutxT = %.1f cutyT = %.1f cutxB = %.1f cutyB = %.1f Vj = %.1f phi = %.2f mu_i = %.1f mu_f = %.1f gx = %.2f.npy" % (dirS, Junc_width, Lx*.1, cutxT_width,  cutyT_width, cutxB_width, cutyB_width, Vj, phi, mu_i, mu_f, gx))
    local_min_idx = np.array(argrelextrema(gap, np.less)[0])
    fig, axs = plt.subplots(2, 1, gridspec_kw={'hspace':0.1}, sharex=True)

    top_arr = np.zeros(mu.shape[0])
    num = 1
    lower_bound = 0
    top_arr[lower_bound:] = num
    for i in range(local_min_idx.shape[0]):
        lower_bound = local_min_idx[i]
        if gap[local_min_idx[i]]/delta < 0.025 and (Lx*kx_of_gap[local_min_idx[i]] <= 0.2 or abs(Lx*kx_of_gap[local_min_idx[i]] - np.pi) <= .6):
            num=num*-1
            #gap[local_min_idx[i]] = 0
        top_arr[lower_bound:] = num

    art = axs[0].fill_between(mu, gap/delta, visible=True, alpha=1, color='lightcyan', where=top_arr[:]<0)
    art.set_edgecolor('k')
    art = axs[1].fill_between(mu, Lx*kx_of_gap[:], visible=True, alpha=1, color='lightcyan', where=top_arr[:]<0)
    art.set_edgecolor('k')

    axs[0].grid()
    axs[1].grid()

    axs[0].scatter(mu[local_min_idx], (1/delta)*gap[local_min_idx], marker='X', c=(1/delta)*gap[local_min_idx], cmap='plasma')#, vmax=0.05)
    axs[1].scatter(mu[local_min_idx], Lx*kx_of_gap[local_min_idx], marker='X', c=(1/delta)*gap[local_min_idx], cmap='plasma')#, vmax=0.05)

    #axs[0].scatter(mu, gap/delta, c='r', zorder=2, s=2)
    axs[0].plot(mu, gap/delta, c='k', lw=2, zorder=1)
    axs[1].plot(mu, kx_of_gap*Lx, c='k', lw=2)

    for ax in axs.flat:
        ax.set_xlabel(r'$\mu$ (meV)')

    for ax in fig.get_axes():
        ax.label_outer()

    axs[0].set_ylabel(r'$\Delta_{qp}/\Delta_{0}$')
    axs[1].set_ylabel(r'$k_{x}*L_{x}$')
    #axs[1].set_yticks([0, np.pi/8, 2*np.pi/8, 3*np.pi/8, 4*np.pi/8, 5*np.pi/8, 6*np.pi/8, 7*np.pi/8, 8*np.pi/8])
    #axs[1].set_yticklabels(['0', r'$\pi/8$', r'$2\pi/8$', r'$3\pi/8$', r'$4\pi/8$', r'$5\pi/8$', r'6$\pi/8$', r'7$\pi/8$', r'$\pi$'])
    #plt.xlabel(r'$\mu$ (meV)')
    #plt.ylabel(r'$\Delta_{qp}/\Delta_{0}$')
    #plt.ylim(0,0.3)

    #plt.subplots_adjust(top=0.75, left=0.25, bottom=0.25)
    #title = r'gx = {} Lx = {} nm, lx = {} nm, W1 = {} nm, W2 = {} nm, Vj = {} meV'.format(gx, Lx*.1, Nod_widthx, Junc_width, Junc_width-2*Nod_widthy, Vj)
    #plt.title(title, loc = 'center', wrap = True)
    #plt.savefig('gapfxmu juncwidth = {} nodwidthx = {} nodwidthy = {} alpha = {} phi = {} Vj = {}.png'.format(Junc_width, Nod_widthx, Nod_widthy, alpha, phi, Vj), dpi=700)
    plt.show()

    sys.exit()
