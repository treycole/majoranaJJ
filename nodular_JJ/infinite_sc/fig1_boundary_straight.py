import sys
import os
import gc

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker

import majoranaJJ.modules.plots as plots #plotting functions
import majoranaJJ.modules.finders as fndrs
import majoranaJJ.modules.SNRG as SNRG
import majoranaJJ.modules.fig_params as params
import matplotlib.colors as colors
###################################################
#Defining System
ax = 50 #lattice spacing in x-direction: [A]
ay = 50 #lattice spacing in y-direction: [A]
Nx = 3 #Number of lattice sites along x-direction
Wj = 1000 #Junction region [A]
cutx = 0 #width of nodule
cuty = 0 #height of nodule
Lx = Nx*ax #Angstrom
Junc_width = Wj*.10 #nm
Nod_widthx = cutx*ax*.1 #nm
Nod_widthy = cuty*ay*.1 #nm

print("Nodule Width in x-direction = ", Nod_widthx, "(nm)")
print("Nodule Width in y-direction = ", Nod_widthy, "(nm)")
print("Junction Width = ", Junc_width, "(nm)")
###################################################
#Defining Hamiltonian parameters
alpha = 200 #Spin-Orbit Coupling constant: [meV*A]
phi = 0*np.pi #SC phase difference
delta = 0.3 #Superconducting Gap: [meV]
Vj = 0 #junction potential: [meV]

mu_i = -2
mu_f = 20
res = 0.01

gi = 0
gf = 5
num_bound = 4
###################################################
#phase diagram mu vs gam
dirS = 'boundary_data'

boundary_pi = np.load("%s/boundary Lx = %.1f Wj = %.1f nodx = %.1f nody = %.1f Vj = %.1f phi = %.3f mu_i = %.1f mu_f = %.1f.npy" % (dirS, Lx*.1, Junc_width, Nod_widthx,  Nod_widthy, Vj, np.pi, mu_i, mu_f))
boundary_0 = np.load("%s/boundary Lx = %.1f Wj = %.1f nodx = %.1f nody = %.1f Vj = %.1f phi = %.3f mu_i = %.1f mu_f = %.1f.npy" % (dirS, Lx*.1, Junc_width, Nod_widthx,  Nod_widthy, Vj, 0, mu_i, mu_f))
mu_0 = np.linspace(mu_i, mu_f, boundary_0.shape[0])
mu_pi = np.linspace(mu_i, mu_f, boundary_pi.shape[0])

fig, axs = plt.subplots(2, gridspec_kw={'hspace':0.1, 'wspace':0.1})
#axs.set_facecolor("w")
axs[0].set_yticks([0, 5, 10])
axs[1].set_yticks([0, 5, 10])
for i in range(boundary_0.shape[1]):
    axs[0].plot(boundary_0[:, i], mu_0, c='k', linewidth=1.7, zorder=2)
    axs[1].plot(boundary_pi[:, i], mu_pi, c='k', linewidth=1.7, zorder=2)

color = colors.colorConverter.to_rgba('lightblue', alpha=1.0)
#color = list(color)
#color[0] = 0.85
print(color)
axs[0].fill_betweenx(mu_0, boundary_0[:, 0], boundary_0[:, 1], visible=True, alpha=1, color=color)
axs[1].fill_betweenx(mu_pi, boundary_pi[:, 0], boundary_pi[:, 1], visible=True, alpha=1, color=color)

for ax in axs.flat:
    ax.set_xlabel(r'$E_Z$ (meV)', size=9)
    ax.set_ylabel(r'$\mu$ (mev)', size=9)
for ax in fig.get_axes():
    ax.label_outer()

axs[1].text(0.15, 10.8, '(b)', fontdict=None, size=9)
axs[0].text(0.15, 10.8, '(a)', fontdict=None, size=9)

cut1 = np.linspace(-2, 12, 100)
cut2 = np.linspace(0, 3, 100)
axs[0].plot(cut1*0+1, cut1, c='b', lw=1.5, zorder=2)
axs[0].plot(cut2, cut2*0, c='b', lw=1.5, zorder=2)
axs[1].plot(cut1*0+1, cut1, c='r', lw=1.5, zorder=2)
axs[1].plot(cut2, cut2*0+10, c='r', lw=1.5, zorder=2)
axs[1].plot(1, 10, 'o', c='r', mec='k', zorder=2.2)

axs[0].set_xlim([0, 4.2])
axs[0].set_ylim([-2.0, 13])
axs[1].set_xlim([0, 4.2])
axs[1].set_ylim([-2.0, 13])

axs[0].tick_params(axis='x', labelsize=9)
axs[1].tick_params(axis='x', labelsize=9)
axs[0].tick_params(axis='y', labelsize=9)
axs[1].tick_params(axis='y', labelsize=9)
axs[0].xaxis.set_major_locator(ticker.MultipleLocator(1))
axs[1].xaxis.set_major_locator(ticker.MultipleLocator(1))


plt.subplots_adjust(top=0.95, left=0.15, bottom=0.15, right=0.98)
plt.savefig('FIG1', dpi=700)
plt.show()
