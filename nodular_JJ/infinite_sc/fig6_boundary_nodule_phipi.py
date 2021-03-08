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
Nx = 12 #Number of lattice sites along x-direction
Wj = 1000 #Junction region [A]
cutx = 4 #width of nodule
cuty = 8 #height of nodule
Lx = Nx*ax #Angstrom
Junc_width = Wj*.1 #nm
Nod_widthx = cutx*ax*.1 #nm
Nod_widthy = cuty*ay*.1 #nm

print("Lx = ", Lx*.1, "(nm)" )
print("Nodule Width in x-direction = ", Nod_widthx, "(nm)")
print("Nodule Width in y-direction = ", Nod_widthy, "(nm)")
print("Junction Width = ", Junc_width, "(nm)")
###################################################
#Defining Hamiltonian parameters
alpha = 200 #Spin-Orbit Coupling constant: [meV*A]
phi = np.pi #SC phase difference
delta = 0.3 #Superconducting Gap: [meV]
Vj = -40 #junction potential: [meV]

mu_i = -5
mu_f = 15
res = 0.005
delta_mu = mu_f - mu_i
mu_steps = int(delta_mu/res)
mu = np.linspace(mu_i, mu_f, mu_steps) #Chemical Potential: [meV]
print("alpha = ", alpha)
print("Mu_i = ", mu_i)
print("Mu_f = ", mu_f)
print("Vj = ", Vj)

gi = 0
gf = 5.0
num_bound = 10
###################################################
#phase diagram mu vs gam
dirS = 'boundary_data'

boundary = np.load("%s/boundary Lx = %.1f Wj = %.1f nodx = %.1f nody = %.1f Vj = %.1f alpha = %.1f delta = %.2f phi = %.3f mu_i = %.1f mu_f = %.1f.npy" % (dirS, Lx*.1, Junc_width, Nod_widthx,  Nod_widthy, Vj, alpha, delta, phi, mu_i, mu_f))
mu = np.linspace(mu_i, mu_f, boundary.shape[0])

fig, axs = plt.subplots(1, gridspec_kw={'hspace':0.1, 'wspace':0.1})
axs.set_yticks([ 0, 5, 10])
axs.set_zorder(100)
axs.label_outer()

for i in range(mu.shape[0]-1):
    for j in range(int(boundary.shape[1]/2)):
        if np.isnan(boundary[i,2*j+1]) and not np.isnan(boundary[i,2*j]):
            boundary[i,2*j+1] = 5
            break

dist_arr = np.zeros((mu.shape[0], num_bound))
for i in range(int(mu.shape[0])-1):
    for j in range(num_bound-1):
        if np.isnan(boundary[i, j]) or np.isnan(boundary[i+1, j]):
            dist_arr[i,j] = 100000
        else:
            dist_arr[i,j] = abs(boundary[i, j] - boundary[i+1, j])
        if dist_arr[i,j]>0.1:
            boundary[i, j:] = None
            pass

for i in range(1, mu.shape[0]-1):
    for j in range(num_bound):
        if np.isnan(boundary[i+1,j]) and np.isnan(boundary[i-1, j]):
            boundary[i,j]=None

color = colors.colorConverter.to_rgba('lightcyan', alpha=1.0)
color = list(color)
color[0] = 0.85
for i in range(int(num_bound/2)):
    art = axs.fill_betweenx(mu, boundary[:, 2*i], boundary[:, 2*i+1], visible = True, ec='k', fc=color, lw=2.0, zorder=1, where=dist_arr[:,i]<0.1)
for i in range(int(num_bound/2)):
    art = axs.fill_betweenx(mu, boundary[:, 2*i], boundary[:, 2*i+1], visible = True, ec='face', fc=color, lw=0.3, zorder=1.1, where=dist_arr[:,i]<0.1)
    #art.set_edgecolor(color)

plt.subplots_adjust(top=0.95, left=0.15, bottom=0.2, right=0.98)
axs.set_xlabel(r'$E_Z$ (meV)', size=9)
axs.set_ylabel(r'$\mu$ (meV)', size=9)

axs.set_xlim([0, 4.2])
axs.set_ylim([-4, 14])

axs.plot([0.6, 0.6], [-2, 13.2], c='r', lw=1.5, mec='k', zorder=1.2)
axs.plot([0, 3], [10.30, 10.30], c='r', lw=1.5, mec='k', zorder=1.2)
axs.plot([0, 3], [6.28, 6.28], c='r', lw=1.5, mec='k', zorder=1.2)
axs.plot([0, 3], [2.37, 2.37], c='r', lw=1.5, mec='k', zorder=1.2)
axs.tick_params(axis='x', labelsize=9)
axs.tick_params(axis='y', labelsize=9)
axs.xaxis.set_major_locator(ticker.MultipleLocator(1))

plt.savefig('FIG6', dpi=700)
plt.show()
