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
Vj = 10 #junction potential: [meV]
mu_i = 5
mu_f = 25
gi = 0
gf = 5
num_bound = 4
###################################################
#phase diagram mu vs gam
dirS = 'boundary_data'

boundary = np.load("%s/boundary Lx = %.1f Wj = %.1f cutxT = %.1f cutyT = %.1f cutxB = %.1f cutyB = %.1f Vj = %.1f phi = %.3f mu_i = %.1f mu_f = %.1f.npy" % (dirS, Lx*.1, Junc_width, cutxT_width, cutyT_width, cutxB_width, cutyB_width, Vj, phi, mu_i, mu_f))
mu = np.linspace(mu_i, mu_f, boundary.shape[0])

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

fig, axs = plt.subplots(1, gridspec_kw={'hspace':0.1, 'wspace':0.1})
for i in range(boundary.shape[1]):
    axs.plot(boundary[:, i], mu, c='k', linewidth=1.7)

color = colors.colorConverter.to_rgba('lightblue', alpha=1.0)
axs.fill_betweenx(mu, boundary[:, 0], boundary[:, 1], visible=True, alpha=1, color=color)

axs.set_xlabel(r'$E_Z$ (meV)', size=9)
axs.set_ylabel(r'$\mu$ (mev)', size=9)

axs.set_yticks([10, 15, 20])

axs.set_xlim([0, 4.3])
axs.set_ymargin(m=0.05)
axs.set_ylim([8, 22])

axs.tick_params(axis='x', labelsize=9)
axs.tick_params(axis='y', labelsize=9)
axs.xaxis.set_major_locator(ticker.MultipleLocator(1))

plt.subplots_adjust(top=0.95, left=0.15, bottom=0.15, right=0.98)
plt.savefig('FIG17', dpi=700)
plt.show()
