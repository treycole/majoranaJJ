import sys
import os
import gc

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
from matplotlib import ticker
import matplotlib.colors as colors

import majoranaJJ.modules.plots as plots #plotting functions
import majoranaJJ.modules.finders as fndrs
import majoranaJJ.modules.SNRG as SNRG
import majoranaJJ.modules.fig_params as params
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
phi = 0*np.pi #SC phase difference
delta = 0.3 #Superconducting Gap: [meV]
Vj = -40 #junction potential: [meV]
gx = 1

mu_iB = -5
mu_fB = 15
mu_iG = -2
mu_fG = 12.8
resB = 0.005
resG = 0.01

gi = 0
gf = 5
num_bound = 10
###################################################
#phase diagram mu vs gam
dirS1 = 'boundary_data'
dirS2 = 'gap_data'

boundary = np.load("%s/boundary Lx = %.1f Wj = %.1f cutxT = %.1f cutyT = %.1f cutxB = %.1f cutyB = %.1f Vj = %.1f phi = %.3f mu_i = %.1f mu_f = %.1f.npy" % (dirS1, Lx*.1, Junc_width, cutxT_width,  cutyT_width, cutxB_width, cutyB_width, Vj, phi, mu_iB, mu_fB))
gap = np.load("%s/gapfxmu Wj = %.1f Lx = %.1f cutxT = %.1f cutyT = %.1f cutxB = %.1f cutyB = %.1f Vj = %.1f phi = %.2f mu_i = %.1f mu_f = %.1f gx = %.2f.npy" % (dirS2, Junc_width, Lx*.1, cutxT_width,  cutyT_width, cutxB_width, cutyB_width, Vj, phi, mu_iG, mu_fG, gx))
kx_of_gap = np.load("%s/kxofgapfxmu Wj = %.1f Lx = %.1f cutxT = %.1f cutyT = %.1f cutxB = %.1f cutyB = %.1f Vj = %.1f phi = %.2f mu_i = %.1f mu_f = %.1f gx = %.2f.npy" % (dirS2, Junc_width, Lx*.1, cutxT_width,  cutyT_width, cutxB_width, cutyB_width, Vj, phi, mu_iG, mu_fG, gx))
muG = np.linspace(mu_iG, mu_fG, gap.shape[0])
muB = np.linspace(mu_iB, mu_fB, boundary.shape[0])

top_arr = np.zeros(muG.shape[0])
num = 1
local_min_idx = np.array(argrelextrema(gap, np.less)[0])
lower_bound = 0
top_arr[lower_bound:] = num
for i in range(local_min_idx.shape[0]):
    lower_bound = local_min_idx[i]
    if gap[local_min_idx[i]]/delta < 0.025 and (Lx*kx_of_gap[local_min_idx[i]] <= 0.2 or abs(Lx*kx_of_gap[local_min_idx[i]]-np.pi) < .14):
        num=num*-1
    if num==1:
        top_arr[lower_bound+1:] = num
    if num==-1:
        top_arr[lower_bound:] = num

for i in range(muB.shape[0]-1):
    for j in range(int(boundary.shape[1]/2)):
        if np.isnan(boundary[i,2*j+1]) and not np.isnan(boundary[i,2*j]):
            boundary[i,2*j+1] = 5
            break

dist_arr = np.zeros((muB.shape[0], num_bound))
for i in range(int(muB.shape[0])-1):
    for j in range(num_bound-1):
        if np.isnan(boundary[i, j]) or np.isnan(boundary[i+1, j]):
            dist_arr[i,j] = 100000
        else:
            dist_arr[i,j] = abs(boundary[i, j] - boundary[i+1, j])
        if dist_arr[i,j]>0.1:
            boundary[i, j:] = None
            pass

for i in range(1, muB.shape[0]-1):
    for j in range(num_bound):
        if np.isnan(boundary[i+1,j]) and np.isnan(boundary[i-1, j]):
            boundary[i,j]=None

fig, axs = plt.subplots(1, 2, gridspec_kw={'hspace':0.1, 'wspace':0.055}, sharey=True)
color = colors.colorConverter.to_rgba('lightblue', alpha=1.0)
#color = list(color)
#color[0] = 0.85
for i in range(int(num_bound/2)):
    art = axs[0].fill_betweenx(muB, boundary[:, 2*i], boundary[:, 2*i+1], visible = True, ec='k', fc=color, lw=4, zorder=1, where=dist_arr[:,i]<0.1)
for i in range(int(num_bound/2)):
    art = axs[0].fill_betweenx(muB, boundary[:, 2*i], boundary[:, 2*i+1], visible = True, ec='face', fc=color, lw=1, zorder=1.2, where=dist_arr[:,i]<0.1)

#axs[1].plot(gap/delta, muG, c='b', linewidth=1.0, zorder=1)
art = axs[1].fill_betweenx(muG, gap/delta, visible=True, alpha=1, color=color, where=top_arr[:]<0, lw=0.8)
art.set_edgecolor('b')

axs[0].plot([1,1], [-2,12], c='b', lw=1.0, zorder=3)

axs[0].set_yticks([0, 5, 10, 15, 20])
axs[0].set_xticks([0, 0.5, 1])
axs[1].set_xticks([0, 0.25, 0.5])

axs[0].set_xlim([0, 1.10])
axs[0].set_ylim([-2.2, 12.2])
axs[1].set_xlim([0, 0.52])

axs[0].tick_params(axis='x', labelsize=9)
axs[1].tick_params(axis='x', labelsize=9)
axs[0].tick_params(axis='y', labelsize=9)
axs[1].tick_params(axis='y', labelsize=9, length=0)

axs[0].set_xlabel(r'$E_Z$ (meV)', size=9, labelpad=1)
axs[0].set_ylabel(r'$\mu$ (meV)', size=9, labelpad=-2)
axs[1].set_xlabel(r'$\Delta_{top}/\Delta_{0}$', size=9, labelpad=1)

for ax in fig.get_axes():
    ax.label_outer()

axs[1].text(0.15, 9, '(b)', fontdict=None, size=9)
axs[0].text(0.15, 9, '(a)', fontdict=None, size=9)

axs[0].grid(True, zorder=2.5)
axs[1].grid(True, zorder=2.5)

f = lambda x,pos:str(x).rstrip('0').rstrip('.')
axs[1].xaxis.set_major_formatter(ticker.FuncFormatter(f))
#axs[1].xaxis.set_major_locator(ticker.MultipleLocator(1))

plt.subplots_adjust(top=0.95, left=0.15, bottom=0.15, right=0.98)
plt.savefig('FIG4', dpi=700)
plt.show()
