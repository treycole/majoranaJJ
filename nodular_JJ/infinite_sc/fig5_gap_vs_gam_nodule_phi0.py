import sys
import time
import os
import gc

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.signal import argrelextrema
import matplotlib.lines as mlines
import matplotlib.colors as colors

import majoranaJJ.modules.SNRG as SNRG
import majoranaJJ.modules.finders as finders
import majoranaJJ.modules.constants as const
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
#########################################
#Defining Hamiltonian parameters
alpha = 200 #Spin-Orbit Coupling constant: [meV*A]
phi = 0*np.pi #SC phase difference
delta = 0.30 #Superconducting Gap: [meV]
Vj = -40 #Junction potential: [meV], for boundary plot phi=pi,0 and gap plots vs Ez and mu
gi = 0
gf = 3
mu = [7.36, 11.12]
###################################################
dirS = 'gap_data'

gap1 = np.load("%s/gapfxgam Wj = %.1f Lx = %.1f cutxT = %.1f cutyT = %.1f cutxB = %.1f cutyB = %.1f Vj = %.1f phi = %.3f gam_i = %.1f gam_f = %.1f mu = %.2f.npy" % (dirS, Junc_width, Lx*.1, cutxT_width, cutyT_width, cutxB_width, cutyB_width, Vj, phi, gi, gf, mu[0]))
gx1 = np.linspace(gi, gf, gap1.shape[0])
kx_of_gap1 = np.load("%s/kxofgapfxgam Wj = %.1f Lx = %.1f cutxT = %.1f cutyT = %.1f cutxB = %.1f cutyB = %.1f Vj = %.1f phi = %.3f gam_i = %.1f gam_f = %.1f mu = %.2f.npy" % (dirS, Junc_width, Lx*.1, cutxT_width, cutyT_width, cutxB_width, cutyB_width, Vj, phi, gi, gf, mu[0]))
gap2 = np.load("%s/gapfxgam Wj = %.1f Lx = %.1f cutxT = %.1f cutyT = %.1f cutxB = %.1f cutyB = %.1f Vj = %.1f phi = %.3f gam_i = %.1f gam_f = %.1f mu = %.2f.npy" % (dirS, Junc_width, Lx*.1, cutxT_width, cutyT_width, cutxB_width, cutyB_width, Vj,phi, gi, gf, mu[1]))
gx2 = np.linspace(gi, gf, gap2.shape[0])
kx_of_gap2 = np.load("%s/kxofgapfxgam Wj = %.1f Lx = %.1f cutxT = %.1f cutyT = %.1f cutxB = %.1f cutyB = %.1f Vj = %.1f phi = %.3f gam_i = %.1f gam_f = %.1f mu = %.2f.npy" % (dirS, Junc_width, Lx*.1, cutxT_width, cutyT_width, cutxB_width, cutyB_width, Vj, phi, gi, gf, mu[1]))
gx2 = np.linspace(gi,gf,gap2.shape[0])

top_arr1 = np.zeros(gx1.shape[0])
top_arr2 = np.zeros(gx2.shape[0])
local_min_idx1 = np.array(argrelextrema(gap1, np.less)[0])
local_min_idx2 = np.array(argrelextrema(gap2, np.less)[0])

num1 = 1
lower_bound1 = 0
top_arr1[lower_bound1:] = num1
num2 = 1
lower_bound2 = 0
top_arr2[lower_bound2:] = num2

for i in range(local_min_idx1.shape[0]):
    lower_bound1 = local_min_idx1[i]
    if gap1[local_min_idx1[i]]/delta < 0.02 and (Lx*kx_of_gap1[local_min_idx1[i]] <= 0.1 or abs(Lx*kx_of_gap1[local_min_idx1[i]] - np.pi) < .15):
        num1=num1*-1
    if num1==1:
        top_arr1[lower_bound1+1:] = num1
    if num1==-1:
        top_arr1[lower_bound1:] = num1
for i in range(local_min_idx2.shape[0]):
    lower_bound2 = local_min_idx2[i]
    if gap2[local_min_idx2[i]]/delta < 0.02 and (Lx*kx_of_gap2[local_min_idx2[i]] <= 0.1 or abs(Lx*kx_of_gap2[local_min_idx2[i]] - np.pi) < .15):
        num2=num2*-1
    if num2==1:
        top_arr2[lower_bound2+1:] = num2
    if num2==-1:
        top_arr2[lower_bound2:] = num2

fig, ax = plt.subplots(2, 1, gridspec_kw={'hspace':0.1}, sharex=True)

color = colors.colorConverter.to_rgba('lightblue', alpha=1.0)
art = ax[1].fill_between(gx1, gap1/delta, visible=True, alpha=1, color=color, where=top_arr1[:]<0)
art.set_edgecolor('b')
art = ax[0].fill_between(gx2, gap2/delta, visible=True, alpha=1, color=color, where=top_arr2[:]<0)
art.set_edgecolor('b')

ax[1].plot(gx1, gap1/delta, c='b', lw=1.50)
ax[0].plot(gx2, gap2/delta, c='b', lw=1.50)

ax[1].set_xlabel(r'$E_Z$ (meV)', size=9)
ax[0].set_ylabel(r'$\Delta_{qp}/\Delta_{0}$', size=9)
ax[1].set_ylabel(r'$\Delta_{qp}/\Delta_{0}$', size=9)

ax[0].set_xticks([0, 1, 2, 3])
ax[0].set_yticks([0, 0.5, 1.0])
ax[1].set_yticks([0, 0.5, 1.0])

ax[0].text(gi+0.7*(gf-gi+0.07*(gf-gi)*2), 0.75,'(a)', fontdict=None, fontsize=9)
ax[1].text(gi+0.7*(gf-gi+0.07*(gf-gi)*2), 0.75,'(b)', fontdict=None, fontsize=9)

ax[0].set_xlim(-0.1, 3.1)
ax[1].set_xlim(-0.1, 3.1)
ax[0].set_ylim(-0.1, 1.1)
ax[1].set_ylim(-0.1, 1.1)

ax[0].tick_params(axis='x', labelsize=8)
ax[1].tick_params(axis='x', labelsize=8)
ax[0].tick_params(axis='y', labelsize=8)
ax[1].tick_params(axis='y', labelsize=8)

ax[0].grid('on')
ax[1].grid('on')

plt.subplots_adjust(bottom=0.15, left=0.2, top=0.95, right=0.98)
plt.savefig('FIG5', dpi=700)
plt.show()
sys.exit()
