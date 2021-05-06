import sys
import time
import os
import gc

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import majoranaJJ.modules.SNRG as SNRG
import majoranaJJ.modules.finders as finders
import majoranaJJ.modules.constants as const
import majoranaJJ.modules.fig_params as params
import matplotlib.lines as mlines
from scipy.signal import argrelextrema
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
cutyT = 2*cuty
cutyB = 0
Lx = Nx*ax #Angstrom
Junc_width = Wj*.1 #nm
cutxT_width = cutxT*ax*.1 #nm
cutyT_width = cutyT*ax*.1 #nm
cutxB_width = cutxB*ax*.1 #nm
cutyB_width = cutyB*ax*.1 #nm
#########################################
#Defining Hamiltonian parameters
alpha = 200 #Spin-Orbit Coupling constant: [meV*A]
delta = 0.30 #Superconducting Gap: [meV]

Vj = 0 #Junction potential: [meV], for boundary plot phi=pi,0 and gap plots vs Ez and mu
gx = 1 #mev, for all gap plots

mu_i = -2
mu_f = 12
gi = 0
gf = 3
Vj_i = -11
Vj_f = 11
###################################################
dirS = 'gap_data'

gap_mu0 = np.load("%s/gapfxmu Wj = %.1f Lx = %.1f cutxT = %.1f cutyT = %.1f cutxB = %.1f cutyB = %.1f Vj = %.1f phi = %.2f mu_i = %.1f mu_f = %.1f gx = %.2f.npy" % (dirS, Junc_width, Lx*.1, cutxT_width, cutyT_width, cutxB_width, cutyB_width, Vj, 0, mu_i, mu_f, gx))
gap_mupi = np.load("%s/gapfxmu Wj = %.1f Lx = %.1f cutxT = %.1f cutyT = %.1f cutxB = %.1f cutyB = %.1f Vj = %.1f phi = %.2f mu_i = %.1f mu_f = %.1f gx = %.2f.npy" % (dirS, Junc_width, Lx*.1, cutxT_width, cutyT_width, cutxB_width, cutyB_width, Vj, np.pi, mu_i, mu_f, gx))
gap_gam0 = np.load("%s/gapfxgam Wj = %.1f Lx = %.1f cutxT = %.1f cutyT = %.1f cutxB = %.1f cutyB = %.1f Vj = %.1f phi = %.3f gam_i = %.1f gam_f = %.1f mu = %.2f.npy" % (dirS, Junc_width, Lx*.1, cutxT_width, cutyT_width, cutxB_width, cutyB_width, Vj, 0, gi, gf, 0))
gap_gampi = np.load("%s/gapfxgam Wj = %.1f Lx = %.1f cutxT = %.1f cutyT = %.1f cutxB = %.1f cutyB = %.1f Vj = %.1f phi = %.3f gam_i = %.1f gam_f = %.1f mu = %.2f.npy" % (dirS, Junc_width, Lx*.1, cutxT_width, cutyT_width, cutxB_width, cutyB_width, Vj, np.pi, gi, gf, 10))
gap_Vjpi = np.load("%s/gapfxVj Wj = %.1f Lx = %.1f cutxT = %.1f cutyT = %.1f cutxB = %.1f cutyB = %.1f mu = %.1f phi = %.3f Vj_i = %.1f Vj_f = %.1f gx = %.2f.npy" % (dirS, Junc_width, Lx*.1, cutxT_width, cutyT_width, cutxB_width, cutyB_width, 10, np.pi, Vj_i, Vj_f, gx))

KOG_mu0 = np.load("%s/kxofgapfxmu Wj = %.1f Lx = %.1f cutxT = %.1f cutyT = %.1f cutxB = %.1f cutyB = %.1f Vj = %.1f phi = %.2f mu_i = %.1f mu_f = %.1f gx = %.2f.npy" % (dirS, Junc_width, Lx*.1, cutxT_width, cutyT_width, cutxB_width, cutyB_width, Vj, 0, mu_i, mu_f, gx))
KOG_mupi = np.load("%s/kxofgapfxmu Wj = %.1f Lx = %.1f cutxT = %.1f cutyT = %.1f cutxB = %.1f cutyB = %.1f Vj = %.1f phi = %.2f mu_i = %.1f mu_f = %.1f gx = %.2f.npy" % (dirS, Junc_width, Lx*.1, cutxT_width, cutyT_width, cutxB_width, cutyB_width, Vj, np.pi, mu_i, mu_f, gx))
KOG_gam0 = np.load("%s/kxofgapfxgam Wj = %.1f Lx = %.1f cutxT = %.1f cutyT = %.1f cutxB = %.1f cutyB = %.1f Vj = %.1f phi = %.3f gam_i = %.1f gam_f = %.1f mu = %.2f.npy" % (dirS, Junc_width, Lx*.1, cutxT_width, cutyT_width, cutxB_width, cutyB_width, Vj, 0, gi, gf, 0))
KOG_gampi = np.load("%s/kxofgapfxgam Wj = %.1f Lx = %.1f cutxT = %.1f cutyT = %.1f cutxB = %.1f cutyB = %.1f Vj = %.1f phi = %.3f gam_i = %.1f gam_f = %.1f mu = %.2f.npy" % (dirS, Junc_width, Lx*.1, cutxT_width, cutyT_width, cutxB_width, cutyB_width, Vj, np.pi, gi, gf, 10))

mu_0 = np.linspace(mu_i, mu_f, gap_mupi.shape[0])
mu_pi = np.linspace(mu_i, mu_f, gap_mu0.shape[0])
gam_0 = np.linspace(gi, gf, gap_gam0.shape[0])
gam_pi = np.linspace(gi, gf, gap_gampi.shape[0])
Vj_pi = np.linspace(Vj_i, Vj_f, gap_Vjpi.shape[0])

local_min_idx = np.array(argrelextrema(gap_mu0, np.less)[0])
for i in range(local_min_idx.shape[0]):
    lower_bound = local_min_idx[i]
    if gap_mu0[local_min_idx[i]]/delta < 0.04 and (Lx*KOG_mu0[local_min_idx[i]] <= 0.2 or abs(Lx*KOG_mu0[local_min_idx[i]] - np.pi) <= .2):
        gap_mu0[local_min_idx[i]] = 0
local_min_idx = np.array(argrelextrema(gap_mupi, np.less)[0])
for i in range(local_min_idx.shape[0]):
    lower_bound = local_min_idx[i]
    if gap_mupi[local_min_idx[i]]/delta < 0.04 and (Lx*KOG_mupi[local_min_idx[i]] <= 0.2 or abs(Lx*KOG_mupi[local_min_idx[i]] - np.pi) <= .2):
        gap_mupi[local_min_idx[i]] = 0
local_min_idx = np.array(argrelextrema(gap_gam0, np.less)[0])
for i in range(local_min_idx.shape[0]):
    lower_bound = local_min_idx[i]
    if gap_gam0[local_min_idx[i]]/delta < 0.04 and (Lx*KOG_gam0[local_min_idx[i]] <= 0.2 or abs(Lx*KOG_gam0[local_min_idx[i]] - np.pi) <= .2):
        gap_gam0[local_min_idx[i]] = 0
local_min_idx = np.array(argrelextrema(gap_gampi, np.less)[0])
for i in range(local_min_idx.shape[0]):
    lower_bound = local_min_idx[i]
    if gap_gampi[local_min_idx[i]]/delta < 0.04 and (Lx*KOG_gampi[local_min_idx[i]] <= 0.2 or abs(Lx*KOG_gampi[local_min_idx[i]] - np.pi) <= .2):
        gap_gampi[local_min_idx[i]] = 0

fig, axs = plt.subplots(3, 1, gridspec_kw={'hspace':0.80})#, sharey='row')#, sharex='col', sharey='row', gridspec_kw={'hspace':0.5, 'wspace':0.1})

plt.grid()
(axmu, axgam, axvj) = axs

axmu.set_xticks([0, 4, 8, 12])
axgam.set_xticks([0, 1, 2, 3])
axvj.set_xticks([-10, -5, 0, 5, 10])

axmu.set_yticks([0, 0.15, 0.3])
axgam.set_yticks([0, 0.15, 0.3])
axvj.set_yticks([0, 0.15, 0.3])

axmu.text(mu_i+0.7*(mu_f-mu_i+0.06*(mu_f-mu_i)*2), 0.20,'(a)', fontdict=None, fontsize=9)
axgam.text(gi+0.7*(gf-gi+0.06*(gf-gi)*2), 0.20,'(b)', fontdict=None, fontsize=9)
axvj.text(Vj_i+0.7*(Vj_f-Vj_i+0.06*(Vj_f-Vj_i)*2), 0.20,'(c)', fontdict=None, fontsize=9)

axmu.plot(mu_0, gap_mu0/delta, c='b', lw=2.0)
axmu.plot(mu_pi, gap_mupi/delta, c='r', lw=1.7)
axgam.plot(gam_0, gap_gam0/delta, c='b', lw=2.0) # gam for phi=0
axgam.plot(gam_pi, gap_gampi/delta, c='r', lw=2.0)
axvj.plot(Vj_pi, gap_Vjpi/delta, c='r', lw=2)
red_circ = mlines.Line2D([],[], c='r', marker='o', mec='k')
axvj.legend(handles=[red_circ])

axmu.set_xlabel(r'$\mu$ (meV)', size=9, labelpad=-0.5)
axgam.set_xlabel(r'$E_Z$ (meV)', size=9, labelpad=-0.5)
axvj.set_xlabel(r'$V_J$ (meV)', size=9, labelpad=1.5)
axmu.set_ylabel(r'$\Delta_{qp}/\Delta_{0}$', size=9)
axgam.set_ylabel(r'$\Delta_{qp}/\Delta_{0}$', size=9)
axvj.set_ylabel(r'$\Delta_{qp}/\Delta_{0}$', size=9)

axmu.set_xmargin(m=0.06)
axgam.set_xmargin(m=0.06)
axvj.set_xmargin(m=0.06)
axmu.set_ylim(-0.03, 0.33)
axgam.set_ylim(-0.03, 0.33)
axvj.set_ylim(-0.03, 0.33)

axmu.tick_params(axis='x', labelsize=8)
axgam.tick_params(axis='x', labelsize=8)
axvj.tick_params(axis='x', labelsize=8)
axmu.tick_params(axis='y', labelsize=8)
axgam.tick_params(axis='y', labelsize=8)
axvj.tick_params(axis='y', labelsize=8)

axmu.grid('on')
axgam.grid('on')
axvj.grid('on')


plt.subplots_adjust(bottom=0.15, left=0.2, top=0.95, right=0.98)
plt.savefig('FIG2', dpi=700)
plt.show()
