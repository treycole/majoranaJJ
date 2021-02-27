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
###################################################
#Defining System
ax = 50 #lattice spacing in x-direction: [A]
ay = 50 #lattice spacing in y-direction: [A]
Nx = 3 #Number of lattice sites along x-direction
Wj = 1000 #Junction region [A]
nodx = 0 #width of nodule
nody = 0 #height of nodule
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

gap_mu0 = np.load("%s/gapfxmu Wj = %.1f nodx = %.1f nody = %.1f Vj = %.1f alpha = %.1f delta = %.2f phi = %.3f mu_i = %.1f mu_f=%.1f gx=%.2f.npy" % (dirS, Junc_width, Nod_widthx,  Nod_widthy, Vj, alpha, delta, 0, mu_i, mu_f, gx))
gap_mupi = np.load("%s/gapfxmu Wj = %.1f nodx = %.1f nody = %.1f Vj = %.1f alpha = %.1f delta = %.2f phi = %.3f mu_i = %.1f mu_f=%.1f gx=%.2f.npy" % (dirS, Junc_width, Nod_widthx,  Nod_widthy, Vj, alpha, delta, np.pi, mu_i, mu_f, gx))
gap_gam0 = np.load("%s/gapfxgam Wj = %.1f nodx = %.1f nody = %.1f Vj = %.1f alpha = %.1f delta = %.2f phi = %.3f gam_i = %.1f gam_f = %.1f mu = %.2f.npy" % (dirS, Junc_width, Nod_widthx,  Nod_widthy, Vj,  alpha, delta, 0, gi, gf, 0))
gap_gampi = np.load("%s/gapfxgam Wj = %.1f nodx = %.1f nody = %.1f Vj = %.1f alpha = %.1f delta = %.2f phi = %.3f gam_i = %.1f gam_f = %.1f mu = %.2f.npy" % (dirS, Junc_width, Nod_widthx,  Nod_widthy, Vj,  alpha, delta, np.pi, gi, gf, 10))
gap_Vjpi = np.load("%s/gapfxVj Wj = %.1f nodx = %.1f nody = %.1f mu = %.1f alpha = %.1f delta = %.2f phi = %.3f Vj_i = %.1f Vj_f = %.1f gx = %.2f.npy" % (dirS, Junc_width, Nod_widthx,  Nod_widthy, 10, alpha, delta, np.pi, Vj_i, Vj_f, gx))
#gap_Vj0 = np.load("%s/gapfxVj Wj = %.1f nodx = %.1f nody = %.1f mu = %.1f alpha = %.1f delta = %.2f phi = %.3f Vj_i = %.1f Vj_f=%.1f gx=%.2f.npy" % (dirS, Junc_width, Nod_widthx,  Nod_widthy, 10, alpha, delta, np.pi, Vj_i, Vj_f, gx))

gam_0 = np.load("%s/gamx Wj = %.1f nodx = %.1f nody = %.1f Vj = %.1f alpha = %.1f delta = %.2f phi = %.3f gam_i = %.1f gam_f = %.1f mu = %.2f.npy" % (dirS, Junc_width, Nod_widthx,  Nod_widthy, Vj,  alpha, delta, 0, gi, gf, 0))
gam_pi = np.load("%s/gamx Wj = %.1f nodx = %.1f nody = %.1f Vj = %.1f alpha = %.1f delta = %.2f phi = %.3f gam_i = %.1f gam_f = %.1f mu = %.2f.npy" % (dirS, Junc_width, Nod_widthx,  Nod_widthy, Vj,  alpha, delta, np.pi, gi, gf, 10))
mu_0 = np.load("%s/mu Wj = %.1f nodx = %.1f nody = %.1f Vj = %.1f alpha = %.1f delta = %.2f phi = %.3f mu_i = %.1f mu_f=%.1f gx=%.2f.npy" % (dirS, Junc_width, Nod_widthx,  Nod_widthy, Vj, alpha, delta, 0, mu_i, mu_f, gx))
mu_pi = np.load("%s/mu Wj = %.1f nodx = %.1f nody = %.1f Vj = %.1f alpha = %.1f delta = %.2f phi = %.3f mu_i = %.1f mu_f=%.1f gx=%.2f.npy" % (dirS, Junc_width, Nod_widthx,  Nod_widthy, Vj, alpha, delta, np.pi, mu_i, mu_f, gx))
Vj_pi = np.load("%s/Vj Wj = %.1f nodx = %.1f nody = %.1f mu = %.1f alpha = %.1f delta = %.2f phi = %.3f Vj_i = %.1f Vj_f = %.1f gx = %.2f.npy" % (dirS, Junc_width, Nod_widthx,  Nod_widthy, 10, alpha, delta, np.pi, Vj_i, Vj_f, gx))
#Vj_pi = np.load("%s/Vj Wj = %.1f nodx = %.1f nody = %.1f mu = %.1f alpha = %.1f delta = %.2f phi = %.3f Vj_i = %.1f Vj_f=%.1f gx=%.2f.npy" % (dirS, Junc_width, Nod_widthx,  Nod_widthy, 10, alpha, delta, np.pi, Vj_i, Vj_f, gx))

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

axmu.set_xlim(left=None, right=12+(mu_f-mu_i)*.06)
#axvj.set_xlim(left=-11, right=11)
#axgam.set_xlim(left=None, right=2.8)
#axmu.set_xmargin(m=0.06)
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

plt.subplots_adjust(bottom=0.2, left=0.2, top=0.95, right=0.98)
axmu.grid('on')
axgam.grid('on')
axvj.grid('on')

#plt.savefig('gapfxmu juncwidth = {} nodwidthx = {} nodwidthy = {} alpha = {} phi = {} Vj = {}.png'.format(Junc_width, Nod_widthx, Nod_widthy, alpha, phi, Vj), dpi=700)
plt.savefig('FIG2', dpi=700)
plt.show()

sys.exit()
