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
Vj_i = -20
Vj_f = 15
###################################################
dirS = 'gap_data'

gap_mu0 = np.load("%s/gapfxmu Wj = %.1f nodx = %.1f nody = %.1f Vj = %.1f alpha = %.1f delta = %.2f phi = %.3f mu_i = %.1f mu_f=%.1f gx=%.2f.npy" % (dirS, Junc_width, Nod_widthx,  Nod_widthy, Vj, alpha, delta, 0, mu_i, mu_f, gx))
gap_mupi = np.load("%s/gapfxmu Wj = %.1f nodx = %.1f nody = %.1f Vj = %.1f alpha = %.1f delta = %.2f phi = %.3f mu_i = %.1f mu_f=%.1f gx=%.2f.npy" % (dirS, Junc_width, Nod_widthx,  Nod_widthy, Vj, alpha, delta, np.pi, mu_i, mu_f, gx))
gap_gam0 = np.load("%s/gapfxgam Wj = %.1f nodx = %.1f nody = %.1f Vj = %.1f alpha = %.1f delta = %.2f phi = %.3f gam_i = %.1f gam_f = %.1f mu = %.2f.npy" % (dirS, Junc_width, Nod_widthx,  Nod_widthy, Vj,  alpha, delta, 0, gi, gf, 0))
gap_gampi = np.load("%s/gapfxgam Wj = %.1f nodx = %.1f nody = %.1f Vj = %.1f alpha = %.1f delta = %.2f phi = %.3f gam_i = %.1f gam_f = %.1f mu = %.2f.npy" % (dirS, Junc_width, Nod_widthx,  Nod_widthy, Vj,  alpha, delta, np.pi, gi, gf, 10))
gap_Vj0 = np.load("%s/gapfxVj Wj = %.1f nodx = %.1f nody = %.1f mu = %.1f alpha = %.1f delta = %.2f phi = %.3f Vj_i = %.1f Vj_f=%.1f gx=%.2f.npy" % (dirS, Junc_width, Nod_widthx,  Nod_widthy, 0, alpha, delta, 0, Vj_i, Vj_f, gx))
gap_Vjpi = np.load("%s/gapfxVj Wj = %.1f nodx = %.1f nody = %.1f mu = %.1f alpha = %.1f delta = %.2f phi = %.3f Vj_i = %.1f Vj_f=%.1f gx=%.2f.npy" % (dirS, Junc_width, Nod_widthx,  Nod_widthy, 10, alpha, delta, np.pi, Vj_i, Vj_f, gx))

gam_0 = np.load("%s/gamx Wj = %.1f nodx = %.1f nody = %.1f Vj = %.1f alpha = %.1f delta = %.2f phi = %.3f gam_i = %.1f gam_f = %.1f mu = %.2f.npy" % (dirS, Junc_width, Nod_widthx,  Nod_widthy, Vj,  alpha, delta, 0, gi, gf, 0))
gam_pi = np.load("%s/gamx Wj = %.1f nodx = %.1f nody = %.1f Vj = %.1f alpha = %.1f delta = %.2f phi = %.3f gam_i = %.1f gam_f = %.1f mu = %.2f.npy" % (dirS, Junc_width, Nod_widthx,  Nod_widthy, Vj,  alpha, delta, np.pi, gi, gf, 10))
mu_0 = np.load("%s/mu Wj = %.1f nodx = %.1f nody = %.1f Vj = %.1f alpha = %.1f delta = %.2f phi = %.3f mu_i = %.1f mu_f=%.1f gx=%.2f.npy" % (dirS, Junc_width, Nod_widthx,  Nod_widthy, Vj, alpha, delta, 0, mu_i, mu_f, gx))
mu_pi = np.load("%s/mu Wj = %.1f nodx = %.1f nody = %.1f Vj = %.1f alpha = %.1f delta = %.2f phi = %.3f mu_i = %.1f mu_f=%.1f gx=%.2f.npy" % (dirS, Junc_width, Nod_widthx,  Nod_widthy, Vj, alpha, delta, np.pi, mu_i, mu_f, gx))
Vj_0 = np.load("%s/Vj Wj = %.1f nodx = %.1f nody = %.1f mu = %.1f alpha = %.1f delta = %.2f phi = %.3f Vj_i = %.1f Vj_f=%.1f gx=%.2f.npy" % (dirS, Junc_width, Nod_widthx,  Nod_widthy, 0, alpha, delta, 0, Vj_i, Vj_f, gx))
Vj_pi = np.load("%s/Vj Wj = %.1f nodx = %.1f nody = %.1f mu = %.1f alpha = %.1f delta = %.2f phi = %.3f Vj_i = %.1f Vj_f=%.1f gx=%.2f.npy" % (dirS, Junc_width, Nod_widthx,  Nod_widthy, 10, alpha, delta, np.pi, Vj_i, Vj_f, gx))

fig, axs = plt.subplots(2, 3, sharex='col', sharey='row', gridspec_kw={'hspace':0.2, 'wspace':0.1})

plt.grid()
(ax1, ax2, ax3),(ax4, ax5, ax6) = axs
ax1.plot(gam_0, gap_gam0, c='r', lw=2) # gam for phi=0
ax4.plot(gam_pi, gap_gampi, c='r', lw=2)
ax2.plot(mu_0, gap_mu0, c='b', lw=2)
ax5.plot(mu_pi, gap_mupi, c='b', lw=2)
ax3.plot(Vj_0, gap_Vj0, c='g', lw=2)
ax6.plot(Vj_pi, gap_Vjpi, c='g', lw=2)

ax4.set_xlabel(r'$E_Z$ (meV)')
ax5.set_xlabel(r'$\mu$ (meV)')
ax6.set_xlabel(r'$V_J$ (meV)')
ax1.set_ylabel(r'$\Delta_{qp}/\Delta_{0}$')
ax4.set_ylabel(r'$\Delta_{qp}/\Delta_{0}$')
ax1.set_ylim(0,0.3)
ax2.set_ylim(0, 0.3)
ax3.set_ylim(0,0.3)
ax4.set_ylim(0,0.3)
ax5.set_ylim(0,0.3)
ax6.set_ylim(0,0.3)

plt.subplots_adjust(top=0.75, left=0.25, bottom=0.25)
ax1.grid('on')
ax2.grid('on')
ax3.grid('on')
ax4.grid('on')
ax5.grid('on')
ax6.grid('on')

plt.savefig('gapfxmu juncwidth = {} nodwidthx = {} nodwidthy = {} alpha = {} phi = {} Vj = {}.png'.format(Junc_width, Nod_widthx, Nod_widthy, alpha, phi, Vj), dpi=700)
plt.show()

sys.exit()
