import sys
import time
import os
import gc

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import majoranaJJ.modules.self_energy as slfNRG
import majoranaJJ.modules.plots as plots #plotting functions
###################################################
#Defining System
ax = 50 #lattice spacing in x-direction: [A]
ay = 50 #lattice spacing in y-direction: [A]
Wj = 2000 #Junction width: [A]
cutx = 0 #width of nodule
cuty = 0 #height of nodule

Junc_width = Wj*.1 #nm
Nod_widthx = cutx*.1 #nm
Nod_widthy = cuty*.1 #nm
print("Nodule Width in x-direction = ", Nod_widthx, "(nm)")
print("Nodule Width in y-direction = ", Nod_widthy, "(nm)")
print("Junction Width = ", Junc_width, "(nm)")
###################################################
#Defining Hamiltonian parameters
alpha = 200 #Spin-Orbit Coupling constant: [meV*A]
phi = np.pi #SC phase difference
delta = 1 #Superconducting Gap: [meV]
Vsc = 0 #SC potential: [meV]
Vj = 5 #Junction potential: [meV]
mu = 0
gx = 0.25

mui = 0
muf = 6
res = 0.01
steps_mu = int((muf-mui)/(res)) + 1
MU = np.linspace(mui, muf, steps_mu) #meV

gi = 0
gf = 5.0
res = 0.01
steps_gam = int((gf - gi)/(res)) + 1
GX = np.linspace(gi, gf, steps_gam)

k = 4
gx_fxd = True
mu_fxd = False
full_range = False
##################################################
dirS = 'gap_data'
if not os.path.exists(dirS):
    os.makedirs(dirS)
try:
    PLOT = str(sys.argv[1])
except:
    PLOT = 'F'
if PLOT != 'P':
    if full_range:
        full_range_gap=np.zeros((GX.shape[0], MU.shape[0]))
        for i in range(GX.shape[0]):
            for j in range(MU.shape[0]):
                full_range_gap[i,j] = slfNRG.gap(ay, GX[i], MU[j], Vj, Wj, alpha, delta, phi)[0]

        np.save("%s/gap Wj = %.1f nodx = %.1f nody = %.1f Vj = %.1f Vsc = %.1f alpha = %.1f delta = %.2f phi = %.3f.npy" % (dirS, Junc_width, Nod_widthx,  Nod_widthy, Vj, Vsc, alpha, delta, phi), full_range_gap)
        gc.collect()
    if gx_fxd:
        gx_fxd_gap = np.zeros(MU.shape[0])
        for i in range(MU.shape[0]):
            gx_fxd_gap[i] = slfNRG.gap(ay, gx, MU[i], Vj, Wj, alpha, delta, phi)[0]

        np.save("%s/gapgxfxd Wj = %.1f nodx = %.1f nody = %.1f Vj = %.1f Vsc = %.1f alpha = %.1f delta = %.2f phi = %.3f.npy" % (dirS, Junc_width, Nod_widthx,  Nod_widthy, Vj, Vsc, alpha, delta, phi), gx_fxd_gap)
        gc.collect()
    if mu_fxd:
        mu_fxd_gap = np.zeros(GX.shape[0])
        for i in range(GX.shape[0]):
            mu_fxd_gap[i] = slfNRG.gap(ay, GX[i], mu, Vj, Wj, alpha, delta, phi)[0]

        np.save("%s/gapmufxd Wj = %.1f nodx = %.1f nody = %.1f Vj = %.1f Vsc = %.1f alpha = %.1f delta = %.2f phi = %.3f.npy" % (dirS, Junc_width, Nod_widthx,  Nod_widthy, Vj, Vsc, alpha, delta, phi), mu_fxd_gap)
        gc.collect()
    sys.exit()
else:
    if full_range:
        gap_full = np.load("%s/gap Wj = %.1f nodx = %.1f nody = %.1f Vj = %.1f Vsc = %.1f alpha = %.1f delta = %.2f phi = %.3f.npy" % (dirS, Junc_width, Nod_widthx,  Nod_widthy, Vj, Vsc, alpha, delta, phi))

        gap = gap/delta
        gap = np.transpose(gap)

        plt.contourf(gx, mu, gap, 500, vmin = 0, vmax = max(gap.flatten()), cmap = 'hot')
        cbar = plt.colorbar()
        cbar.set_label(r'$E_{gap}/\Delta$')

        plt.xlabel(r'$\Gamma_x$ (meV)')
        plt.ylabel(r'$\mu$ (meV)')

        plt.xlim(gi, gf)
        title = r"$W_j$ = %.1f nm, $nodule_x$ = %.1f nm, $nodule_y$ = %.1f nm, $V_j$ = %.1f meV, $V_{SC}$ = %.1f meV, $\phi$ = %.2f " % (Junc_width, Nod_widthx, Nod_widthy, Vj, Vsc, phi)
        plt.title(title, loc = 'center', wrap = True)
        plt.subplots_adjust(top=0.85)
        plt.savefig('gap juncwidth = {} nodwidthx = {} nodwidthy = {} phi = {} Vj = {} Vsc = {}.png'.format(Junc_width, Nod_widthx, Nod_widthy, delta, alpha, phi, Vj, Vsc))
        plt.show()

    if gx_fxd:
        gapgxfxd = np.load("%s/gapgxfxd Wj = %.1f nodx = %.1f nody = %.1f Vj = %.1f Vsc = %.1f alpha = %.1f delta = %.2f phi = %.3f.npy" % (dirS, Junc_width, Nod_widthx,  Nod_widthy, Vj, Vsc, alpha, delta, phi))
        plt.plot(MU, gapgxfxd)
        plt.xlabel(r'$\mu$ (meV)')
        plt.ylabel(r'$E_{gap}/\Delta$')
        plt.ylim(0, 0.05)
        plt.show()
    if mu_fxd:
        gapmufxd = np.load("%s/gapmufxd Wj = %.1f nodx = %.1f nody = %.1f Vj = %.1f Vsc = %.1f alpha = %.1f delta = %.2f phi = %.3f mu = %.3f.npy" % (dirS, Junc_width, Nod_widthx,  Nod_widthy, Vj, Vsc, alpha, delta, phi, mu))
        plt.plot(GX, gapgxfxd)
        plt.xlim(gi, gf)
        plt.xlabel(r'$\Gamma_x$ (meV)')
        plt.ylabel(r'$E_{gap}/\Delta$')
        plt.show()
    sys.exit()
