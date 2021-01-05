import sys
import time
import os
import gc

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import majoranaJJ.modules.self_energy_nodule as slfNRG
import majoranaJJ.modules.plots as plots #plotting functions
###################################################
#Defining System
ax = 50 #lattice spacing in x-direction: [A]
ay = 50 #lattice spacing in y-direction: [A]
Nx = 3
Wj = 2000 #Junction width: [A]
cutx = 0 #width of nodule
cuty = 0 #height of nodule

Lx = Nx*ax
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
Vj = -5 #Junction potential: [meV]

mui = 0
muf = 6
res = 0.02
steps_mu = int((muf-mui)/(res)) + 1
MU = np.linspace(mui, muf, steps_mu) #meV

gi = 0
gf = 1.25
res = 0.01
steps_gam = int((gf - gi)/(res)) + 1
GX = np.linspace(gi, gf, steps_gam)

k = 4
##################################################
dirS = 'gap_data'
if not os.path.exists(dirS):
    os.makedirs(dirS)
try:
    PLOT = str(sys.argv[1])
except:
    PLOT = 'F'
if PLOT != 'P':
    gap=np.zeros((GX.shape[0], MU.shape[0]))
    for i in range(GX.shape[0]):
        for j in range(MU.shape[0]):
            print(GX.shape[0]-i, MU.shape[0]-j)
            gap[i,j] = slfNRG.gap(Wj=Wj, Lx=Lx, nodx=cutx, nody=cuty, ax=ax, ay=ay, gam=GX[i], mu=MU[j], Vj=Vj, alpha=alpha, delta=delta, phi=phi, muf=6)[0]

    np.save("%s/gap Wj = %.1f nodx = %.1f nody = %.1f Vj = %.1f Vsc = %.1f alpha = %.1f delta = %.2f phi = %.3f.npy" % (dirS, Junc_width, Nod_widthx,  Nod_widthy, Vj, Vsc, alpha, delta, phi), gap)
    gc.collect()
    sys.exit()
else:
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
