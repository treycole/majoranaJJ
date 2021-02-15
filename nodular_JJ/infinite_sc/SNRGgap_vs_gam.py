import sys
import time
import os
import gc

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import majoranaJJ.modules.SNRG as SNRG
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
delta = 0.3 #Superconducting Gap: [meV]
Vsc = 0 #SC potential: [meV]
Vj = 0 #Junction potential: [meV]
mu = 10 #meV

gi = 0
gf = 3
res = 0.1
steps_gam = int((gf - gi)/(0.5*res)) + 1
gx = np.linspace(gi, gf, steps_gam)

k = 4
gap_gam = np.zeros(gx.shape[0])
###################################################
dirS = 'gap_data'
if not os.path.exists(dirS):
    os.makedirs(dirS)
try:
    PLOT = str(sys.argv[1])
except:
    PLOT = 'F'
if PLOT != 'P':
    np.save("%s/gamx Wj = %.1f Lx = %.1f nodx = %.1f nody = %.1f Vj = %.1f alpha = %.1f delta = %.2f phi = %.3f gam_i = %.1f gam_f = %.1f mu = %.2f.npy" % (dirS, Junc_width, Lx, Nod_widthx,  Nod_widthy, Vj,  alpha, delta, phi, gi, gf, mu), gx)
    for i in range(gx.shape[0]):
        print(gx.shape[0]-i)
        gap_gam[i] = SNRG.gap(Wj=Wj, Lx=Lx, nodx=nodx, nody=nody, ax=ax, ay=ay, gam=gx[i], mu=mu, Vj=Vj, alpha=alpha, delta=delta, phi=phi, targ_steps=5000, n_avg=8)[0]

    np.save("%s/gapfxgam Wj = %.1f Lx = %.1f nodx = %.1f nody = %.1f Vj = %.1f alpha = %.1f delta = %.2f phi = %.3f gam_i = %.1f gam_f = %.1f mu = %.2f.npy" % (dirS, Junc_width, Lx, Nod_widthx,  Nod_widthy, Vj,  alpha, delta, phi, gi, gf, mu), gap_gam)
    gc.collect()

    sys.exit()
else:
    gap = np.load("%s/gapfxgam Wj = %.1f Lx = %.1f nodx = %.1f nody = %.1f Vj = %.1f alpha = %.1f delta = %.2f phi = %.3f gam_i = %.1f gam_f = %.1f mu = %.2f.npy" % (dirS, Junc_width, Lx, Nod_widthx,  Nod_widthy, Vj,  alpha, delta, phi, gi, gf, mu))
    gx = np.load("%s/gamx Wj = %.1f Lx = %.1f nodx = %.1f nody = %.1f Vj = %.1f alpha = %.1f delta = %.2f phi = %.3f gam_i = %.1f gam_f = %.1f mu = %.2f.npy" % (dirS, Junc_width, Lx, Nod_widthx,  Nod_widthy, Vj,  alpha, delta, phi, gi, gf, mu))

    plt.plot(gx, gap/delta)
    plt.grid()

    plt.xlabel(r'$\Gamma_x$ (meV)')
    plt.ylabel(r'$E_{gap}/\Delta$ (meV)')
    plt.xlim(gi, gf)
    #plt.ylim(0, 0.1)
    title = r"$\mu$ = %.1f, $W_j$ = %.1f nm, $nodule_x$ = %.1f nm, $nodule_y$ = %.1f nm, $V_j$ = %.1f meV, $V_{SC}$ = %.1f meV, $\phi$ = %.2f " % (mu, Junc_width, Nod_widthx, Nod_widthy, Vj, Vsc, phi)
    plt.title(title, loc = 'center', wrap = True)
    plt.subplots_adjust(top=0.85)
    plt.show()
