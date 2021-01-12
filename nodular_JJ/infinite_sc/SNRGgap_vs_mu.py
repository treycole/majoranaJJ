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
Nx = 10 #Number of lattice sites along x-direction
Wj = 2000 #Junction region [A]
nodx = 4 #width of nodule
nody = 10 #height of nodule
Lx = Nx*ax

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
delta = 1 #Superconducting Gap: [meV]
Vj = -5 #Junction potential: [meV]

mu_i = -4.2
mu_f = -3.4
delta_mu = mu_f - mu_i
res = 0.05
steps = int(abs(delta_mu/res))
mu = np.linspace(mu_i, mu_f, steps) #meV
gx = 1.00 #mev

print("alpha = ", alpha)
print("Mu_i = ", mu_i)
print("Mu_f = ", mu_f)
print("Gamma_x = ", gx)
print("Vj = ", Vj)

gapmu = np.zeros(mu.shape[0])
###################################################
dirS = 'gap_data'
if not os.path.exists(dirS):
    os.makedirs(dirS)
try:
    PLOT = str(sys.argv[1])
except:
    PLOT = 'F'
if PLOT != 'P':
    for i in range(mu.shape[0]):
        print(mu[i], mu.shape[0]-i)
        gapmu[i] = SNRG.gap(Wj=Wj, Lx=Lx, nodx=nodx, nody=nody, ax=ax, ay=ay, gam=gx, mu=mu[i], Vj=Vj, alpha=alpha, delta=delta, phi=phi, muf=10)[0]

    np.save("%s/gapfxmu Wj = %.1f nodx = %.1f nody = %.1f Vj = %.1f alpha = %.1f delta = %.2f phi = %.3f mu_i = %.1f mu_f=%.1f gx=%.2f.npy" % (dirS, Junc_width, Nod_widthx,  Nod_widthy, Vj,  alpha, delta, phi, mu_i, mu_f, gx), gapmu)
    np.save("%s/mu Wj = %.1f nodx = %.1f nody = %.1f Vj = %.1f alpha = %.1f delta = %.2f phi = %.3f mu_i = %.1f mu_f=%.1f gx=%.2f.npy" % (dirS, Junc_width, Nod_widthx,  Nod_widthy, Vj, alpha, delta, phi, mu_i, mu_f, gx), mu)
    gc.collect()

    sys.exit()
else:
    gap = np.load("%s/gapfxmu Wj = %.1f nodx = %.1f nody = %.1f Vj = %.1f alpha = %.1f delta = %.2f phi = %.3f mu_i = %.1f mu_f=%.1f gx=%.2f.npy" % (dirS, Junc_width, Nod_widthx,  Nod_widthy, Vj, alpha, delta, phi, mu_i, mu_f, gx))
    mu = np.load("%s/mu Wj = %.1f nodx = %.1f nody = %.1f Vj = %.1f alpha = %.1f delta = %.2f phi = %.3f mu_i = %.1f mu_f=%.1f gx=%.2f.npy" % (dirS, Junc_width, Nod_widthx,  Nod_widthy, Vj, alpha, delta, phi, mu_i, mu_f, gx))

    steps = gap.shape[0]
    mu = np.linspace(mu_i, mu_f, steps)
    plt.plot(mu, gap)
    plt.grid()
    plt.xlabel(r'$\mu$ (meV)')
    plt.ylabel(r'$E_{gap}$ (meV)')
    #plt.xlim(0, 2)
    #plt.ylim(0, 0.1)
    title = r"SNRG $E_Z$ = %.2f meV $W_j$ = %.1f nm, $nodule_x$ = %.1f nm, $nodule_y$ = %.1f nm, $V_j$ = %.1f meV, $\phi$ = %.2f " % (gx, Junc_width, Nod_widthx, Nod_widthy, Vj, phi)

    plt.title(title, loc = 'center', wrap = True)
    plt.subplots_adjust(top=0.85)
    plt.savefig('gapfxmu juncwidth = {} nodwidthx = {} nodwidthy = {} alpha = {} phi = {} Vj = {}.png'.format(Junc_width, Nod_widthx, Nod_widthy, alpha, phi, Vj))
    plt.show()

    sys.exit()
