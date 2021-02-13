import sys
import os
import gc

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
import scipy.sparse.linalg as spLA

import majoranaJJ.modules.plots as plots #plotting functions
import majoranaJJ.modules.finders as fndrs
import majoranaJJ.modules.SNRG as SNRG
import majoranaJJ.modules.fig_params as params
###################################################
#Defining System
ax = 50 #lattice spacing in x-direction: [A]
ay = 50 #lattice spacing in y-direction: [A]
Nx = 20 #Number of lattice sites along x-direction
Wj = 1000 #Junction region [A]
cutx = 5 #width of nodule
cuty = 8 #height of nodule
Lx = Nx*ax #Angstrom
Junc_width = Wj*.10 #nm
Nod_widthx = cutx*ax*.1 #nm
Nod_widthy = cuty*ay*.1 #nm

print("Nodule Width in x-direction = ", Nod_widthx, "(nm)")
print("Nodule Width in y-direction = ", Nod_widthy, "(nm)")
print("Junction Width = ", Junc_width, "(nm)")
###################################################
#Defining Hamiltonian parameters
alpha = 200 #Spin-Orbit Coupling constant: [meV*A]
phi = 0*np.pi #SC phase difference
delta = 0.3 #Superconducting Gap: [meV]
Vsc = 0 #SC potential: [meV]
Vj = -25 #Junction potential: [meV]
mu = Vj
gam = 0 #mev

steps = 100
kx = np.linspace(0, np.pi/Lx, steps)
k = 200
#kx = np.linspace(0.004, 0.0042, steps)
omega0_bands = np.zeros((k, kx.shape[0]))
true_bands = np.zeros((k, kx.shape[0]))
###################################################
#phase diagram mu vs gam
dirS = 'bands_data'
if not os.path.exists(dirS):
    os.makedirs(dirS)
try:
    PLOT = str(sys.argv[1])
except:
    PLOT = 'F'
if PLOT != 'P':
    for i in range(kx.shape[0]):
        print(omega0_bands.shape[1]-i, kx[i])
        H = SNRG.Junc_eff_Ham_gen(omega=0, Wj=Wj, Lx=Lx, nodx=cutx, nody=cuty, ax=ax, ay=ay, kx=kx[i], m_eff=0.026, alp_l=alpha, alp_t=alpha, mu=mu, Vj=Vj, Gam=gam, Gam_SC_factor=0, delta=delta, phi=phi, iter=50, eta=0)
        S = int(H.shape[0]/2)
        H = (H[:S, :])[:, :S]
        eigs, vecs = spLA.eigsh(H, k=k, sigma=0, which='LM')
        idx_sort = np.argsort(eigs)
        eigs = eigs[idx_sort]
        #print(eigs)
        #arg = np.argmin(np.absolute(eigs))
        #print(arg)
        omega0_bands[:, i] = eigs[:]

    np.save("%s/SOCbands Lx = %.1f Wj = %.1f nodx = %.1f nody = %.1f Vj = %.1f alpha = %.1f delta = %.2f phi = %.3f mu = %.1f gam = %.1f.npy" % (dirS, Lx*.1, Junc_width, Nod_widthx,  Nod_widthy, Vj, alpha, delta, phi, mu, gam), omega0_bands)
    gc.collect()

    #for i in range(omega0_bands.shape[0]):
    #    for j in range(omega0_bands.shape[1]):
    #        print(omega0_bands.shape[0]-i, omega0_bands.shape[1]-j)
    #        true_eig = SNRG.self_consistency_finder(Wj=Wj, Lx=Lx,nodx=cutx, nody=cuty, ax=ax, ay=ay, gam=gam, mu=mu, Vj=Vj, alpha=alpha, delta=delta, phi=phi, kx=kx[i], eigs_omega0=omega0_bands[i,j], m_eff=0.026, tol=1e-8, k=k)
        #    true_bands[i, j] = true_eig

    #np.save("%s/SOCbands Lx = %.1f Wj = %.1f nodx = %.1f nody = %.1f Vj = %.1f alpha = %.1f delta = %.2f phi = %.3f mu = %.1f gam = %.1f.npy" % (dirS, Lx*.1, Junc_width, Nod_widthx,  Nod_widthy, Vj, alpha, delta, phi, mu, gam), true_bands)

    sys.exit()
else:
    omega0_bands = np.load("%s/SOCbands Lx = %.1f Wj = %.1f nodx = %.1f nody = %.1f Vj = %.1f alpha = %.1f delta = %.2f phi = %.3f mu = %.1f gam = %.1f.npy" % (dirS, Lx*.1, Junc_width, Nod_widthx,  Nod_widthy, Vj, alpha, delta, phi, mu, gam))

    for i in range(k):
        #plt.scatter(kx, omega0_bands[i,:], c='r', s=2)
        plt.plot(kx, omega0_bands[i, :], c='b')
    #plt.ylim(0,30)
    plt.show()
