import sys
import time
import os
import gc

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.signal import argrelextrema
import scipy.linalg as LA
import scipy.sparse.linalg as spLA

import majoranaJJ.operators.sparse_operators as spop #sparse operators
import majoranaJJ.modules.self_energy as slfNRG
from majoranaJJ.operators.potentials import Vjj #potential JJ
import majoranaJJ.lattice.nbrs as nb #neighbor arrays
import majoranaJJ.lattice.shapes as shps #lattice shapes
import majoranaJJ.modules.plots as plots #plotting functions
import majoranaJJ.modules.gamfinder as gamfinder
from majoranaJJ.modules.checkers import boundary_check as bc
import majoranaJJ.modules.checkers as check
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

mui = 0
muf = 6
res = 0.05
steps_mu = int((muf-mui)/(res)) + 1
mu = np.linspace(mui, muf, steps_mu) #meV

gi = 0
gf = 2.0
res = 0.01
steps_gam = int((gf - gi)/(res)) + 1
gx = np.linspace(gi, gf, steps_gam)

q_steps = 501
#qx = np.linspace(0, np.pi/Lx, q_steps) #kx in the first Brillouin zone
qx=0

k = 4
gap_k0 = np.zeros((gx.shape[0], mu.shape[0]))
###################################################
dirS = 'gap_data'
if not os.path.exists(dirS):
    os.makedirs(dirS)
try:
    PLOT = str(sys.argv[1])
except:
    PLOT = 'F'
if PLOT != 'P':
    for i in range(gx.shape[0]):
        for j in range(mu.shape[0]):
            print(gx.shape[0]-i, mu.shape[0]-j)
            H = slfNRG.Junc_eff_Ham_gen(omega=0,W=Wj,ay_targ=ay,kx=0,m_eff=0.023,alp_l=alpha,alp_t=alpha,mu=mu[j],V_J=Vj,Gam=gx[i],Gam_SC_factor=0,Delta=delta,phi=phi,iter=50,eta=0)

            eigs, vecs = spLA.eigsh(H, k=k, sigma=0, which='LM')
            idx_sort = np.argsort(eigs)
            eigs = eigs[idx_sort]
            gap_k0[i,j] = eigs[int(k/2)]


    np.save("%s/gap Wj = %.1f nodx = %.1f nody = %.1f Vj = %.1f Vsc = %.1f alpha = %.1f delta = %.2f phi = %.3f.npy" % (dirS, Junc_width, Nod_widthx,  Nod_widthy, Vj, Vsc, alpha, delta, phi), gap_k0)
    gc.collect()

    sys.exit()
else:
    gap = np.load("%s/gap Wj = %.1f nodx = %.1f nody = %.1f Vj = %.1f Vsc = %.1f alpha = %.1f delta = %.2f phi = %.3f.npy" % (dirS, Junc_width, Nod_widthx,  Nod_widthy, Vj, Vsc, alpha, delta, phi))

    print(gap.shape, mu.shape, gx.shape)
    gap = gap/delta
    gap = np.transpose(gap)

    plt.contourf(gx, mu, gap, 500, vmin = 0, vmax = max(gap.flatten()), cmap = 'hot')
    cbar = plt.colorbar()
    cbar.set_label(r'$E_{gap}/\Delta$')

    plt.xlabel(r'$\Gamma_x$ (meV)')
    plt.ylabel(r'$\mu$ (meV)')

    plt.xlim(gi, gf)
    title = r"$W_j$ = %.1f nm, $nodule_x$ = %.1f nm, $nodule_y$ = %.1f nm, $V_j$ = %.1f meV, $V_{SC}$ = %.1f meV, $\phi$ = %.2f " % (Junc_width, Nod_widthx, Nod_widthy, Vj, Vsc, phi)
    #title = r"$L_x =$ {} nm, $L_y =$ {} nm, SC width = {} nm, $W_j =$ {} nm, $nodule_x = ${} nm, $nodule_y = ${} nm, $\alpha = $ {} meV*A, $\phi =$ {} ".format(Lx*.1, Ly*.1, SC_width, Junc_width, Nod_widthx, Nod_widthy, alpha, phi)
    plt.title(title, loc = 'center', wrap = True)
    plt.subplots_adjust(top=0.85)
    plt.savefig('gap juncwidth = {} nodwidthx = {} nodwidthy = {} phi = {} Vj = {} Vsc = {}.png'.format(Junc_width, Nod_widthx, Nod_widthy, delta, alpha, phi, Vj, Vsc))
    plt.show()

    sys.exit()
