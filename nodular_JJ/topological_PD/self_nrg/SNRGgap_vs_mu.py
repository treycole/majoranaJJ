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
import majoranaJJ.modules.self_energy_nodule as slfNRG
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
nodx = 0 #width of nodule
nody = 0 #height of nodule

Junc_width = Wj*.1 #nm
Nod_widthx = nodx*.1 #nm
Nod_widthy = nody*.1 #nm
print("Nodule Width in x-direction = ", Nod_widthx, "(nm)")
print("Nodule Width in y-direction = ", Nod_widthy, "(nm)")
print("Junction Width = ", Junc_width, "(nm)")
#########################################
#Defining Hamiltonian parameters
alpha = 200 #Spin-Orbit Coupling constant: [meV*A]
phi = np.pi #SC phase difference
delta = 1 #Superconducting Gap: [meV]
Vsc = 0 #SC potential: [meV]
Vj = 0 #Junction potential: [meV]
mu = np.linspace(0,1,15) #meV
gx = 0.25 #mev

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
        print(mu.shape[0]-i)
        gapmu[i] = slfNRG.gap(Wj,nodx,nody,ax, ay, gx, mu[i], Vj, alpha, delta, phi)[0]

    np.save("%s/gapfxmu Wj = %.1f nodx = %.1f nody = %.1f Vj = %.1f Vsc = %.1f alpha = %.1f delta = %.2f phi = %.3f.npy" % (dirS, Junc_width, Nod_widthx,  Nod_widthy, Vj, Vsc, alpha, delta, phi), gapmu)
    gc.collect()

    sys.exit()
else:
    gap = np.load("%s/gapfxmu Wj = %.1f nodx = %.1f nody = %.1f Vj = %.1f Vsc = %.1f alpha = %.1f delta = %.2f phi = %.3f.npy" % (dirS, Junc_width, Nod_widthx,  Nod_widthy, Vj, Vsc, alpha, delta, phi))

    plt.plot(mu, gap)
    plt.xlabel(r'$\mu$ (meV)')
    plt.ylabel(r'$E_{gap}$ (meV)')

    plt.xlim(0, 2)
    plt.ylim(0, 0.05)
    title = r"$W_j$ = %.1f nm, $nodule_x$ = %.1f nm, $nodule_y$ = %.1f nm, $V_j$ = %.1f meV, $V_{SC}$ = %.1f meV, $\phi$ = %.2f " % (Junc_width, Nod_widthx, Nod_widthy, Vj, Vsc, phi)

    plt.title(title, loc = 'center', wrap = True)
    plt.subplots_adjust(top=0.85)
    plt.savefig('gapfxmu juncwidth = {} nodwidthx = {} nodwidthy = {} phi = {} Vj = {} Vsc = {}.png'.format(Junc_width, Nod_widthx, Nod_widthy, delta, alpha, phi, Vj, Vsc))
    plt.show()

    sys.exit()
