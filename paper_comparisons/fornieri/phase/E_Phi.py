import sys
import time
import os
import gc

import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse.linalg as spLA

import majoranaJJ.operators.sparse_operators as spop #sparse operators
from majoranaJJ.operators.potentials import Vjj #potential JJ
import majoranaJJ.lattice.nbrs as nb #neighbor arrays
import majoranaJJ.lattice.shapes as shps #lattice shapes
import majoranaJJ.modules.plots as plots #plotting functions
import majoranaJJ.modules.gamfinder as gamfinder
from majoranaJJ.modules.checkers import boundary_check as bc
import majoranaJJ.modules.checkers as check

Nx = 320 #Number of lattice sites along x-direction
Ny = 80 #Number of lattice sites along y-direction
ax = 50 #lattice spacing in x-direction: [A]
ay = 50 #lattice spacing in y-direction: [A]
Wj = 16  #Junction region
cutx = 0
cuty = 0

Junc_width = Wj*ay*.10 #nm
Wsc = ((Ny - Wj)*ay*.10)/2 #nm
Nod_widthx = cutx*ax*.1 #nm
Nod_widthy = cuty*ay*.1 #nm
print("Nodule Width in x-direction = ", Nod_widthx, "(nm)")
print("Nodule Width in y-direction = ", Nod_widthy, "(nm)")
print("Junction Width = ", Junc_width, "(nm)")
print("Supercondicting Lead Width = ", Wsc, "(nm)")
###################################################
coor = shps.square(Nx, Ny) #square lattice
NN = nb.NN_sqr(coor)
NNb = nb.Bound_Arr(coor)

Lx = (max(coor[:, 0]) - min(coor[:, 0]) + 1)*ax #Unit cell size in x-direction
Ly = (max(coor[:, 1]) - min(coor[:, 1]) + 1)*ay #Unit cell size in y-direction
print("Lx = {} nm".format(Lx*.1))
print("Ly = {} nm".format(Ly*.1))
###################################################
phi_steps = 31 #Number of phi values that are evaluated
k = 12

alpha = 100 #Spin-Orbit Coupling constant: [meV*A]
gx = 0.0 #Zeeman field energy contribution: [meV]
phi = np.linspace(0, 2*np.pi, phi_steps) #SC phase difference
delta = 0.15 #Superconducting Gap: [meV]
Vsc = 0.0 #Amplitude of potential : [meV]
Vj = 0.0
mu = 79.1 #Chemical Potential: [meV]

dirS = 'E_phi_data'
if not os.path.exists(dirS):
    os.makedirs(dirS)
try:
    PLOT = str(sys.argv[1])
except:
    PLOT = 'F'
if PLOT != 'P':
    eig_arr = np.zeros((phi_steps, k))
    for i in range(phi_steps):
        print(phi_steps - i)
        H = spop.HBDG(coor, ax, ay, NN, NNb = NNb, Wj=Wj, mu=mu, gamx=gx, alpha=alpha, delta=delta, phi=phi[i])
        eigs, vecs = spLA.eigsh(H, k=k, sigma=0, which='LM')
        idx_sort = np.argsort(eigs)
        eigs = eigs[idx_sort]
        eig_arr[i, :] = np.sort(eigs)

    np.save("%s/eig_arr Lx = %.1f Ly = %.1f Wsc = %.1f Wj = %.1f nodx = %.1f nody = %.1f alpha = %.1f delta = %.2f mu = %.1f gx = %.1f.npy" % (dirS, Lx*.1, Ly*.1, Wsc, Junc_width, Nod_widthx, Nod_widthy, alpha, delta, mu, gx), eig_arr)
    gc.collect()
    sys.exit()
else:
    eig_arr = np.load("%s/eig_arr Lx = %.1f Ly = %.1f Wsc = %.1f Wj = %.1f nodx = %.1f nody = %.1f alpha = %.1f delta = %.2f mu = %.1f gx = %.1f.npy" % (dirS, Lx*.1, Ly*.1, Wsc, Junc_width, Nod_widthx, Nod_widthy, alpha, delta, mu, gx))

    plots.phi_phase(phi, eig_arr, Ez = gx, savenm = 'E_phi.png', ylim = [-0.15, 0.15])
