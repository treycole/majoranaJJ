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
import majoranaJJ.lattice.nbrs as nb #neighbor arrays
import majoranaJJ.lattice.shapes as shps #lattice shapes
import majoranaJJ.modules.plots as plots #plotting functions
import majoranaJJ.modules.finders as fndrs
from majoranaJJ.modules.checkers import boundary_check as bc
import majoranaJJ.modules.checkers as check
import majoranaJJ.modules.constants as const
import majoranaJJ.operators.potentials as potentials
###################################################
#Defining System
Nx = 10 #Number of lattice sites along x-direction
Ny = 290 #Number of lattice sites along y-direction
ax = 50 #lattice spacing in x-direction: [A]
ay = 50 #lattice spacing in y-direction: [A]
Wj = 40 #Junction region
cutx = 4 #width of nodule
cuty = 10 #height of nodule
Nx, Ny, cutx, cuty, Wj = check.junction_geometry_check(Nx, Ny, cutx, cuty, Wj)

Junc_width = Wj*ay*.10 #nm
SC_width = ((Ny - Wj)*ay*.10)/2 #nm
Nod_widthx = cutx*ax*.1 #nm
Nod_widthy = cuty*ay*.1 #nm
print("Nodule Width in x-direction = ", Nod_widthx, "(nm)")
print("Nodule Width in y-direction = ", Nod_widthy, "(nm)")
print("Junction Width = ", Junc_width, "(nm)")
print("Supercondicting Lead Width = ", SC_width, "(nm)")
###################################################
coor = shps.square(Nx, Ny) #square lattice
NN = nb.NN_Arr(coor) #neighbor array
NNb = nb.Bound_Arr(coor) #boundary array

Lx = (max(coor[:, 0]) - min(coor[:, 0]) + 1)*ax #Unit cell size in x-direction
Ly = (max(coor[:, 1]) - min(coor[:, 1]) + 1)*ay #Unit cell size in y-direction
###################################################
#Hamiltonian Parameters
alpha = 100 #Spin-Orbit Coupling constant: [meV*A]
gx = 0 #parallel to junction: [meV]
mu = 0 #Chemical Potential: [meV]
Vj = -5
potentials.Vjj(coor, Wj, 0, Vj, cutx = cutx, cuty = cuty)
###################################################
#phase diagram mu vs gamx
k = 100 #This is the number of eigenvalues and eigenvectors you want
steps = 101 #Number of kx and ky values that are evaluated

qx = np.linspace(0, np.pi/Lx, steps) #kx in the first Brillouin zone
bands = np.zeros((steps, k))
for i in range(steps):
    print(steps - i)
    H = spop.H0(coor, ax, ay, NN, NNb=NNb, Wj=Wj, cutx=cutx, cuty=cuty, V=0, mu=mu, alpha=alpha, gamx=gx, qx=qx[i])
    eigs, vecs = spLA.eigsh(H, k=k, sigma=0, which='LM')
    idx_sort = np.argsort(eigs)
    eigs = eigs[idx_sort]
    bands[i, :] = eigs

title = r"$L_x = %d nm$ , $L_y = %d nm$ , $W_{SC} = %.1f nm$, $W_j = %.1f$ , $nodule_x = %.1f$, $nodule_y = %.1f$, $\alpha = %.1f$, $\mu = %.5f$" % (Lx*.1, Ly*.1, SC_width, Junc_width, Nod_widthx, Nod_widthy, alpha, mu)
plots.bands(qx, bands, units = "meV", savenm = 'nodx={}nm_nody={}nm_Wj={}nm_Wsc={}nm_mu={}.png'.format(Nod_widthx, Nod_widthy, Junc_width, SC_width, mu), title = title)
