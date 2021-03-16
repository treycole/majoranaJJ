import sys
import os
import gc

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
import scipy.interpolate as interp

import majoranaJJ.modules.plots as plots #plotting functions
import majoranaJJ.modules.finders as fndrs
import majoranaJJ.modules.SNRG as SNRG
import majoranaJJ.modules.distance as distance
import matplotlib.colors as colors
###################################################
#Defining System
ax = 50 #lattice spacing in x-direction: [A]
ay = 50 #lattice spacing in y-direction: [A]
Nx = 3 #Number of lattice sites along x-direction
Ny = 500
Wj = 20 #Junction region [A]
cutx = 0 #width of nodule
cuty = 0 #height of nodule

cutxT = cutx
cutxB = cutx
cutyT = cuty
cutyB = cuty
Lx = Nx*ax #Angstrom

coor = shps.square(Nx, Ny) #square lattice
NN = nb.NN_Arr(coor) #neighbor array
NNb = nb.Bound_Arr(coor) #boundary array

Junc_width = Wj*.1 #nm
cutxT_width = cutxT*ax*.1 #nm
cutyT_width = cutyT*ax*.1 #nm
cutxB_width = cutxB*ax*.1 #nm
cutyB_width = cutyB*ax*.1 #nm

print("Lx = ", Lx*.1, "(nm)" )
print("Top Nodule Width in x-direction = ", cutxT_width, "(nm)")
print("Bottom Nodule Width in x-direction = ", cutxB_width, "(nm)")
print("Top Nodule Width in y-direction = ", cutyT_width, "(nm)")
print("Bottom Nodule Width in y-direction = ", cutyB_width, "(nm)")
print("Junction Width = ", Junc_width, "(nm)")
###################################################
#Defining Hamiltonian parameters
alpha = 200 #Spin-Orbit Coupling constant: [meV*A]
phi = 0*np.pi #SC phase difference
delta = 0.3 #Superconducting Gap: [meV]
Vj = -40 #junction potential: [meV]
mu = 0
gx = 1
###################################################
k = 4 #This is the number of eigenvalues and eigenvectors you want
steps = 100 #Number of kx values that are evaluated
qx = np.linspace(0.0035, 0.0036, steps) #kx in the first Brillouin zone
qmax = np.sqrt(2*(5-Vj)*0.026/const.hbsqr_m0)*1.25
qx = np.linspace(0, qmax, steps)
bands = np.zeros((steps, k))
for i in range(steps):
    print(steps - i)
    H = spop.HBDG(coor, ax, ay, NN, NNb=NNb, Wj=Wj,cutxT=cutxT, cutyT=cutyT, cutxB = cutxB, cutyB = cutyB, Vj=Vj, mu=mu, alpha=alpha, delta=delta, phi=phi, gamx=gx, qx=qx[i])
    eigs, vecs = spLA.eigsh(H, k=k, sigma=0, which='LM')
    idx_sort = np.argsort(eigs)
    eigs = eigs[idx_sort]
    bands[i, :] = eigs

H = spop.HBDGHBDG(coor, ax, ay, NN, NNb=NNb, Wj=Wj, cutxT=cutxT, cutyT=cutyT, cutxB = cutxB, cutyB = cutyB, Vj = Vj, mu = mu, gamx = gx,alpha = alpha, delta = delta, phi = phi, qx = None)
eigs, vecs = spLA.eigsh(H, k=k, sigma=0, which='LM')
idx_sort = np.argsort(eigs)
eigs = eigs[idx_sort]
