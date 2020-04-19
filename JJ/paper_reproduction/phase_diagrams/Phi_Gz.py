import numpy as np
import matplotlib.pyplot as plt

import majoranaJJ.operators.sparse.qmsops as spop #sparse operators
import majoranaJJ.lattice.nbrs as nb #neighbor arrays
import majoranaJJ.lattice.shapes as shps #lattice shapes
import majoranaJJ.modules.plots as plots #plotting functions

from majoranaJJ.etc.mufinder import mufinder

Nx = 20 #Number of lattice sites allong x-direction
Ny = 20 #Number of lattice sites along y-direction
ax = 50 #lattice spacing in x-direction: [A]
ay = 50 #lattice spacing in y-direction: [A]

coor = shps.square(Nx, Ny) #square lattice
NN = nb.NN_Arr(coor)
NNb = nb.Bound_Arr(coor)
lat_size = coor.shape[0]
print("Lattice Size: ", lat_size)

Lx = (max(coor[:, 0]) - min(coor[:, 0]) + 1)*ax #Unit cell size in x-direction
Ly = (max(coor[:, 1]) - min(coor[:, 1]) + 1)*ay #Unit cell size in y-direction

steps = 50 #Number of phi values that are evaluated

Wj = 5  #Junction region
alpha = 0*3e-4   #Spin-Orbit Coupling constant: [eV*A]
gammaz = np.linspace(0, 3e-3, steps)   #Zeeman field energy contribution: [T]
phi = np.linspace(0, 2*np.pi, steps)
delta = 3e-4 #Superconducting Gap: [eV]
V0 = 0.0 #Amplitude of potential : [eV]
mu = 0.00121 #mufinder(coor, ax, ay, NN, NNb=NNb) #Chemical Potential: [eV]

eig_arr = np.zeros(gammaz.size)
for g in range(steps):
    for p in range(steps):
        energy = spop.EBDG(coor, ax, ay, NN, NNb=NNb, Wj=Wj, mu=mu, alpha=alpha, gammaz=gammaz[g], delta=delta, phi=phi[p], periodicX=True, periodicY=True, neigs=neigs, which = 'LM', tol=1e-3, maxiter = 800)
