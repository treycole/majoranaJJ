import numpy as np
import matplotlib.pyplot as plt

import majoranaJJ.operators.sparsOP as spop #sparse operators
import majoranaJJ.lattice.neighbors as nb #neighbor arrays
import majoranaJJ.lattice.shapes as shps #lattice shapes
import majoranaJJ.etc.plots as plots #plotting functions
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
gammaz = 0   #Zeeman field energy contribution: [T]
phi = np.linspace(0, 2*np.pi, steps)
delta = 3e-4 #Superconducting Gap: [eV]
V0 = 0.0 #Amplitude of potential : [eV]
mu = 0.00121 #mufinder(coor, ax, ay, NN, NNb=NNb) #Chemical Potential: [eV]

neigs = 2 # This is the number of eigenvalues and eigenvectors you want
eig_arr = np.zeros((steps, neigs))
for i in range(steps):
    print(steps - i)
    energy = spop.EBDG(coor, ax, ay, NN, NNb=NNb, Wj=Wj, mu=mu, alpha=alpha, gammaz=gammaz, delta=delta, phi=phi[i], periodicX=True, periodicY=True, neigs=neigs, tol=1e-3, maxiter = 800)

    eig_arr[i, :] = 1000*energy

plots.phi_phase(phi, eig_arr, ylabel='Energy [meV]', title="Energy vs Phi at kx = 0")
