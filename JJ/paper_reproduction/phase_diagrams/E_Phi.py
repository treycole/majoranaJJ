import numpy as np
import matplotlib.pyplot as plt

import majoranaJJ.operators.sparse.qmsops as spop #sparse operators
import majoranaJJ.lattice.nbrs as nb #neighbor arrays
import majoranaJJ.lattice.shapes as shps #lattice shapes
import majoranaJJ.modules.plots as plots #plotting functions
#from majoranaJJ.modules.mufinder import mufinder

Nx = 320 #Number of lattice sites along x-direction
Ny = 80 #Number of lattice sites along y-direction
ax = 50 #lattice spacing in x-direction: [A]
ay = 50 #lattice spacing in y-direction: [A]
Wj = 16  #Junction region
print("Junction Width = ", Wj*ay*.10, "(nm)")
print("Supercondicting Lead Width = ", ((Ny - Wj)*ay*.10)/2, "(nm)")
print("L1 parallel to Junction Length = ", (Nx*ax)*.1, "(nm)")

coor = shps.square(Nx, Ny) #square lattice
NN = nb.NN_sqr(coor)
NNb = nb.Bound_Arr(coor)
lat_size = coor.shape[0]
print("Lattice Size: ", lat_size)

Lx = (max(coor[:, 0]) - min(coor[:, 0]) + 1)*ax #Unit cell size in x-direction
Ly = (max(coor[:, 1]) - min(coor[:, 1]) + 1)*ay #Unit cell size in y-direction

phi_steps = 21 #Number of phi values that are evaluated

alpha = 100 #Spin-Orbit Coupling constant: [meV*A]
gammaz = 0 #Zeeman field energy contribution: [meV]
gammax = 0
phi = np.linspace(0, 2*np.pi, phi_steps) #SC phase difference
delta = 0.15 #Superconducting Gap: [meV]
V0 = 0.0 #Amplitude of potential : [meV]
mu = 79.1 #Chemical Potential: [meV]

neigs = 12
eig_arr = np.zeros((phi_steps, neigs))
for i in range(phi_steps):
    print(phi_steps - i)
    energy = spop.EBDG(coor, ax, ay, NN, NNb=NNb, Wj=Wj, mu=mu, alpha=alpha, delta=delta, phi=phi[i], periodicX=False, k=neigs, tol=1e-5, maxiter=300)

    eig_arr[i, :] = energy

plots.phi_phase(phi, eig_arr, Ez = gammaz, savenm = 'E_phi.png', ylim = [-0.15, 0.15])
