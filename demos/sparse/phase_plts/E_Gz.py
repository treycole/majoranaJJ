import numpy as np
import matplotlib.pyplot as plt

import majoranaJJ.operators.sparsOP as spop #sparse operators
import majoranaJJ.lattice.neighbors as nb #neighbor arrays
import majoranaJJ.lattice.shapes as shps #lattice shapes
import majoranaJJ.etc.plots as plots #plotting functions
from majoranaJJ.etc.mufinder import mufinder

Nx = 3 #Number of lattice sites allong x-direction
Ny = 5 #Number of lattice sites along y-direction
ax = 50 #lattice spacing in x-direction: [A]
ay = 50 #lattice spacing in y-direction: [A]
Wj = 0  #Junction region

coor = shps.square(Nx, Ny) #square lattice
NN = nb.NN_Arr(coor)
NNb = nb.Bound_Arr(coor)lat_size = coor.shape[0]
print("Lattice Size: ", lat_size)

Lx = (max(coor[:, 0]) - min(coor[:, 0]) + 1)*ax #Unit cell size in x-direction
Ly = (max(coor[:, 1]) - min(coor[:, 1]) + 1)*ay #Unit cell size in y-direction

gz_steps = 100 #Number of gamma-Z values that are evaluated

alpha = 3e-4 #Spin-Orbit Coupling constant: [eV*A]
gammaz = np.linspace(0, 1.0, gz_steps) #Zeeman field energy contribution:[eV T]
phi = 0 #SC phase difference
delta = 3e-4 #Superconducting Gap: [eV]
V0 = 0.0 #Amplitude of potential : [eV]
mu  = 8.9e-3#0*42.2e-3 #Chemical Potential: [eV]

neigs = 2 # This is the number of eigenvalues and eigenvectors you want
eig_arr = np.zeros((gz_steps, neigs))
for i in range(gz_steps):
    print(gz_steps - i)
    energy = spop.EBDG(coor, ax, ay, NN, Wj=Wj, NNb=NNb, mu=mu, alpha=alpha, delta=delta, gammaz=gammaz[i], periodicX=True, k=neigs)

    eig_arr[i, :] = 1000*energy

k_steps = 501
neigs = 24
kx = np.linspace(-np.pi/Lx, np.pi/Lx, k_steps) #kx in the first Brillouin zone
bands = np.zeros((k_steps, neigs))
for i in range(k_steps):
    print(k_steps - i)
    energy = spop.EBDG(coor, ax, ay, NN, Wj=Wj, NNb=NNb, alpha=alpha, delta=delta, mu=mu, qx=kx[i], gammaz=.6e-3, periodicX=True, k=neigs)

    bands[i, :] = 1000*energy

plots.bands(bands, kx)
plots.phase(gammaz, eig_arr, xlabel = 'GammaZ [meV]', ylabel = 'Energy [meV]', title = "Energy vs GammaZ at phi = 0")
