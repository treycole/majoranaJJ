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

coor = shps.square(Nx, Ny) #square lattice
NN = nb.NN_Arr(coor)
NNb = nb.Bound_Arr(coor)

Lx = (max(coor[:, 0]) - min(coor[:, 0]) + 1)*ax #Unit cell size in x-direction
Ly = (max(coor[:, 1]) - min(coor[:, 1]) + 1)*ay #Unit cell size in y-direction

#Hamiltonian for size test
H = spop.HBDG(coor, ax, ay, NN)
print("H shape: ", H.shape)

steps = 100 #Number of gammaZ values that are evaluated

Wj = 0  #Junction region
alpha = 3e-4 #Spin-Orbit Coupling constant: [eV*A]
gammaz = np.linspace(0, 1e-3, steps)  #Zeeman field energy contribution: [eV T]
phi = 0
delta = 3e-4 #Superconducting Gap: [eV]
V0 = 0.0 #Amplitude of potential : [eV]
mu  = 40.2e-3 #Chemical Potential: [eV]

neigs = 2 # This is the number of eigenvalues and eigenvectors you want
eig_arr = np.zeros((steps, 2))
for i in range(steps):
    print(steps - i)
    energy = spop.EBDG(coor, ax, ay, NN, Wj=Wj, NNb = NNb, mu = mu, alpha=alpha, delta=delta, gammaz = gammaz[i], qx = (1/50)*(np.pi/Lx), periodicX = True, periodicY = False, neigs=neigs, tol=1e-5, maxiter=1000, which = 'LM')

    eig_arr[i, :] = 1000*energy

steps = 501
neigs = 24
qx = np.linspace(-np.pi/Lx, np.pi/Lx, steps) #kx in the first Brillouin zone
qy = np.linspace(-np.pi/Ly, np.pi/Ly, steps) #ky in the first Brillouin zone
bands = np.zeros((steps, neigs))
for i in range(steps):
    print(steps - i)
    energy = spop.EBDG(coor, ax, ay, NN, Wj=Wj, NNb=NNb, alpha=alpha, delta=delta, mu=mu, qx=qx[i], gammaz = 8e-4, periodicX=True, periodicY=False, neigs=neigs)

    bands[i, :] = 1000*energy

plots.bands(bands, qx, units = "[meV]")
plots.phase(1000*gammaz, eig_arr, xlabel = 'GammaZ [meV]', ylabel = 'Energy [meV]', title = "Energy vs GammaZ at phi = 0")
