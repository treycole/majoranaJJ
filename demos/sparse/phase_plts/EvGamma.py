import numpy as np
import matplotlib.pyplot as plt

import majoranaJJ.operators.sparsOP as spop #sparse operators
import majoranaJJ.lattice.neighbors as nb #neighbor arrays
import majoranaJJ.lattice.shapes as shps #lattice shapes
import majoranaJJ.etc.plots as plots #plotting functions
from majoranaJJ.etc.mu_finder import mu_finder as mufinder

Nx = 15 #Number of lattice sites allong x-direction
Ny = 15 #Number of lattice sites along y-direction
ax = 2 #lattice spacing in x-direction: [A]
ay = 2 #lattice spacing in y-direction: [A]

coor = shps.square(Nx, Ny) #square lattice
NN = nb.NN_Arr(coor)
NNb = nb.Bound_Arr(coor)

Wsc = Ny - 5 #Superconducting region
Wj = 5  #Junction region
Lx = (max(coor[:, 0]) - min(coor[:, 0]) + 1)*ax #Unit cell size in x-direction
Ly = (max(coor[:, 1]) - min(coor[:, 1]) + 1)*ay #Unit cell size in y-direction

#Hamiltonian for size test
H = spop.HBDG(coor, ax, ay, NN, Wsc, Wj)
print("H shape: ", H.shape)

num = 2 # This is the number of eigenvalues and eigenvectors you want
steps = 30 #Number of kx and ky values that are evaluated

alpha = 0.0   #Spin-Orbit Coupling constant: [eV*A]
gammaz = np.linspace(0, 30e-2, steps)   #Zeeman field energy contribution: [T]
phi = 0
delta = 3e-2 #Superconducting Gap: [eV]
V0 = 0.0 #Amplitude of potential : [eV]
mu = 0.0 #Chemical Potential: [eV]

eig_arr = np.zeros((steps, 2))
for i in range(steps):
    energy = spop.EBDG(coor, ax, ay, NN, Wsc, Wj, NNb = NNb, mu = mu, alpha = alpha, delta = delta, gammaz = gammaz[i], phi = 0, periodicX = True, periodicY = False, num = num)
    energy = np.sort(energy)
    eig_arr[i, :] = energy[int(num/2)]

qx = np.linspace(-np.pi/Lx, np.pi/Lx, steps) #kx in the first Brillouin zone
bands = np.zeros((steps, 12))
for i in range(steps):
    Energy = spop.EBDG(coor, ax, ay, NN, Wsc, Wj, NNb=NNb, mu=mu, V=V0, alpha=alpha, delta=delta, gammaz=0, qx=qx[i], qy=0, periodicX=True, periodicY=False, num=12)

    bands[i,:] = np.sort(Energy)

plots.bands(bands, qx, Lx, Ly, title="Band structure of system with GammaZ = 0")
plots.phase(gammaz, eig_arr, xlabel = 'GammaZ', ylabel = 'Energy', title = "Energy vs GammaZ at kx = 0")
