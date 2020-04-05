import time
import majoranaJJ.operators.sparsOP as spop
import majoranaJJ.lattice.neighbors as nb
import majoranaJJ.lattice.shapes as shps
import majoranaJJ.etc.plots as plots
from majoranaJJ.etc.mu_finder import mu_finder as mufinder

import numpy as np
import matplotlib.pyplot as plt

Nx = 10
Ny = 10
ax = 50 #[A]
ay = 50 #[A]

coor = shps.square(Nx, Ny) #square lattice
NN = nb.NN_Arr(coor)
NNb = nb.NN_Bound(coor)

Wsc = Ny - 5 #Superconducting region
Wj = 5  #Junction region
Lx = (max(coor[:, 0]) - min(coor[:, 0]) + 1)*ax #Unit cell size in x-direction
Ly = (max(coor[:, 1]) - min(coor[:, 1]) + 1)*ay #Unit cell size in y-direction

#Hamiltonian for size test
H = spop.HBDG(coor, ax, ay, NN, Wsc, Wj)
print("H shape: ", H.shape)

num = 12 # This is the number of eigenvalues and eigenvectors you want
steps = 50 #Number of kx and ky values that are evaluated

alpha = 0.0   #Spin-Orbit Coupling constant: [eV*A]
gammax = 0
gammay = 0
gammaz = np.linspace(0, 1e-3, steps)  #Zeeman field energy contribution: [T]
delta = 3e-3 #Superconducting Gap: [meV]
V0 = 0.0 #Amplitude of potential : [eV]
mu = 0.0 #Chemical Potential: [eV]

#Finding energy of bottom band
#H_etest = spop.H0(coor, ax, ay, NN)
#mu = mufinder(H_etest)

eig_arr = np.zeros((steps, 2))
qx = np.linspace(-np.pi/Lx, np.pi/Lx, steps) #kx in the first Brillouin zone

for i in range(steps):
    energy = spop.EBDG(coor, ax, ay, NN, Wsc, Wj, mu = mu, alpha = alpha, delta = delta, gammaz = gammaz, NNb = NNb, qx = qx[i], periodicX = 'yes', periodicY = 'yes', num = num)
    energy = np.sort(energy)
    eig_arr[i, :] = energy[int(num/2)]

plots.phase(gammaz, eig_arr, title = "Energy vs GammaZ at phi = 0")
