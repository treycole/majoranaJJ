import time
import majoranaJJ.operators.sparsOP as spop
import majoranaJJ.lattice.neighbors as nb
import majoranaJJ.lattice.shapes as shps
import majoranaJJ.etc.plots as plots
from majoranaJJ.etc.mu_finder import mu_finder as mufinder

import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse.linalg as spLA

Nx = 10
Ny = 10
ax = 50 #[A]
ay = 50 #[A]

start = time.time()
coor = shps.square(Nx, Ny) #square lattice
NN = nb.NN_Arr(coor)
NNb = nb.NN_Bound(coor)
end = time.time()
print("Time for lattice arrays for lattice of size {} = ".format(coor.shape[0]), end-start, "[s]")

Wsc = Ny - 5 #Superconducting region
Wj = 5  #Junction region
Lx = (max(coor[:, 0]) - min(coor[:, 0]) + 1)*ax #Unit cell size in x-direction
Ly = (max(coor[:, 1]) - min(coor[:, 1]) + 1)*ay #Unit cell size in y-direction

#Hamiltonian for size test
H = spop.HBDG(coor,ax,ay, NN, Wsc, Wj)
N = H.shape[0]
print("H shape: ", N, "x", N)

num = 12 # This is the number of eigenvalues and eigenvectors you want
sigma = 0  # This is the eigenvalue we search around
which = 'LM'
steps = 50 #Number of kx and ky values that are evaluated

alpha = 0.0   #Spin-Orbit Coupling constant: [eV*A]
gammax = 0
gammay = 0
gammaz = np.linspace(0, 1e-3, steps)  #Zeeman field energy contribution: [T]
delta = 1e-3 #Superconducting Gap: [meV]
V0 = 0.0 #Amplitude of potential : [eV]
mu = 0.0 #Chemical Potential: [eV]

#Finding energy of bottom band
#H_etest = spop.H0(coor, ax, ay, NN)
#mu = mufinder(H_etest)

eig_arr = np.zeros((steps, 2))
qx = np.linspace(-np.pi/Lx, np.pi/Lx, steps) #kx in the first Brillouin zone

for i in range(steps):
    e = np.sort(spLA.eigsh(1000*spop.HBDG(coor, ax, ay, NN, Wsc, Wj, NNb = NNb, mu = mu,
    alpha = alpha, delta = delta, gammaz = gammaz[i], qx = 0, qy=0,
    periodicX = 'yes', periodicY = 'yes'), k = num, sigma = sigma, which = which)[0])
    eig_arr[i, :] = e[6]

plots.phase(gammaz, eig_arr, title = "Energy vs GammaZ at phi = 0")
