import time
import majoranaJJ.operators.sparsOP as spop
import majoranaJJ.lattice.neighbors as nb
import majoranaJJ.lattice.shapes as shps
import majoranaJJ.etc.plots as plots

import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse.linalg as spLA

Nx = 15
Ny = 15
ax = 2 #[A]
ay = 2 #[A]

start = time.time()
coor = shps.square(Nx, Ny) #square lattice
NN = nb.NN_Arr(coor)
NNb = nb.NN_Bound(coor)
end = time.time()
print("Time for lattice arrays for lattice of size {} = ".format(coor.shape[0]), end-start, "[s]")

Wsc = Ny #Superconducting region
Wj = 0  #Junction region
Lx = (max(coor[:, 0]) - min(coor[:, 0]) + 1)*ax #Unit cell size in x-direction
Ly = (max(coor[:, 1]) - min(coor[:, 1]) + 1)*ay #Unit cell size in y-direction

#Hamiltonian for size test
H = spop.HBDG(coor,ax,ay, NN, Wsc, Wj)
num = 12 # This is the number of eigenvalues and eigenvectors you want
sigma = 0 *0.001766 # This is the eigenvalue we search around
which = 'LM'
print("H shape: ", H.shape)

alpha = 0.0   #Spin-Orbit Coupling constant: [eV*A]
gammax = 0
gammay = 0
gammaz = 0.001  #Zeeman field energy contribution: [T]
delta = 0.03 #Superconducting Gap: [eV]
V0 = 0.0 #Amplitude of potential : [eV]
mu = 0.0 #Chemical Potential: [eV]

steps = 80 #Number of kx and ky values that are evaluated
qx = np.linspace(-np.pi/Lx, np.pi/Lx, steps) #kx in the first Brillouin zone
qy = np.linspace(-np.pi/Ly, np.pi/Ly, steps) #ky in the first Brillouin zone
eig_arr = np.zeros((steps,num))

start = time.time()
for i in range(steps):
    eig_arr[i,:] = np.sort(spLA.eigsh(spop.HBDG(coor, ax, ay, NN, Wsc, Wj, mu = mu, alpha = alpha,
    delta = delta, gammaz = gammaz, NNb = NNb, qx = qx[i], qy = 0, periodicX = 'yes', periodicY = 'yes'), k = num, sigma = sigma, which = which)[0])
end = time.time()
print("Time for finding bands for Hamiltonian of size {} at {} k values ".format(H.shape[0], steps), end-start, "[s]")

plots.bands(eig_arr, qx, Lx, Ly, title = "delta = 0.03 [eV] BDG Bands")
