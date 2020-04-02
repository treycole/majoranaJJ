import time
import majoranaJJ.operators.sparsOP as spop
import majoranaJJ.lattice.neighbors as nb
import majoranaJJ.lattice.shapes as shps

import numpy as np
import majoranaJJ.plots as plots
import matplotlib.pyplot as plt
import scipy.sparse.linalg as spLA

Nx = 100
Ny = 105
ax = 2
ay = 2

start = time.time()
coor = shps.square(Nx, Ny)
NN = nb.NN_Arr(coor)
NNb = nb.NN_Bound(coor)
end = time.time()
print("Time for lattice arrays for lattice of size {} = ".format(coor.shape[0]), end-start, "[s]")

Wsc = Ny
Wj = 0
Lx = (max(coor[:, 0]) - min(coor[:, 0]) + 1)*ax #Unit cell size in x-direction
Ly = (max(coor[:, 1]) - min(coor[:, 1]) + 1)*ay #Unit cell size in y-direction

alpha = 0.0   #Spin-Orbit Coupling constant: [eV*A]
gammaz = 0.0000001  #Zeeman field energy contribution: [T]
delta = 0.01 #Superconducting Gap: [eV]
V0 = 0.0 #Amplitude of potential : [eV]
mu = 0 #Chemical Potential: [eV]

H = spop.HBDG(coor, ax, ay, NN, Wsc, Wj, mu = mu, delta = delta, gammaz = gammaz, NNb = NNb)
num = 20 # This is the number of eigenvalues and eigenvectors you want
sigma = 0 # This is the eigenvalue we search around
which = 'LM'
print("H shape: ", H.shape)

start = time.time()
eigs, vecs = spLA.eigsh(H, k = num, sigma = sigma, which = which)
end = time.time()
print("Time for diagonalization for Hamiltonian of size {} = ".format(H.shape[0]), end-start, "[s]")


idx_sort = np.argsort(eigs)
eigs = eigs[idx_sort]
vecs = vecs[:,idx_sort]
print(eigs[:])

plots.state_cplot(coor,vecs[:, 7],title = 'hole  n = 3 energy eigenstate')
plots.state_cplot(coor,vecs[:, 12],title = 'particle  n = 3 energy eigenstate')
plots.state_cplot(coor,vecs[:, 9],title = 'hole n = 1 energy eigenstate')
plots.state_cplot(coor,vecs[:, 10],title =' particle n = 1 energy eigenstate')
