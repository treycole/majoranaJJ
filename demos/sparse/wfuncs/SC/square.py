import time
import majoranaJJ.operators.sparsOP as spop
import majoranaJJ.lattice.neighbors as nb
import majoranaJJ.lattice.shapes as shps
import majoranaJJ.etc.plots as plots

import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse.linalg as spLA

Nx = 12
Ny = 12
ax = 2
ay = 2

coor = shps.square(Nx, Ny)
NN = nb.NN_Arr(coor)
NNb = nb.Bound_Arr(coor)

Wsc = Ny #Width of Superconductor
Wj = 0 #Width of Junction
Lx = (max(coor[:, 0]) - min(coor[:, 0]) + 1)*ax #Unit cell size in x-direction
Ly = (max(coor[:, 1]) - min(coor[:, 1]) + 1)*ay #Unit cell size in y-direction

alpha = 0.0   #Spin-Orbit Coupling constant: [eV*A]
gammaz = 0.0000001  #Zeeman field energy contribution: [T]
delta = 0.01 #Superconducting Gap: [eV]
V0 = 0.0 #Amplitude of potential : [eV]
mu = 0 #Chemical Potential: [eV]

H = spop.HBDG(coor, ax, ay, NN, Wsc, Wj, mu = mu, delta = delta, gammaz = gammaz, NNb = NNb)
print("H shape: ", H.shape)

num = 20 # This is the number of eigenvalues and eigenvectors you want
sigma = 0 # This is the eigenvalue we search around
which = 'LM'
eigs, vecs = spLA.eigsh(H, k = num, sigma = sigma, which = which)

plots.state_cmap(coor, eigs, vecs, n = 7, title = 'hole  n = 3 energy eigenstate')
plots.state_cmap(coor, eigs, vecs, n = 12, title = 'particle n = 3 energy eigenstate')
plots.state_cmap(coor, eigs, vecs, n = 9, title = 'hole n = 1 energy eigenstate')
plots.state_cmap(coor, eigs, vecs, n = 10, title = 'particle n = 1 energy eigenstate')
