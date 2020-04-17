import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse.linalg as spLA

import majoranaJJ.operators.sparsOP as spop
import majoranaJJ.lattice.neighbors as nb
import majoranaJJ.lattice.shapes as shps
import majoranaJJ.etc.plots as plots

Nx = 30
Ny = 30
ax = 10 #[A]
ay = 10 #[A]

coor = shps.square(Nx, Ny)
NN = nb.NN_Arr(coor)
NNb = nb.Bound_Arr(coor)
print("lattice size", coor.shape[0])

alpha = 0   #Spin-Orbit Coupling constant: [eV*A]
gammaz = 0  #Zeeman field energy contribution: [T]
delta = 0 #Superconducting Gap: [eV]
V0 = 0.0 #Amplitude of potential : [eV]
mu = 0 #Chemical Potential: [eV]

H = spop.H_BDG(coor, ax, ay, NN, Wsc=Ny, Wj=0)
print("H shape: ", H.shape)

num = 20 # This is the number of eigenvalues and eigenvectors you want
sigma = 0 # This is the eigenvalue we search around
which = 'LM'
eigs, vecs = spLA.eigsh(H, k = num, sigma = sigma, which = which)

plots.state_cmap(coor, eigs, vecs, n = 7, title = 'hole  n = 3 energy eigenstate')
plots.state_cmap(coor, eigs, vecs, n = 12, title = 'particle n = 3 energy eigenstate')
plots.state_cmap(coor, eigs, vecs, n = 9, title = 'hole n = 1 energy eigenstate')
plots.state_cmap(coor, eigs, vecs, n = 10, title = 'particle n = 1 energy eigenstate')
