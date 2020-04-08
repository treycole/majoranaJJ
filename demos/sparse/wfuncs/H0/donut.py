import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse.linalg as spLA

import majoranaJJ.operators.sparsOP as spop
import majoranaJJ.lattice.neighbors as nb
import majoranaJJ.lattice.shapes as shps
import majoranaJJ.etc.plots as plots

R = 25
r = 10
ax = 10 #[A]
ay = 10 #[A]

coor = shps.donut(R, r)
NN = nb.NN_Arr(coor)
print("lattice size", coor.shape[0])

alpha = 0   #Spin-Orbit Coupling constant: [eV*A]
gammaz = 0  #Zeeman field energy contribution: [T]
delta = 0 #Superconducting Gap: [eV]
V0 = 0.0 #Amplitude of potential : [eV]
mu = 0 #Chemical Potential: [eV]

H = spop.H0(coor, ax, ay, NN)
print("H shape: ", H.shape)

num = 12 # This is the number of eigenvalues and eigenvectors you want
sigma = 0 # This is the eigenvalue we search around
which = 'LM'
eigs, vecs = spLA.eigsh(H, k = num, sigma = sigma, which = which)

plots.state_cmap(coor, eigs, vecs, n = 0, title = 'SPARSE Free Particle Ground State')
n = 4
plots.state_cmap(coor, eigs, vecs, n = n, title = 'DENSE: State # {}'.format(n))
