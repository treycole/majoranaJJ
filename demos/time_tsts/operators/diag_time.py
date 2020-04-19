import time
import numpy as np
from numpy import linalg as LA
import scipy.sparse.linalg as spLA

import majoranaJJ.lattice.nbrs as nb #neighbor arrays
import majoranaJJ.lattice.shapes as shps #lattice shapes
import majoranaJJ.modules.plots as plots #plotting functions

#Compared packages
import majoranaJJ.operators.sparse.qmsops as spop #sparse operators
import majoranaJJ.operators.dense.qmdops as dpop #dense operators

Nx = 50
Ny = 50
ax = 2
ay = 2

coor = shps.square(Nx, Ny)
NN = nb.NN_Arr(coor)
NNb = nb.Bound_Arr(coor)

H_sparse = spop.H0(coor, ax, ay, NN)
H_dense = dpop.H0(coor, ax, ay, NN)

start = time.time()

eigs, vecs = LA.eigh(H_dense)

end = time.time()
t_dense = end-start
print("DENSE time for diagonalization for Hamiltonian of size {} = ".format(H_dense.shape), t_dense, "[s]")

print("----------")

start = time.time()

num = 20 # This is the number of eigenvalues and eigenvectors you want
sigma = 0 # This is the eigenvalue we search around
which = 'LM'
spLA.eigsh(H_sparse, k = num, sigma = sigma, which = which)

end = time.time()
t_sparse = end-start
print("SPARSE time for diagonalization for Hamiltonian of size {} = ".format(H_sparse.shape), t_sparse, "[s]")
