import time
import numpy as np

import majoranaJJ.lattice.neighbors as nb #neighbor arrays
import majoranaJJ.lattice.shapes as shps #lattice shapes
import majoranaJJ.etc.plots as plots #plotting functions

#Compared packages
import majoranaJJ.operators.sparsOP as spop #sparse operators
import majoranaJJ.operators.densOP as dpop #dense operators
print(" ")

N = range(3, 50)
t_sparse = np.zeros(len(N))
t_dense = np.zeros(len(N))
ax = 2
ay = 2

for i in range(0, len(N)):
    Nx = N[i]
    Ny = N[i]
    coor = shps.square(Nx, Ny)
    NN = nb.NN_Arr(coor)
    NNb = nb.Bound_Arr(coor)

    H_sparse = spop.H0(coor, ax, ay, NN)
    H_dense = dpop.H0(coor, ax, ay, NN)

    start = time.time()

    eigs, vecs = LA.eigh(H_dense)

    end = time.time()
    t_dense[i] = end-start

    start = time.time()

    num = N[i] # This is the number of eigenvalues and eigenvectors you want
    sigma = 0 # This is the eigenvalue we search around
    which = 'LM'
    spLA.eigsh(H_sparse, k = num, sigma = sigma, which = which)

    end = time.time()
    t_sparse[i] = end-start

N = np.array(N)
N = N**2

plt.scatter(N, t_sparse, c = 'b', label = 'Scipy')
plt.scatter(N, t_dense, c='r', label = 'Numpy')
plt.xlabel('Number of Lattice Sites')
plt.ylabel('Total Time [s]')
plt.title('Creation and Diagonalization of Hamiltonians')
plt.legend()
