import time
import numpy as np
from numpy import linalg as LA
import scipy.sparse.linalg as spLA
import matplotlib.pyplot as plt

import majoranaJJ.lattice.nbrs as nbrs #neighbor arrays
import majoranaJJ.lattice.shapes as shps #lattice shapes
import majoranaJJ.modules.plots as plots #plotting functions

#Compared packages
import majoranaJJ.operators.sparse.qmsops as spop #sparse operators
import majoranaJJ.operators.dense.qmdops as dpop #dense operators

print(" ")

N = range(3, 20)
t_sparse = np.zeros(len(N))
t_dense = np.zeros(len(N))
ax = 2
ay = 2

for i in range(0, len(N)):

    Nx = N[i] #incrementing size of lattice
    Ny = N[i] #incrementing size of lattice
    coor = shps.square(Nx, Ny) #creating coordinate array
    NN = nbrs.NN_Arr(coor) #nearest neighbor array
    NNb = nbrs.Bound_Arr(coor) #boundary array

    H_sparse = spop.H0(coor, ax, ay, NN) #creating sparse hamiltonian

    start = time.time() #Time start for numpy

    H_dense = dpop.H0(coor, ax, ay, NN) #creating dense hamiltonian
    eigs, vecs = LA.eigh(H_dense) #diagonalizing

    end = time.time() #Time end for numpy
    t_dense[i] = end-start #append time taken to diagonalize

    start = time.time() #Time start for scipy

    H_sparse = spop.H0(coor, ax, ay, NN) #creating sparse hamiltonian
    num = int(H_sparse.shape[0]/2) #Number of eigenvalues and eigenvectors you want
    sigma = 0 #This is the eigenvalue we search around
    which = 'LM' #Largest magnitude eigenvalues
    spLA.eigsh(H_sparse, k = num, sigma = sigma, which = which) #diagonlizing

    end = time.time() #time end for scipy
    t_sparse[i] = end-start #append time to sparse time array

N = np.array(N)
N = N**2 #total number of lattice sites is Nx*Ny or N^2

plt.scatter(N, t_sparse, c = 'b', label = 'Scipy')
plt.scatter(N, t_dense, c='r', label = 'Numpy')
plt.xlabel('Number of Lattice Sites')
plt.ylabel('Total Time [s]')
plt.title('Creation and Diagonalization of Hamiltonians')
plt.legend()
plt.savefig("np_vs_scipy_timePlot.png")
plt.show()
