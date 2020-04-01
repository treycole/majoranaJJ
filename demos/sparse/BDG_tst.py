import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse.linalg as spLA
import majoranaJJ.operators.sparsOP as spop
import majoranaJJ.lattice.neighbors as nb
import majoranaJJ.lattice.shapes as shps

Nx=3
Ny=3
ax=2
ay=2
coor = shps.square(Nx, Ny)
NN = nb.NN_Arr(coor)
NNb = nb.NN_Bound(coor)

Wsc = 2
Wj = 2
Lx = (max(coor[:, 0]) - min(coor[:, 0]) + 1)*ax  #Unit cell size in x-direction
Ly = (max(coor[:, 1]) - min(coor[:, 1]) + 1)*ay 


alpha = 0.0   #Spin-Orbit Coupling constant: [eV*A]
gamma = 0.0  #Zeeman field energy contribution: [T]
delta = 0.0 #Superconducting Gap: [eV]
V0 = 0.0 #Amplitude of potential : [eV]
mu = 0.114 #Chemical Potential: [eV]

steps = 70 #Number of kx and ky values that are evaluated
nbands = 5 #Number of bands shown
qx = np.linspace(-np.pi/Lx, np.pi/Lx, steps) #kx in the first Brillouin zone
qy = np.linspace(-np.pi/Ly, np.pi/Ly, steps) #ky in the first Brillouin zone
V = op.V_periodic(V0, coor) #Periodic potential with same periodicity as the unit cell lattice sites


num = 5 # This is the number of eigenvalues and eigenvectors you want
sigma = 0 *0.001766 # This is the eigenvalue we search around
which = 'LM'
print("H shape: ", H.shape)
eigs,vecs = spLA.eigsh(H,k=num,sigma = sigma, which = which)
idx_sort = np.argsort(eigs)
eigs = eigs[idx_sort]
vecs = vecs[:,idx_sort]
print(eigs[0])