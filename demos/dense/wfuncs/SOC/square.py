import time
from numpy import linalg as LA
import scipy.sparse.linalg as spLA

import majoranaJJ.operators.sparsOP as spop
import majoranaJJ.operators.densOP as dop
import majoranaJJ.lattice.neighbors as nb
import majoranaJJ.lattice.shapes as shps
import majoranaJJ.etc.plots as plot

Nx = 30
Ny = 30
ax = 2      #unit cell size along x-direction in [A]
ay = 2      #unit cell size along y-direction in [A]

coor = shps.square(Nx, Ny) #donut coordinate array
NN = nb.NN_Arr(coor)
NNb = nb.Bound_Arr(coor)
print("lattice size", coor.shape[0])

Wsc = Ny #Width of Superconductor
Wj = 0 #Width of Junction
Lx = (max(coor[:, 0]) - min(coor[:, 0]) + 1)*ax #Unit cell size in x-direction
Ly = (max(coor[:, 1]) - min(coor[:, 1]) + 1)*ay #Unit cell size in y-direction

""" This Hamiltonians is defined in operators/densOP.py. The basis is of spin up and spin down, so for a system without spin coupling the wavefunctions should only be different for every other excited state
"""

alpha = 0.0   #Spin-Orbit Coupling constant: [eV*A]
gammaz = 0 #Zeeman field energy contribution: [T]
V0 = 0.0 #Amplitude of potential : [eV]
mu = 0 #Chemical Potential: [eV]

H_dense = dop.H0(coor, ax, ay, NN, mu = mu, gammaz = gammaz)

start = time.time()
energy_dense, states_dense = LA.eigh(H_dense)
n = 0
plot.state_cmap(coor, energy_dense, states_dense, n = n, title = 'DENSE Free Particle Ground State')
end = time.time()
print("Time taken to plot probability map for lattice of size {} = {} [s]".format(coor.shape[0], end-start))
