from numpy import linalg as LA

import majoranaJJ.operators.densOP as dop
import majoranaJJ.lattice.neighbors as nb
import majoranaJJ.lattice.shapes as shps
import majoranaJJ.etc.plots as plot

ax = 10   #unit cell size along x-direction in [A]
ay = 10   #unit cell size along y-direction in [A]
Nx = 50
Ny = 50

coor = shps.square(Nx, Ny) #donut coordinate array
NN = nb.NN_Arr(coor)
NNb = nb.Bound_Arr(coor)
print("lattice size", coor.shape[0])

""" This Hamiltonians is defined in operators/densOP.py. The basis is of spin up and spin down, so for a system without spin coupling the wavefunctions should only be different for every other excited state
"""

alpha = 0.0   #Spin-Orbit Coupling constant: [eV*A]
gammaz = 0 #Zeeman field energy contribution: [T]
V0 = 0.0 #Amplitude of potential : [eV]
mu = 0 #Chemical Potential: [eV]

H_dense = dop.H0(coor, ax, ay, NN, mu = mu, gammaz = gammaz, alpha = alpha)
print("H shape: ", H_dense.shape)

energy_dense, states_dense = LA.eigh(H_dense)
n = 0
plot.state_cmap(coor, energy_dense, states_dense, n = 0, title = 'DENSE Free Particle Ground State')
plot.state_cmap(coor, energy_dense, states_dense, n = n, title = 'DENSE: State # {}'.format(n))
