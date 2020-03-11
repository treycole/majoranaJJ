import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA

import modules.constants as const
import modules.lattice as lat
import modules.operators as op
import modules.alt_mod.altoperators as aop

ax = 2      #atomic spacing along x-direction in [A]
ay = 2      #atomic spacing along y-direction in [A]

Wsc = 30
Wj = 2
Ny = 2*Wsc + Wj
Nx = 3     #number of lattice sites in x direction

N = Ny*Nx   #Total number of lattice sites

coor = lat.square(Nx, Ny)       #square coordinate array
NN =  lat.NN_Arr(coor)          #nearest neighbor array of square lattice
NNb = lat.NN_Bound(NN, coor)    #periodic NN array
coor = lat.square(Nx, Ny)       #square coordinate array
NN =  lat.NN_Arr(coor)          #nearest neighbor array of square lattice
NNb = lat.NN_Bound(NN, coor)    #periodic NN array

Lx = (max(coor[:, 0]) - min(coor[:, 0]) + 1)*ax  #Unit cell size in x-direction
Ly = (max(coor[:, 1]) - min(coor[:, 1]) + 1)*ay  #Unit cell size in y-direction

tx = -const.xi/(ax**2) #Hopping in [eV]
ty = -const.xi/(ay**2) #Hopping in [eV]

print("Number of Lattice Sites = ", N)
print("Unit cell size in x-direction = {} [A] = {} [m]".format(Lx, Lx*1e-10))
print("Unit cell size in y-direction = {} [A] = {} [m]".format(Ly, Ly*1e-10))
print("Hopping Parameter tx = {} [ev]".format(tx))
print("Hopping Parameter ty = {} [ev]".format(ty))

# HBDG(coor, ax, ay,
#    potential = 0,
#    gammax = 0, gammay = 0, gammaz = 0,
#    alpha = 0, qx = 0, qy = 0,
#    periodic = 'yes'
#    ):
#V_periodic(V0, Nx, Ny, coor)
a = 0.0   #[eV*A]
gamma = 0.0  #[T]
delta = 0.0
V0 = 0.0
mu = 0.114

steps = 50
nbands = 5
qx = np.linspace(-np.pi/Lx, np.pi/Lx, steps)
qy = np.linspace(-np.pi/Ly, np.pi/Ly, steps)
V = op.V_periodic(V0, coor)
eigarr = np.zeros((steps, 2*nbands))

for i in range(steps):
    eigarr[i, :] = np.sort( LA.eigh(op.HBDG(coor, ax, ay, Wsc, Wj, mu = mu, delta = delta, alpha = a, gammaz = gamma,
    potential = V, qx = qx[i], periodicx = 'yes'))[0] )[2*N - nbands: 2*N + nbands]

print('Size of BDG Hamiltonian = {}'.format(np.shape(op.HBDG(coor, ax, ay, Wsc, Wj, delta = delta,
alpha = a, gammaz = gamma, potential = V, qx = 0))) )
print('Zero Point Energy = {}'.format(np.sort( LA.eigh(op.HBDG(coor, ax, ay, Wsc, Wj, qx = 0, periodicx = 'yes'))[0])[2*N + 1]))
op.bands(eigarr, qx, Lx, Ly, title = 'Superconducting Spectrum'.format(a, gamma, V0))
