import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA

import majoranas.modules.constants as const
import majoranas.modules.lattice as lat
import majoranas.modules.operators as op
import majoranas.modules.alt_mod.altoperators as aop

print("hbar = {} [J*s]".format(const.hbarJ))
print("hbar = {} [ev*s]".format(const.hbar))
print("mass of electron = {} [kg]".format(const.m0))
print("hbar**2/m0 = {} [eV A^2]".format(const.xi))

ax = 2      #atomic spacing along x-direction in [A]
ay = 2      #atomic spacing along y-direction in [A]

Nx = 3      #number of lattice sites in x direction
Ny = 3      #number of lattice sites in y direction
N = Ny*Nx   #Total number of lattice sites

coor = lat.square(Nx, Ny)       #square coordinate array
NN =  lat.NN_Arr(coor)          #nearest neighbor array of square lattice
NNb = lat.NN_Bound(NN, coor)    #periodic NN array

Lx = (max(coor[:, 0]) - min(coor[:, 0]) + 1)*ax  #Unit cell size in x-direction
Ly = (max(coor[:, 1]) - min(coor[:, 1]) + 1 )*ay  #Unit cell size in y-direction

tx = -const.xi/(ax**2) #Hopping in [eV]
ty = -const.xi/(ay**2) #Hopping in [eV]

print("Lx", Lx)
print("Number of Lattice Sites = ", N)
print("Unit cell size in x-direction = {} [A] = {} [m]".format(Lx, Lx*1e-10))
print("Unit cell size in y-direction = {} [A] = {} [m]".format(Ly, Ly*1e-10))
print("Hopping Parameter tx = {} [ev]".format(tx))
print("Hopping Parameter ty = {} [ev]".format(ty))

# H0(coor, ax, ay, potential = 0, gammax = 0, gammay = 0, gammaz = 0,
#    alpha = 0, qx = 0, qy = 0,
#    periodic = 'yes'
#    ):
#V_periodic(V0, Nx, Ny, coor)
a = 0.2   #[eV*A]
gamma = 0.2  #[T]
delta = 0.3
V0 = 0.0
mu = 0.3

steps = 50
nbands = 5
qx = np.linspace(-np.pi/Lx, np.pi/Lx, steps)
qy = np.linspace(-np.pi/Ly, np.pi/Ly, steps)
V = op.V_periodic(V0, coor)
eigarr = np.zeros((steps, 2*nbands))

for i in range(steps):
    eigarr[i, :] = np.sort( LA.eigh(op.HBDG(coor, ax, ay, mu = mu, delta = delta, alpha = a, gammaz = gamma, potential = V, qx = qx[i]))[0] )[2*N - nbands: 2*N + nbands]

print('size of bdg = {}'.format(np.size(op.HBDG(coor, ax, ay, delta = delta, alpha = a, gammaz = gamma, potential = V, qx = 0))) )
op.bands(eigarr, qx, Lx, Ly, title = 'Superconducting Spectrum'.format(a, gamma, V0))
