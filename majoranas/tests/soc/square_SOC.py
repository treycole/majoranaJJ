from os import path
import sys
sys.path.append(path.abspath('./Modules'))

import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA

import lattice as lat
import constants as const
import operators as op

ax = 2e-10  #unit cell size along x-direction in [m]
ay = 2e-10  #unit cell size along y-direction in [m]
Ny = 25     #number of lattice sites in y direction
Nx = 25     #number of lattice sites in x direction
N = Ny*Nx   #Total number of lattice sites
print(N)

coor = lat.square(Nx, Ny) #square coordinate array
NN =  lat.NN_Arr(coor) #nearest neighbor array of square lattice

#H_SOC(coor, ax, ay, V, gamma, alpha)
gamma = 0.01*const.evtoJ #[T] converted Telsa in units of J to eV
alpha = 0.02*1e-10*const.evtoJ #[J*m]
H = op.H_SOC(coor, ax, ay, 0, 0, alpha)
energy, states = LA.eigh(H)

#Square SOC Eigenvalues
print(energy.shape[0])
print(energy[0:10])
plt.scatter(np.arange(0, energy.shape[0], 1), energy*const.Jtoev)
plt.ylabel("Energy [eV]")
plt.show()

op.state_cplot(coor, states[:, 0])