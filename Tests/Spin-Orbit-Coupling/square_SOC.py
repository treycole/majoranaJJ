from os import path
import sys
sys.path.append(path.abspath('./Modules'))

import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA

import lattice as lat
import constants as const
import operators as op

ax = 2  #unit cell size along x-direction in [A]
ay = 2
Ny = 25    #number of lattice sites in y direction
Nx = 25     #number of lattice sites in x direction
N = Ny*Nx
print(N)

coor = lat.square(Nx, Ny) #square coordinate array
NN =  lat.NN_Arr(coor) #nearest neighbor array of square lattice

gamma = 0.0 #Tesla
alpha = 0.1  #[ev*A]
H = op.H_SOC(coor, ax, ay, 0, gamma, alpha)

energy, states = LA.eigh(H)

#Square SOC Eigenvalues
print(energy.shape[0])
print(energy[0:10])
plt.scatter(np.arange(0, energy.shape[0], 1), energy)
plt.show()

op.state_cplot(coor, states[:, 0])
