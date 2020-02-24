from os import path
import sys
sys.path.append(path.abspath('./Modules'))

import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
import matplotlib.lines as mlines
from scipy import interpolate

import lattice as lat
import constants as const
import operators as op

ax = 2      #unit cell size along x-direction in [A]
ay = 2      #unit cell size along y-direction in [A]
Ny = 25     #number of lattice sites in y direction
Nx = 25     #number of lattice sites in x direction
N = Ny*Nx   #Total number of lattice sites
print(N)

R = 20

coor = lat.halfdisk(R)
NN = lat.NN_Arr(coor)
H0 = op.H0(coor, ax, ay)
energy, states = LA.eigh(H0)

#Ibeam Eigenvalues
print(energy.shape)
print(energy[0:10]/energy[0])

plt.xlim(0, max(coor[:, 0]))
plt.ylim(0, max(coor[:, 1]))
op.state_cplot(coor, states[:, 25])
