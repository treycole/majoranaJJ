from os import path
import sys
sys.path.append(path.abspath('./Modules'))

import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA

import lattice as lat
import constants as const
import operators as op

ax = .1  #unit cell size along x-direction in [A]
ay = .1
Ny = 25    #number of lattice sites in y direction
Nx = 25     #number of lattice sites in x direction
N = Ny*Nx

R = 25
r = 10

coor = lat.donut(R, r) #donut coordinate array
NN =  lat.NN_Arr(coor) #nearest neighbor array of donut lattice

#checking eigenvalues
#H = op.H0(coor, ax, ay) + V
gamma = 0.1 #Zeeman field
alpha = 0.1 #SOC term
H = op.H_SOC(coor, ax, ay, 0, gamma, alpha) #zero potential
energy, states = LA.eigh(H)

#Donut Eigenvalues
print(energy.shape[0])
print(energy[0:10])
plt.scatter(np.arange(0, energy.shape[0], 1), energy)
plt.show()

op.state_cplot(coor, states[:, 15])
