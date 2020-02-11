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

ax = .1  #unit cell size along x-direction in [A]
ay = .1
Ny = 25    #number of lattice sites in y direction
Nx = 25     #number of lattice sites in x direction
N = Ny*Nx
print(N)

coor = lat.square(Nx, Ny) #square coordinate array
NN =  lat.NN_Arr(coor) #nearest neighbor array of square lattice

V = op.V_barrier(10000, 10, 15, coor)
print(V[11,11])

#checking eigenvalues
#H = op.H0(coor, ax, ay) + V
gamma = 0.01
alpha = 0.08
H = op.H_SOC(coor, ax, ay, V, gamma, alpha)
energy, states = LA.eigh(H)

op.state_cplot(coor, states[:, 100])
