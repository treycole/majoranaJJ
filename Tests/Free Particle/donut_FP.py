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

R = 25
r = 10
donut = lat.donut(R, r) #donut coordinate array
NN_d = lat.NN_Arr(donut) #nearest neighbor array for donut

H = op.H0(donut, ax, ay)    #free particle Hamiltonian
energy, states = LA.eigh(H) #energy eigenvalues and eigenvectors of donut lattice

#Donut Eigenvalues
print(energy.shape)          #number of eigenvalues, or dimension of the Hilbert space
print(energy[0:10]/energy[0])  #first 10 eigenvalue ratios

#Wavefunction
op.state_cplot(donut, states[:, 36])   #36th excited state 
