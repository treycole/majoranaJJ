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
NNk = lat.NN_Bound(NN, coor)

idx = 1
plt.scatter(coor[:,0],coor[:,1],c = 'b')
plt.scatter(coor[idx,0],coor[idx,1],c = 'r')

if NN[idx, 0] != -1:
    plt.scatter(coor[NN[idx, 0], 0], coor[NN[idx, 0], 1], c = 'g')
if NN[idx, 1] != -1:
    plt.scatter(coor[NN[idx,1], 0], coor[NN[idx, 1], 1], c = 'magenta')
if NN[idx, 2] != -1:
    plt.scatter(coor[NN[idx,2], 0], coor[NN[idx, 2], 1], c = 'purple')
if NN[idx, 3] != -1:
    plt.scatter(coor[NN[idx,3], 0], coor[NN[idx, 3], 1], c = 'cyan')
plt.show()

idx = 15
print(NN[idx, 0], NN[idx, 1], NN[idx, 2], NN[idx, 3])
plt.scatter(coor[:, 0],coor[:, 1] ,c = 'b')
plt.scatter(coor[idx, 0],coor[idx, 1], c = 'r')
if NNk[idx, 0] != -1:
    plt.scatter(coor[NNk[idx, 0], 0], coor[NNk[idx, 0], 1], c = 'g')
if NNk[idx, 1] != -1:
    plt.scatter(coor[NNk[idx,1], 0], coor[NNk[idx, 1], 1], c = 'magenta')
if NNk[idx, 2] != -1:
    plt.scatter(coor[NNk[idx,2], 0], coor[NNk[idx, 2], 1], c = 'purple')
if NNk[idx, 3] != -1:
    plt.scatter(coor[NNk[idx,3], 0], coor[NNk[idx, 3], 1], c = 'cyan')
plt.show()
