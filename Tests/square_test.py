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

#lattice viewing
plt.scatter(coor[:,0], coor[:,1])
plt.show()

#This is to visualize the array as points and see if the nearest neighbor array is working correctly
idx = 130
plt.scatter(coor[:,0],coor[:,1],c = 'b')
plt.scatter(coor[idx,0],coor[idx,1],c = 'r')
plt.scatter(coor[NN[idx,0],0], coor[NN[idx,0],1],c = 'g')
plt.scatter(coor[NN[idx,1],0], coor[NN[idx,1],1],c = 'magenta')
plt.scatter(coor[NN[idx,2],0], coor[NN[idx,2],1],c = 'purple')
plt.scatter(coor[NN[idx,3],0], coor[NN[idx,3],1],c = 'cyan')
plt.show()

#energy eigenvalues of square lattice
E0, states = op.diagH(coor, ax, ay)

#checking eigenvalues
print(np.shape(states[5]))
print(E0[0:10]/E0[0])
print (E0.shape)

op.state_cplot(coor, states[:, 5])
