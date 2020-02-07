
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
import matplotlib.lines as mlines
from scipy import interpolate
####################################################
import lattice as lat
import constants as const
import operators as op
####################################################
ax = 100.0  #unit cell size along x-direction in [A]
ay = 100.0
Ny = 10     #number of lattice sites in y direction
Nx = 10     #number of lattice sites in x direction
N = Ny*Nx
####################################################

NN =   NN_Arr(coor, ax, ay)
idx = 24
plt.scatter(coor[:,0],coor[:,1],c = 'b')
plt.scatter(coor[idx,0],coor[idx,1],c = 'r')
plt.scatter(coor[NN[idx,0],0],coor[NN[idx,0],1],c = 'g')
plt.scatter(coor[NN[idx,1],0],coor[NN[idx,1],1],c = 'magenta')
plt.scatter(coor[NN[idx,2],0],coor[NN[idx,2],1],c = 'purple')
plt.scatter(coor[NN[idx,3],0],coor[NN[idx,3],1],c = 'cyan')
plt.show()
