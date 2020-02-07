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
ax = .1  #unit cell size along x-direction in [A]
ay = .1
Ny = 10     #number of lattice sites in y direction
Nx = 10     #number of lattice sites in x direction
N = Ny*Nx
####################################################
#donut creation, neighbor array, eigenvalues, eigenstates

donut = lat.donut(1.5, 0.3, ax, ay) #donut coordinate array
NN_d = lat.NN_Arr(donut, ax, ay) #nearest neighbor array for donut
E0_d = op.E0(donut, ax, ay) #energy eigenvalues of donut lattice
states_d = op.eigstate(donut, ax, ay) #energy eigenvectors of donut lattice

####################################################
#Lattice Testing

plt.scatter(donut[:,0], donut[:,1])
plt.show()

###################################################
#Nearest Neighbor Testing

print(NN_d.shape)
print(NN_d[100:125][:])
idx = 100
plt.scatter(donut[:, 0], donut[:, 1],c = 'b')
plt.scatter(donut[idx, 0], donut[idx, 1],c = 'r')
plt.scatter(donut[NN_d[idx,0],0], donut[NN_d[idx,0],1],c = 'g')
plt.scatter(donut[NN_d[idx,1],0], donut[NN_d[idx,1],1],c = 'magenta')
plt.scatter(donut[NN_d[idx,2],0], donut[NN_d[idx,2],1],c = 'purple')
plt.scatter(donut[NN_d[idx,3],0], donut[NN_d[idx,3],1],c = 'cyan')
plt.show()

###################################################
#Eigenvalue testing
print(E0_d.shape)
print(E0_d[0:100]/E0_d[0])
