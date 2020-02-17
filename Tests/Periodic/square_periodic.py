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
Ny = 10    #number of lattice sites in y direction
Nx = 10     #number of lattice sites in x direction
N = Ny*Nx
print(N)

coor = lat.square(Nx, Ny) #square coordinate array
NN =  lat.NN_Arr(coor) #nearest neighbor array of square lattice
NNb = lat.NN_Bound(NN, coor) #periodic NN array

#checking eigenvalues
#V = op.V_periodic(2*ax, coor)
steps = 40
nbands = 40
qx = np.linspace(-np.pi/Nx, np.pi/Nx, steps)
qy = np.linspace(-np.pi/Ny, np.pi/Ny, steps)
eigarr = np.zeros((steps, nbands))
for i in range(steps):
    eigarr[i, :] = LA.eigh(op.H0k(qx[i], 0, coor, ax, ay))[0][:nbands]
for j in range(eigarr.shape[1]):
    plt.plot(qx, eigarr[:, j], c ='b', linestyle = 'solid')

plt.plot(np.linspace(-0.005, 0.005, 1000), 0*np.linspace(-0.005,0.005, 1000), c='k', linestyle='solid', lw=1)
plt.xlabel('k (1/A)')
plt.ylabel('Energy (meV)')
plt.show()
#plt.ion()
