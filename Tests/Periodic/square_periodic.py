from os import path
import sys
sys.path.append(path.abspath('./Modules'))
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA

import lattice as lat
import constants as const
import operators as op

ax = 2e-10      #unit cell size along x-direction in [m]
ay = 2e-10      #unit cell size along y-direction in [m]
Ny = 3          #number of lattice sites in y direction
Nx = 3          #number of lattice sites in x direction
N = Ny*Nx
print(N)

coor = lat.square(Nx, Ny)       #square coordinate array
NN =  lat.NN_Arr(coor)          #nearest neighbor array of square lattice
NNb = lat.NN_Bound(NN, coor)    #periodic NN array

#checking eigenvalues
V = op.V_periodic(0.10, Nx, Ny, coor)
steps = 50
nbands = 18
qx = np.linspace(-np.pi/(Nx*ax), np.pi/(Nx*ax), steps)
qy = np.linspace(-np.pi/Ny, np.pi/Ny, steps)
eigarr = np.zeros((steps, nbands))
for i in range(steps):
    eigarr[i, :] = LA.eigh(op.H0k(qx[i], 0, coor, ax, ay))[0][:nbands]
op.bands(eigarr, qx)
plt.show()
