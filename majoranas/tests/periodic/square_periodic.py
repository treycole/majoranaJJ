from os import path
import sys
sys.path.append("...\majoranas")
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA

import majoranas.modules.constants as const
import majoranas.modules.lattice as lat
import majoranas.modules.operators as op

print("hbar = {} [J*s]".format(const.hbarJ))
print("hbar = {} [ev*s]".format(const.hbar))
print("mass of electron = {} [kg]".format(const.m0))
print("hbar**2/m0 = {} [eV A^2]".format(const.xi))

ax = 2      #atomic spacing along x-direction in [A]
ay = 2      #atomic spacing along y-direction in [A]

Nx = 3      #number of lattice sites in x direction
Ny = 3      #number of lattice sites in y direction
N = Ny*Nx   #Total number of lattice sites

Lx = Nx*ax  #Unit cell size in x-direction
Ly = Ny*ay  #Unit cell size in y-direction

tx = -const.xi/(ax**2) #Hopping in [eV]
ty = -const.xi/(ay**2) #Hopping in [eV]

print("Number of Lattice Sites= ", N)
print("Unit cell size in x-direction = {} [m] = {} [A]".format(Lx, Lx*1e10))
print("Unit cell size in y-direction = {} [m] = {} [A]".format(Ly, Ly*1e10))
print("Hopping Parameter tx = {} [ev]".format(tx*6.242e18))
print("Hopping Parameter ty = {} [ev]".format(ty*6.242e18))

coor = lat.square(Nx, Ny)       #square coordinate array
NN =  lat.NN_Arr(coor)          #nearest neighbor array of square lattice
NNb = lat.NN_Bound(NN, coor)    #periodic NN array

#checking eigenvalues
#H0k(qx, qy, coor, ax, ay)
steps = 50
nbands = 9
qx = np.linspace(-np.pi/Lx, np.pi/Lx, steps)
qy = np.linspace(-np.pi/Ly, np.pi/Ly, steps)
eigarr = np.zeros((steps, nbands))
for i in range(steps):
    eigarr[i, :] = LA.eigh(op.H0k(qx[i], 0, coor, ax, ay))[0][:nbands]

op.bands(eigarr, qx, Lx, Ly)

#H_SOC(coor, ax, ay, V, gamma, alpha)
#V_periodic(V0, Nx, Ny, coor)
alpha = 0.2   #[eV*A]
gamma = 0*0.01  #[T]
V0 = 0.1

steps = 50
nbands = 18
qx = np.linspace(-np.pi/Lx, np.pi/Lx, steps)
qy = np.linspace(-np.pi/Ly, np.pi/Ly, steps)
V = op.V_periodic(V0, Nx, Ny, coor)
eigarr = np.zeros((steps, nbands))

for i in range(steps):
    eigarr[i, :] = LA.eigh(op.H_SOCk(qx[i], 0, coor, ax, ay, V, gamma, alpha))[0][:nbands]

op.bands(eigarr, qx, Lx, Ly)
