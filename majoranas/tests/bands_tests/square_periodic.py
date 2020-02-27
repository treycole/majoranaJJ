import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA

import majoranas.modules.constants as const
import majoranas.modules.lattice as lat
import majoranas.modules.operators as op
import majoranas.modules.alt_mod.altoperators as aop

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
print("Unit cell size in x-direction = {} [A] = {} [m]".format(Lx, Lx*1e-10))
print("Unit cell size in y-direction = {} [A] = {} [m]".format(Ly, Ly*1e-10))
print("Hopping Parameter tx = {} [ev]".format(tx))
print("Hopping Parameter ty = {} [ev]".format(ty))

coor = lat.square(Nx, Ny)       #square coordinate array
NN =  lat.NN_Arr(coor)          #nearest neighbor array of square lattice
NNb = lat.NN_Bound(NN, coor)    #periodic NN array

#H0k(qx, qy, coor, ax, ay)
steps = 50
nbands = N
qx = np.linspace(-np.pi/Lx, np.pi/Lx, steps)
qy = np.linspace(-np.pi/Ly, np.pi/Ly, steps)
eigarr = np.zeros((steps, nbands))
for i in range(steps):
    eigarr[i, :] = LA.eigh(aop.H0k(coor, ax, ay, qx[i], 0))[0][:nbands]

op.bands(eigarr, qx, Lx, Ly, title = 'original FP')

#H_SOk(coor, ax, ay, qx, qy, V, gamma, alpha)
#V_periodic(V0, Nx, Ny, coor)
alpha = 0.2   #[eV*A]
gamma = 0*0.01  #[T]
V0 = 0.0
V = op.V_periodic(V0, Nx, Ny, coor)

steps = 100
nbands = 2*N
qx = np.linspace(-np.pi/Lx, np.pi/Lx, steps)
qy = np.linspace(-np.pi/Ly, np.pi/Ly, steps)
eigarr = np.zeros((steps, nbands))
for i in range(steps):
    eigarr[i, :] = LA.eigh(aop.H_SOk(coor, ax, ay, qx[i], 0, 0, gamma, alpha))[0][:nbands]
op.bands(eigarr, qx, Lx, Ly, title = " original SOC")

# H0(coor, ax, ay, potential = 0, gammax = 0, gammay = 0, gammaz = 0,
#    alpha = 0, qx = 0, qy = 0,
#    periodic = 'yes'
#    ):
#V_periodic(V0, Nx, Ny, coor)
a = 0.2   #[eV*A]
gamma = 0*0.01  #[T]
V0 = 0

steps = 100
nbands = 2*N
qx = np.linspace(-np.pi/Lx, np.pi/Lx, steps)
qy = np.linspace(-np.pi/Ly, np.pi/Ly, steps)
V = op.V_periodic(V0, Nx, Ny, coor)
eigarrsoc = np.zeros((steps, nbands))
eigarrfp = np.zeros((steps, nbands))

for i in range(steps):
    eigarrsoc[i, :] = LA.eigh(op.H0(coor, ax, ay, alpha = a, gammaz = gamma, potential = V, qx = qx[i]))[0][:nbands]
for i in range(steps):
    eigarrfp[i, :] = LA.eigh(op.H0(coor, ax, ay, qx = qx[i]))[0][:nbands]

op.bands(eigarrsoc, qx, Lx, Ly, title = 'New SOC: alpha = {}, gammaz = {}, Potential = {}, qy = 0'.format(a, gamma, V0))
op.bands(eigarrfp, qx, Lx, Ly, title = 'New FP: alpha = {}, gammaz = {}, Potential = {}, qy = 0'.format(0, 0, 0))
