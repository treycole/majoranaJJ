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
print()

ax = 2      #atomic spacing along x-direction in [A]
ay = 2      #atomic spacing along y-direction in [A]

tx = -const.xi/(ax**2) #Hopping in [eV]
ty = -const.xi/(ay**2) #Hopping in [eV]

R = 3 # Outer radius of the Donut
r = 1  #Inner radius of the Donut
coor = lat.donut(R, r)       #square coordinate array
NN =  lat.NN_Arr(coor)          #nearest neighbor array of square lattice
NNb = lat.NN_Bound(NN, coor)    #periodic NN array
N = np.shape(coor)[0]
#number of lattice sites in x/y direction, no longer Nx and Ny becuase of the way donut is defined
#try the lattice test for donut and you can see that Nx is 2*R - 1

Lx = (max(coor[:, 0]) - min(coor[:, 0]) + 1)*ax  #Unit cell size in x-direction
Ly = (max(coor[:, 1]) - min(coor[:, 1]) + 1 )*ay  #Unit cell size in y-direction
"""
 more general way of calculating Lx than in square case. Could also just say R*ax or R*ay

"""

print("Nx = (2*R-1) = ", (2*R-1))
print("Number of Lattice Sites N = ", N)
print("Unit cell size in x-direction = {} [A] = {} [m]".format(Lx, Lx*1e-10))
print("Unit cell size in y-direction = {} [A] = {} [m]".format(Ly, Ly*1e-10))
print()
print("Hopping Parameter tx = {} [ev]".format(tx))
print("Hopping Parameter ty = {} [ev]".format(ty))
print()

#H0k(qx, qy, coor, ax, ay)
#H_SOk(coor, ax, ay, qx, qy, V, gamma, alpha)
#H0(coor, ax, ay, potential = 0, gammax = 0, gammay = 0, gammaz = 0,
#    alpha = 0, qx = 0, qy = 0,
#    periodic = 'yes'
#    ):
#V_periodic(V0, Nx, Ny, coor)

steps = 50
nbands_nosp = 5 #in no spin basis
nbands = 10  #in spin basis

alpha = 0.2   #[eV*A]
gamma = 0  #[T]
V0 = 0.0
V = op.V_periodic(V0, coor)

qx = np.linspace(-np.pi/Lx, np.pi/Lx, steps)
qy = np.linspace(-np.pi/Ly, np.pi/Ly, steps)

eigarrofp = np.zeros((steps, nbands_nosp)) #ofp ~ original free particle
eigarroso = np.zeros((steps, nbands)) #oso ~ original spin orbit
eigarrfp = np.zeros((steps, nbands))
eigarrsoc = np.zeros((steps, nbands))
for i in range(steps):
    eigarrofp[i, :] = LA.eigh(aop.H0k(coor, ax, ay, qx[i], 0))[0][:nbands_nosp]
    eigarroso[i, :] = LA.eigh(aop.H_SOk(coor, ax, ay, qx[i], 0, 0, gamma, alpha))[0][:nbands]
    eigarrfp[i, :] = LA.eigh(op.H0(coor, ax, ay, qx = qx[i]))[0][:nbands]
    eigarrsoc[i, :] = LA.eigh(op.H0(coor, ax, ay, alpha = alpha, gammaz = gamma, potential = V, qx = qx[i]))[0][:nbands]

op.bands(eigarrofp, qx, Lx, Ly, title = 'original FP')
op.bands(eigarrfp, qx, Lx, Ly, title = 'New FP')
op.bands(eigarroso, qx, Lx, Ly, title = " original SOC")
op.bands(eigarrsoc, qx, Lx, Ly, title = 'New SOC')
