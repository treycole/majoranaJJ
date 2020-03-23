import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA

import modules.constants as const
import modules.lattice as lat
import modules.operators as op
import modules.alt_mod.altoperators as aop

ax = 2  #atomic spacing along x-direction in [A]
ay = 2  #atomic spacing along y-direction in [A]

Wsc = 30 #width of the superconductor along the y-direction
Wj = 2 #width of the 2DEG junction along the y-diretion
Ny = 2*Wsc + Wj #JJ consists of 2 SC and 1 Junction
Nx = 3    #number of lattice sites in x direction

N = Ny*Nx   #Total number of lattice sites

coor = lat.square(Nx, Ny)       #square coordinate array
NN =  lat.NN_Arr(coor)          #nearest neighbor array of square lattice
NNb = lat.NN_Bound(NN, coor)    #periodic NN array

Lx = (max(coor[:, 0]) - min(coor[:, 0]) + 1)*ax  #Unit cell size in x-direction
Ly = (max(coor[:, 1]) - min(coor[:, 1]) + 1)*ay  #Unit cell size in y-direction
H_SizeTest = op.HBDG(coor, ax, ay, Wsc, Wj)

print("Number of Lattice Sites = ", N)
print('Size of BDG Hamiltonian = {}'.format(np.shape(H_SizeTest)))
print("Unit cell size in x-direction = {} [A] = {} [m]".format(Lx, Lx*1e-10))
print("Unit cell size in y-direction = {} [A] = {} [m]".format(Ly, Ly*1e-10))

#Method paramters
""" HBDG(coor, ax, ay,
    potential = 0,
    gammax = 0, gammay = 0, gammaz = 0,
    alpha = 0, qx = 0, qy = 0,
    periodic = 'yes'
    ):
"""
"""V_periodic(V0, Nx, Ny, coor)"""

alpha = 0.0   #Spin-Orbit Coupling constant: [eV*A]
gamma = 0.0  #Zeeman field energy contribution: [T]
delta = 0.0 #Superconducting Gap: [eV]
V0 = 0.0 #Amplitude of potential : [eV]
mu = 0.114 #Chemical Potential: [eV]

steps = 70 #Number of kx and ky values that are evaluated
nbands = 5 #Number of bands shown
qx = np.linspace(-np.pi/Lx, np.pi/Lx, steps) #kx in the first Brillouin zone
qy = np.linspace(-np.pi/Ly, np.pi/Ly, steps) #ky in the first Brillouin zone
V = op.V_periodic(V0, coor) #Periodic potential with same periodicity as the unit cell lattice sites

eigarr = np.zeros((steps, 2*nbands)) #
for i in range(steps):
    eigarr[i, :] = np.sort( LA.eigh(op.HBDG(coor, ax, ay, Wsc, Wj, mu = mu, delta = delta, alpha = alpha, gammaz = gamma,
    potential = V, qx = qx[i], periodicx = 'yes'))[0] )[2*N - nbands: 2*N + nbands]

print('Zero Point Energy = {}'.format(np.sort( LA.eigh(op.HBDG(coor, ax, ay, Wsc, Wj, qx = 0, periodicx = 'yes'))[0])[2*N + 1]))
op.bands(eigarr, qx, Lx, Ly, title = 'Superconducting Spectrum'.format(a, gamma, V0))
