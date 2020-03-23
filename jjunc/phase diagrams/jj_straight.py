import numpy as np
#import matplotlib.pyplot as plt
from numpy import linalg as LA

import modules.constants as const
import modules.lattice as lat
import modules.operators as op
import modules.alt_mod.altoperators as aop

ax = 2      #Lattice spacing along x-direction in [A]
ay = 2      #Lattice spacing along y-direction in [A]

Wsc = 30 #width of the superconductor along the y-direction
Wj = 2 #width of the 2DEG junction along the y-diretion
Ny = 2*Wsc + Wj #Unit cell consists of 2 SC's and 1 Junction
Nx = 3  #number of lattice sites in x direction

N = Ny*Nx   #Total number of lattice sites


coor = lat.square(Nx, Ny)  #square coordinate array
NN =  lat.NN_Arr(coor)  #nearest neighbor array of square lattice
NNb = lat.NN_Bound(NN, coor)  #periodic NN array

#Method paramters
""" HBDG(coor, ax, ay,
    potential = 0,
    gammax = 0, gammay = 0, gammaz = 0,
    alpha = 0, qx = 0, qy = 0,
    periodic = 'yes'
    ):
"""
"""V_periodic(V0, Nx, Ny, coor)"""

steps = 60
a = 0.0   #Spin-Orbit Coupling constant: [eV*A]
gamma = np.linspace(0, .02, steps)  #Zeeman field energy contribution: [T]
delta = 0.0 #Superconducting Gap: [eV]
V = op.V_periodic(0, coor) #Amplitude of potential : [eV]
mu = 0.114 #Chemical Potential: [eV]

eigarr = np.zeros((steps, 2))
for i in range(steps):
    print(steps - i)
    eigarr[i, :] = np.sort( LA.eigh(op.HBDG(coor, ax, ay, Wsc, Wj, mu = mu, delta = delta, phi = phi, alpha = a, gammax = gamma[i],
    potential = V, qx = 0, periodicx = 'yes'))[0] )[2*N - 1: 2*N + 1]

op.phase(eigarr, gamma, xlabel = 'Gamma', ylabel = 'E(kx = 0)')
