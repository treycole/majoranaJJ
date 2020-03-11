import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA

import modules.constants as const
import modules.lattice as lat
import modules.operators as op
import modules.alt_mod.altoperators as aop

ax = 2      #atomic spacing along x-direction in [A]
ay = 2      #atomic spacing along y-direction in [A]

Wsc = 30
Wj = 2
Ny = 2*Wsc + Wj
Nx = 3     #number of lattice sites in x direction

N = Ny*Nx   #Total number of lattice sites

coor = lat.square(Nx, Ny)       #square coordinate array
NN =  lat.NN_Arr(coor)          #nearest neighbor array of square lattice
NNb = lat.NN_Bound(NN, coor)    #periodic NN array

# HBDG(coor, ax, ay,
#    potential = 0,
#    gammax = 0, gammay = 0, gammaz = 0,
#    alpha = 0, qx = 0, qy = 0,
#    periodic = 'yes'
#    ):
#V_periodic(V0, Nx, Ny, coor)
a = 0.0   #[eV*A]
delta = 0.01
V0 = 0.0
mu = 0.114
phi = 0

steps = 60
nbands = 2
gamma = np.linspace(0, .02, steps) #[T]
V = op.V_periodic(V0, coor)
eigarr = np.zeros((steps, 2))

for i in range(steps):
    print(steps - i)
    eigarr[i, :] = np.sort( LA.eigh(op.HBDG(coor, ax, ay, Wsc, Wj, mu = mu, delta = delta, phi = phi, alpha = a, gammax = gamma[i],
    potential = V, qx = 0, periodicx = 'yes'))[0] )[2*N - 1: 2*N + 1]
op.phase(eigarr, gamma, xlabel = 'Gamma', ylabel = 'E(kx = 0)')
