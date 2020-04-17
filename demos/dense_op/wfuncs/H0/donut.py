import numpy as np
from numpy import linalg as LA

import majoranaJJ.etc.constants as const
import majoranaJJ.lattice.shapes as shps
import majoranaJJ.lattice.neighbors as nb
import majoranaJJ.operators.densOP as dop
import majoranaJJ.etc.plots as plots

R = 25
r = 10
ax = 10      #unit cell size along x-direction in [A]
ay = 10      #unit cell size along y-direction in [A]

coor = shps.donut(R, r) #donut coordinate array
NN = nb.NN_Arr(coor)
NNb = nb.Bound_Arr(coor)
print("lattice size", coor.shape[0])

""" This Hamiltonians is defined in operators/densOP.py. The basis is of spin up and spin down, so for a system without spin coupling the wavefunctions should only be different for every other excited state
"""

H = dop.H0(coor, ax, ay, NN)
print("H shape: ", H.shape)

energy, states = LA.eigh(H)

n = 4
plots.state_cmap(coor, energy, states, n = n, title = 'DENSE: State # {}'.format(n))
