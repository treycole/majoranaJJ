import numpy as np
from numpy import linalg as LA

import majoranaJJ.etc.constants as const
import majoranaJJ.lattice.shapes as shps
import majoranaJJ.lattice.neighbors as nb
import majoranaJJ.operators.densOP as dop
import majoranaJJ.etc.plots as plot

R = 15
r = 5
ax = 2      #unit cell size along x-direction in [A]
ay = 2      #unit cell size along y-direction in [A]

coor = shps.donut(R, r) #donut coordinate array
NN = nb.NN_Arr(coor)
NNb = nb.Bound_Arr(coor)

""" This Hamiltonians is defined in operators/densOP.py. The basis is of spin up and spin down, so for a system without spin coupling the wavefunctions should only be different for every other excited state
"""

H = dop.H0(coor, ax, ay, NN, NNb = None)
print("H shape: ", H.shape)

energy, states = LA.eigh(H)

n = 3
plots.state_cmap(coor, energy, states, n = n, title = 'DENSE: State # {}'.format(n))
