import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
import matplotlib.lines as mlines
from scipy import interpolate

import majoranas.modules.constants as const
import majoranas.modules.lattice as lat
import majoranas.modules.operators as op
import majoranas.modules.alt_mod.altoperators as aop

ax = 2      #unit cell size along x-direction in [A]
ay = 2      #unit cell size along y-direction in [A]

R = 15
r = 5
coor = lat.donut(R, r) #donut coordinate array
NN_d = lat.NN_Arr(coor) #nearest neighbor array for donut
print(np.shape(coor[:,0]))

H = op.H0(coor, ax, ay, periodic = 'no')
Hp = op.H0(coor, ax, ay, periodic = 'yes')
"""
This is the new Hamiltonian redefined in operators.py. In this Hamiltonian, we are in the basis
of spin up and spin down, so for a free particle the wavefunctions should be the same for every other 2
states. When the hamiltonian is defined this way with only (coor, ax, ay) as arguments the Hamiltonian
is for a free particle becuase gammma, alpha are preset to 0. By telling the function (periodic = 'no') we
aren't considering nearest neighbor hopping to the next cell, so the kx and ky operators are defined differently.

"""

H_original = aop.H0(coor, ax, ay)
"""
This is how the Hamiltonian was originally defined, separate from the periodic case. If we wanted
periodicity we had to call a different Hamiltonian. The basis for this Hamiltonian doesn't account for Spin
so as a check the states for this Hamiltonian should be the same as the states in groups of 2 for "H"
"""

energy, statesnp = LA.eigh(H)
energy2, statesp = LA.eigh(Hp)
energy3, stateso = LA.eigh(H_original)

#Donut Eigenvalues
print(energy.shape)
print("states are double degenerate w/o Zeeman field", energy[0:10]/energy[0])

nnp = 6
no = 3
np = 12
op.state_cplot(coor, statesp[:, np], title = 'Free particle, PBC, spin up/down basis: {}th excited state'.format(np))
op.state_cplot(coor, statesnp[:, nnp], title = 'Free particle no PBC: {} excited state'.format(nnp))
op.state_cplot(coor, stateso[:, no], title = 'Free particle, no PBS, no spin basis: {} excited state'.format(no))
