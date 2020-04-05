import numpy as np
from numpy import linalg as LA

import majoranaJJ.modules.constants as const
import majoranaJJ.modules.lattice as lat
import majoranaJJ.modules.operators as op
import majoranaJJ.modules.plots as plot
import majoranaJJ.alternatives.hams as altH

ax = 2      #unit cell size along x-direction in [A]
ay = 2      #unit cell size along y-direction in [A]

R = 10
coor = lat.halfdisk(R)

H_original = altH.H0(coor, ax, ay)
"""
This is how the Hamiltonian was originally defined. If we wanted
periodicity we had to call a different Hamiltonian. The basis for this Hamiltonian doesn't account for spin. So as a check, the states for this Hamiltonian should be the same as the states in groups of 2 for "H"
"""

H = op.H0(coor, ax, ay)
Hperiodic = op.H0(coor, ax, ay, periodicx = 'yes', periodicy = 'yes')
"""
These Hamiltonians aer defined in modules/operators.py. In these Hamiltonians, the basis is of spin up and spin down, so for a system without spin coupling the wavefunctions should only be different for every n+2'th excited state

"""

energy, states = LA.eigh(H_original)
n = 3
plot.state_cplot(coor, states[:, n], title = 'Free particle, not periodic, no spin basis: {} excited state'.format(n))

energy, states = LA.eigh(H)
n = 6
plot.state_cplot(coor, states[:, n], title = 'Free particle not periodic: {} excited state'.format(n))

energy, states = LA.eigh(Hperiodic)
n = 24
plot.state_cplot(coor, states[:, n], title = 'Free particle, Periodic, Spin Basis: {}th excited state'.format(n))
