import numpy as np
from numpy import linalg as LA

import majoranaJJ.modules.constants as const
import majoranaJJ.modules.lattice as lat
import majoranaJJ.modules.operators as op
import majoranaJJ.modules.plots as plot
import majoranaJJ.alternatives.hams as altH


ax = 2  #Lattice spacing along x-direction in [A]
ay = 2  #Lattice spacing along y-direction in [A]

R = 15  #Number of sites of Outer Radius
r = 5  #Number of sites of Inner radius
coor = lat.donut(R, r) #donut coordinate array
NN =  lat.NN_Arr(coor) #nearest neighbor array of donut lattice

#H = op.H0(coor, ax, ay) + V
gamma = 0.1 #[T]
alpha = 0.1 #[eV]

H_original = altH.H_SOC(coor, ax, ay, 0, gamma, alpha) #zero potential
H_new = op.H0(coor, ax, ay, gammaz = gamma, alpha = alpha)

energy, states = LA.eigh(H_original)
n = 15
plot.state_cplot(coor, states[:, 15], title = 'Original SOC wavefunction for {} excited state'.format(n))

energy, states = LA.eigh(H_new)
n = 15
plot.state_cplot(coor, states[:, 15], title = 'New SOC wavefunction for {} excited state'.format(n))
