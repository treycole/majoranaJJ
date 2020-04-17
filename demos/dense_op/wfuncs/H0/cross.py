from numpy import linalg as LA

import majoranaJJ.operators.sparsOP as spop
import majoranaJJ.operators.densOP as dop
import majoranaJJ.lattice.neighbors as nb
import majoranaJJ.lattice.shapes as shps
import majoranaJJ.etc.plots as plots

ax = 10 #unit cell size along x-direction in [A]
ay = 10 #unit cell size along y-direction in [A]
x1 = 15
x2 = 15
y1 = 15
y2 = 15

coor = shps.cross(x1, x2, y1, y2)
NN = nb.NN_Arr(coor)
NNb = nb.Bound_Arr(coor)
print("lattice size", coor.shape[0])

alpha = 0.0   #Spin-Orbit Coupling constant: [eV*A]
gammaz = 0 #Zeeman field energy contribution: [T]
V0 = 0.0 #Amplitude of potential : [eV]
mu = 0 #Chemical Potential: [eV]

H_dense = dop.H0(coor, ax, ay, NN, mu = mu, gammaz = gammaz, alpha = alpha)
print("H shape: ", H_dense.shape)

energy_dense, states_dense = LA.eigh(H_dense)
n = 20
plots.state_cmap(coor, energy_dense, states_dense, n = n, title = 'DENSE: State # {}'.format(n))
