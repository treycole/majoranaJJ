from numpy import linalg as LA

import majoranaJJ.operators.sparsOP as spop
import majoranaJJ.operators.densOP as dop
import majoranaJJ.lattice.neighbors as nb
import majoranaJJ.lattice.shapes as shps
import majoranaJJ.etc.plots as plots

ax = 10  #unit cell size along x-direction in [A]
ay = 10   #unit cell size along y-direction in [A]
Nx = 25
Ny = 25

coor = shps.square(Nx, Ny) #donut coordinate array
NN = nb.NN_Arr(coor)
NNb = nb.Bound_Arr(coor)
print("lattice size", coor.shape[0])


alpha = 0.0   #Spin-Orbit Coupling constant: [eV*A]
delta = 0 #superconducting gap: [eV]
gammaz = 0 #Zeeman field energy contribution: [T]
V0 = 0.0 #Amplitude of potential : [eV]
mu = 0 #Chemical Potential: [eV]

H_dense = dop.HBDG(coor, ax, ay, NN, Wsc = Ny, Wj = 0)
print("H shape: ", H_dense.shape)

"""States labeled starting from most negative energy"""
energy, states = LA.eigh(H_dense)
print(energy[0 : 5])
n1 = 0
n2 = int(H_dense.shape[0]/2)
plots.state_cmap(coor, energy, states, n = n1, title = 'DENSE: State # {}'.format(n))
plots.state_cmap(coor, energy, states, n = n2, title = 'DENSE: State # {}'.format(n2))
