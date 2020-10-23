import majoranaJJ.operators.sparse_operators as spop #sparse operators
from majoranaJJ.operators.potentials import Vjj #potential JJ
import majoranaJJ.lattice.nbrs as nb #neighbor arrays
import majoranaJJ.lattice.shapes as shps #lattice shapes
import majoranaJJ.modules.plots as plots #plotting functions
import majoranaJJ.modules.gamfinder as gamfinder
from majoranaJJ.modules.checkers import boundary_check as bc
import majoranaJJ.modules.golden_search as gs
import majoranaJJ.modules.checkers as check

import numpy as np
import matplotlib.pyplot as plt
###################################################
#Defining System
Nx = 3 #Number of lattice sites along x-direction
Ny = 290 #Number of lattice sites along y-direction
ax = 50 #lattice spacing in x-direction: [A]
ay = 50 #lattice spacing in y-direction: [A]
Wj = 40 #Junction region
cutx = 0 #width of nodule
cuty = 0 #height of nodule
Nx, Ny, cutx, cuty, Wj = check.junction_geometry_check(Nx, Ny, cutx, cuty, Wj)
print("Nx = {}, Ny = {}, cutx = {}, cuty = {}, Wj = {}".format(Nx, Ny, cutx, cuty, Wj))

Junc_width = Wj*ay*.10 #nm
SC_width = ((Ny - Wj)*ay*.10)/2 #nm
Nod_widthx = cutx*ax*.1 #nm
Nod_widthy = cuty*ay*.1 #nm
print("Nodule Width in x-direction = ", Nod_widthx, "(nm)")
print("Nodule Width in y-direction = ", Nod_widthy, "(nm)")
print("Junction Width = ", Junc_width, "(nm)")
print("Supercondicting Lead Width = ", SC_width, "(nm)")
###################################################coor = shps.square(Nx, Ny) #square lattice
coor = shps.square(Nx, Ny) #square lattice
NN = nb.NN_Arr(coor) #neighbor array
NNb = nb.Bound_Arr(coor) #boundary array
lat_size = coor.shape[0]
print("Lattice Size: ", lat_size)

Lx = (max(coor[:, 0]) - min(coor[:, 0]) + 1)*ax #Unit cell size in x-direction
Ly = (max(coor[:, 1]) - min(coor[:, 1]) + 1)*ay #Unit cell size in y-direction
###################################################
#Hamiltonian Parameters
alpha = 100 #Spin-Orbit Coupling constant: [meV*A]
gamx = 1 #parallel to junction: [meV]
phi = np.pi #SC phase difference
delta = 1.0 #Superconducting Gap: [meV]
mu = 2.1 #Chemical Potential: [meV]
#####################################
E_min, mu = gs.mu_scan_2(coor, ax, ay, NN, 5.0, 6.0, NNb=NNb, Wj=Wj, cutx=cutx, cuty=cuty, gamx=gamx, alpha=alpha, delta=delta, phi=phi, Vj=0)

plt.plot(mu, E_min)
plt.show()
