import majoranaJJ.operators.sparse_operators as spop #sparse operators
import majoranaJJ.lattice.nbrs as nb #neighbor arrays
import majoranaJJ.lattice.shapes as shps #lattice shapes
import majoranaJJ.modules.plots as plots #plotting functions
import numpy as np

###################################################
ax = 1
ay = 1
alpha=1
Nx = 3
Ny = 1
#square lattice
coor = shps.square(Nx, Ny) #square lattice
NN = nb.NN_Arr(coor) #neighbor array
NNb = nb.Bound_Arr(coor) #boundary array

H = spop.H0(coor, ax, ay, NN, NNb = NNb,alpha=1, qx = 1)
print(H.todense())
