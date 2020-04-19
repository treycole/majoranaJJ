import time
import numpy as np

import majoranaJJ.lattice.nbrs as nb #neighbor arrays
import majoranaJJ.lattice.shapes as shps #lattice shapes
import majoranaJJ.modules.plots as plots #plotting functions

#Compared packages
import majoranaJJ.operators.sparse.qmsops as spop #sparse operators
import majoranaJJ.operators.dense.qmdops as dpop #dense operators
print(" ")

Nx = 50 #Number of lattice sites allong x-direction
Ny = 50 #Number of lattice sites along y-direction
ax = 2 #lattice spacing in x-direction: [A]
ay = 2 #lattice spacing in y-direction: [A]

coor = shps.square(Nx, Ny) #square lattice
NN = nb.NN_Arr(coor) #neighbor array
NNb = nb.Bound_Arr(coor) #boundary array

start = time.time()

H_dense = dpop.H0(coor, ax, ay, NN)

end = time.time()
print("DENSE (Numpy) construction time for size {} = {}".format(H_dense.shape, end-start),"[s]")
print("----------")

start = time.time()

H_sparse = spop.H0(coor, ax, ay, NN)

end = time.time()
print("SPARSE (Scipy) construction time for size {} = {}".format(H_sparse.shape, end-start), "[s]")
