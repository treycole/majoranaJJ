import matplotlib.pyplot as plt

import majoranaJJ.etc.constants as const
import majoranaJJ.lattice.shapes as shp
import majoranaJJ.lattice.neighbors as nb
import majoranaJJ.etc.plots as plots

Ny = 25    #number of lattice sites in y direction
Nx = 25     #number of lattice sites in x direction
N = Ny*Nx
print(N)

coor = shp.square(Nx, Ny) #square coordinate array
NN = nb.NN_Arr(coor) #nearest neighbor array of square lattice
NNb = nb.Bound_Arr(coor)

idx = 1
plots.lattice(idx, coor, NN = NN)
plots.lattice(idx, coor, NNb = NNb)
