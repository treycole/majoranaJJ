import majoranaJJ.operators.sparse_operators as spop #sparse operators
import majoranaJJ.lattice.nbrs as nb #neighbor arrays
import majoranaJJ.lattice.shapes as shps #lattice shapes
import majoranaJJ.modules.plots as plots #plotting functions
import majoranaJJ.modules.checkers as check

import numpy as np

Nx = 4
Ny = 5
Wj = 1
cutx = 0 #nodule size in x
cuty = 0 #nodule size in y
Nx, Ny, cutx, cuty, Wj = check.junction_geometry_check(Ny, Nx, Wj, cutx, cuty)

coor = shps.square(Nx, Ny)
NN = nb.NN_Arr(coor)
NNb = nb.Bound_Arr(coor)


kx = spop.kx(coor, 1, 1, NN, NNb = None, qx = None, Wj = Wj, cutx = cutx, cuty = cuty, SOC_in_SC = False)
spop.print_matrix(kx.toarray())

plots.potential_profile(coor, kx)
