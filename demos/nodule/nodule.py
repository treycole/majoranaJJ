import majoranaJJ.operators.sparse_operators as spop #sparse operators
import majoranaJJ.lattice.nbrs as nb #neighbor arrays
import majoranaJJ.lattice.shapes as shps #lattice shapes
import majoranaJJ.modules.plots as plots #plotting functions
import numpy as np

Nx = 12
Wj = int(1000/50)  #Junction width
Ny = Wj+2
cutx = 4 #nodule size in x
cuty = 8 #nodule size in y

coor = shps.square(Nx, Ny)
NN = nb.NN_Arr(coor)
NNb = nb.Bound_Arr(coor)

D_test = spop.Delta(coor, Wj = Wj, delta = 0.3, phi = np.pi, cutxT = cutx, cutyT = cuty, cutxB = cutx+6, cutyB = cuty-4)
plots.junction(coor, D_test)
