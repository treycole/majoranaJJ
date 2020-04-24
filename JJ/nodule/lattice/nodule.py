import majoranaJJ.operators.sparse.qmsops as spop #sparse operators
import majoranaJJ.lattice.nbrs as nb #neighbor arrays
import majoranaJJ.lattice.shapes as shps #lattice shapes
import majoranaJJ.modules.plots as plots #plotting functions

Nx = 16
Ny = 16

Wj = 6  #Junction region
Sx = 5
cutx = 6
cuty = 2

coor = shps.square(Nx, Ny)
NN = nb.NN_Arr(coor)
NNb = nb.Bound_Arr(coor)

D = spop.Delta(coor, Wj = Wj, delta = 0.3, Sx = Sx, cutx = cutx, cuty = cuty)
plots.nodule(coor, D, savenm = 'nodule.jpg')
