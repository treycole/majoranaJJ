import majoranaJJ.operators.sparse.qmsops as spop #sparse operators
import majoranaJJ.lattice.nbrs as nb #neighbor arrays
import majoranaJJ.lattice.shapes as shps #lattice shapes
import majoranaJJ.modules.plots as plots #plotting functions

Nx = 20
Ny = 20
ax = 50 #lattice spacing in x-direction: [A]
ay = 50 #lattice spacing in y-direction: [A]
Wj = 6  #Junction region
cutx = 0
cuty = 0

coor = shps.square(Nx, Ny)
NN = nb.NN_Arr(coor)
NNb = nb.Bound_Arr(coor)

D_test = spop.Delta(coor, Wj = Wj, delta = 0.3, cutx = cutx, cuty = cuty)
plots.junction(coor, D_test, savenm = 'nodule.jpg')
