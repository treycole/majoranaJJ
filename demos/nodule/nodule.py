import majoranaJJ.operators.sparse.qmsops as spop #sparse operators
import majoranaJJ.lattice.nbrs as nb #neighbor arrays
import majoranaJJ.lattice.shapes as shps #lattice shapes
import majoranaJJ.modules.plots as plots #plotting functions

Nx = 50
Ny = 50
ax = 50 #lattice spacing in x-direction: [A]
ay = 50 #lattice spacing in y-direction: [A]
Wj = 15  #Junction region
Sx = 20
cutx = 10
cuty = 5

coor = shps.square(Nx, Ny)
NN = nb.NN_Arr(coor)
NNb = nb.Bound_Arr(coor)

D = spop.Delta(coor, Wj = Wj, delta = 0.3, Sx = Sx, cutx = cutx, cuty = cuty)
plots.nodule(coor, D, savenm = 'nodule.jpg')
