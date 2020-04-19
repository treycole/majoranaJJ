import majoranaJJ.operators.sparse.qmsops as spop #sparse operators
import majoranaJJ.lattice.nbrs as nb #neighbor arrays
import majoranaJJ.lattice.shapes as shps #lattice shapes
import majoranaJJ.modules.plots as plots #plotting functions

x1 = 10
x2 = 10
y1 = 10
y2 = 10

coor = shps.cross(x1, x2, y1, y2)
NN = nb.NN_Arr(coor)
NNb = nb.Bound_Arr(coor)

idx = 1
plots.lattice(idx, coor, NN = NN)
plots.lattice(idx, coor, NNb = NNb)
