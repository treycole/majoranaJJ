import majoranaJJ.operators.sparse.qmsops as spop #sparse operators
import majoranaJJ.lattice.nbrs as nb #neighbor arrays
import majoranaJJ.lattice.shapes as shps #lattice shapes
import majoranaJJ.modules.plots as plots #plotting functions

R = 25
r = 10

coor = shps.donut(R, r) #donut coordinate array
NN = nb.NN_Arr(coor)
NNb = nb.Bound_Arr(coor)

idx = 1
plots.lattice(idx, coor, NN = NN)
plots.lattice(idx, coor, NNb = NNb)
