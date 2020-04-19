import majoranaJJ.operators.sparse.qmsops as spop #sparse operators
import majoranaJJ.lattice.nbrs as nb #neighbor arrays
import majoranaJJ.lattice.shapes as shps #lattice shapes
import majoranaJJ.modules.plots as plots #plotting functions

R = 20

coor = shps.halfdisk(R)
NN = nb.NN_Arr(coor)
NNb = nb.Bound_Arr(coor)

idx = 0
plots.lattice(idx, coor, NN = NN)
plots.lattice(idx, coor, NNb = NNb)
