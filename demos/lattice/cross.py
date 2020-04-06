import majoranaJJ.etc.constants as const
import majoranaJJ.lattice.shapes as shps
import majoranaJJ.lattice.neighbors as nb
import majoranaJJ.etc.plots as plots

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
