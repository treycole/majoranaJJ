import majoranaJJ.etc.constants as const
import majoranaJJ.lattice.shapes as shps
import majoranaJJ.lattice.neighbors as nb
import majoranaJJ.etc.plots as plots

xbase = 40
xcut = 5
y1 = 10
y2 = 10

coor = shps.ibeam(xbase, xcut, y1, y2)
NN = nb.NN_Arr(coor)
NNb = nb.Bound_Arr(coor)

idx = 1
plots.lattice(idx, coor, NN = NN)
plots.lattice(idx, coor, NNb = NNb)
