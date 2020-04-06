import majoranaJJ.etc.constants as const
import majoranaJJ.lattice.shapes as shps
import majoranaJJ.lattice.neighbors as nb
import majoranaJJ.etc.plots as plots

R = 25
r = 10

coor = shps.donut(R, r) #donut coordinate array
NN = nb.NN_Arr(coor)
NNk = nb.Bound_Arr(coor)

idx = 1
plots.lattice(idx, coor, NN = NN)
plots.lattice(idx, coor, NNb = NNb)
