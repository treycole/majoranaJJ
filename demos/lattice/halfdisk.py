import majoranaJJ.etc.constants as const
import majoranaJJ.lattice.shapes as shps
import majoranaJJ.lattice.neighbors as nb
import majoranaJJ.etc.plots as plots

R = 20

coor = shps.halfdisk(R)
NN = nb.NN_Arr(coor)
NNb = nb.Bound_Arr(coor)

idx = 0
plots.lattice(idx, coor, NN = NN)
plots.lattice(idx, coor, NNb = NNb)
