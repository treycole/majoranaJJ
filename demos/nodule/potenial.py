import majoranaJJ.operators.sparse.potentials as potentials #sparse operators
import majoranaJJ.lattice.nbrs as nb #neighbor arrays
import majoranaJJ.lattice.shapes as shps #lattice shapes
import majoranaJJ.modules.plots as plots #plotting functions

Nx = 20
Ny = 20
Wj = 6  #Junction width
cutx = 3 #nodule size in x
cuty = 3 #nodule size in y

coor = shps.square(Nx, Ny)
NN = nb.NN_Arr(coor)
NNb = nb.Bound_Arr(coor)

V0 = -5
V = potentials.Vjj(coor, Wj, Vsc = 0, Vj = V0, cutx = cutx, cuty = cuty)

plots.potential_profile(coor, V)
