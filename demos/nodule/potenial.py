import majoranaJJ.operators.potentials as potentials #sparse operators
import majoranaJJ.lattice.nbrs as nb #neighbor arrays
import majoranaJJ.lattice.shapes as shps #lattice shapes
import majoranaJJ.modules.plots as plots #plotting functions

Nx = 12
Wj = int(1000/50)  #Junction width
Ny = Wj+2
cutx = 4 #nodule size in x
cuty = 8 #nodule size in y

coor = shps.square(Nx, Ny)
NN = nb.NN_Arr(coor)
NNb = nb.Bound_Arr(coor)

V0 = -5
V = potentials.Vjj(coor, Wj, Vsc = 0, Vj = V0, cutxT = cutx+6, cutyT = cuty, cutxB=cutx, cutyB=cuty)

plots.potential_profile(coor, V)
