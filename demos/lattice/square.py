import majoranaJJ.operators.sparse.qmsops as spop #sparse operators
import majoranaJJ.lattice.nbrs as nb #neighbor arrays
import majoranaJJ.lattice.shapes as shps #lattice shapes
import majoranaJJ.modules.plots as plots #plotting functions

Ny = 25    #number of lattice sites in y direction
Nx = 25     #number of lattice sites in x direction
N = Ny*Nx
print(N)

coor = shps.square(Nx, Ny) #square coordinate array
NN = nb.NN_Arr(coor) #nearest neighbor array of square lattice
NN2 = nb.NN_sqr(coor)
NNb = nb.Bound_Arr(coor)

Lx = int(max(coor[:, 0]) + 1)
idx = 2*Lx - 1
plots.lattice(idx, coor, NN = NN)
plots.lattice(idx, coor, NN = NN2)
plots.lattice(idx, coor, NNb = NNb)
