import majoranaJJ.operators.sparsOP as spop
import majoranaJJ.lattice.neighbors as nb
import majoranaJJ.lattice.shapes as shps

Nx = 2
Ny = 2
coor = shps.square(Nx, Ny)
NN = nb.NN_Arr(coor)
NNb = nb.NN_Bound(coor)

delta = 0.3 #[eV]
Wsc = 2
Wj = 0

D = spop.Delta(coor, delta, Wsc, Wj).toarray()
print(D)
