import majoranaJJ.operators.sparse_operators as spop #sparse operators
import majoranaJJ.lattice.nbrs as nb #neighbor arrays
import majoranaJJ.lattice.shapes as shps #lattice shapes
import majoranaJJ.modules.plots as plots #plotting functions
import majoranaJJ.modules.checkers as check
import majoranaJJ.modules.constants as const
import scipy.sparse as sparse

import numpy as np

Nx = 2
Ny = 3
Wj = 1
ax = 1
ay = 1
cutx = 0 #nodule size in x
cuty = 0 #nodule size in y
Nx, Ny, cutx, cuty, Wj = check.junction_geometry_check(Ny, Nx, Wj, cutx, cuty)

coor = shps.square(Nx, Ny)
NN = nb.NN_Arr(coor)
NNb = nb.Bound_Arr(coor)
N = coor.shape[0] #number of lattice sites
I = sparse.identity(N) #identity matrix of size NxN
Nx = int((max(coor[: , 0]) - min(coor[:, 0])) + 1) #number of lattice sites in x-direction, parallel to junction
Ny = int((max(coor[: , 1]) - min(coor[:, 1])) + 1) #number of lattice sites in y-direction, perpendicular to junction
Sx = int((Nx - cutx)/2) #length of either side of nodule, leftover length after subtracted nodule length divided by two
Wsc = int((Ny - Wj)/2) #width of single superconductor
tx = 1/(ax**2)
print("tx = ", tx)
meff_sc = 2
meff_normal = 1

def xi(meff):
    return meff

diff_meff = False

if diff_meff:
    row = []; col = []; data = []
    for i in range(N):
        for j in range(N):
            row.append(i); col.append(j)
            inSC_i = check.is_in_SC(i, coor, Wsc, Wj, Sx, cutx, cuty)[0]
            inSC_j = check.is_in_SC(j, coor, Wsc, Wj, Sx, cutx, cuty)[0]
            if inSC_i or inSC_j:
                data.append((xi(meff_sc)))
            else:
                data.append((xi(meff_normal)))
    meff = sparse.csc_matrix((data, (row, col)), shape = (N,N))
elif not diff_meff:
    meff = xi(meff_normal)

k_x2 = spop.kx2(coor, ax, ay, NN)
k_y2 = spop.ky2(coor, ax, ay, NN)
H00 = (k_x2 + k_y2).multiply(meff)
spop.print_matrix(H00.toarray())
print(meff)
#spop.print_matrix(meff.toarray())
#print(H00.toarray())

#plots.potential_profile(coor, kx)
