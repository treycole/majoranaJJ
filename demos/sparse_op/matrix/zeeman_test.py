import majoranaJJ.operators.sparse_operators as spop #sparse operators
import majoranaJJ.lattice.nbrs as nb #neighbor arrays
import majoranaJJ.lattice.shapes as shps #lattice shapes
import majoranaJJ.modules.plots as plots #plotting functions
import majoranaJJ.modules.checkers as check
import scipy.sparse as sparse

import numpy as np

Nx = 13#130 #Number of lattice sites along x-direction
Ny = 8#80 #Number of lattice sites along y-direction
ax = 100 #lattice spacing in x-direction: [A]
ay = 100 #lattice spacing in y-direction: [A]
Wj = 2#20 #Junction region
cutx = 0 #width of nodule
cuty = 0 #height of nodule
Nx, Ny, cutx, cuty, Wj = check.junction_geometry_check(Ny, Nx, Wj, cutx, cuty)

Wsc = int((Ny - Wj)/2) #width of single superconductor
Sx = int((Nx - cutx)/2) #length of either side of nodule, leftover length after subtracted nodule length divided by two
print(Nx, Ny, Wsc, Sx, Wj)

coor = shps.square(Nx, Ny)
NN = nb.NN_Arr(coor)
NNb = nb.Bound_Arr(coor)

N = coor.shape[0]
I = sparse.identity(N) #identity matrix of size NxN
Zeeman_in_SC = False
if Zeeman_in_SC:
    I_Zeeman = I
if not Zeeman_in_SC:
    row = []; col = []; data = []
    for i in range(N):
        x = coor[i, 0]
        y = coor[i, 1]
        row.append(i); col.append(i)
        bool_inSC, which = check.is_in_SC(x, y, Wsc, Wj, Sx, cutx, cuty)
        if bool_inSC:
            data.append(0)
        else:
            data.append(1)
    I_Zeeman = sparse.csc_matrix((data, (row, col)), shape = (N,N))

H_G0 = spop.HBDG(coor, ax, ay, NN, NNb=NNb, Wj=Wj, cutx=cutx, cuty=cuty, gammaz=0, qx=0, Tesla = False, Zeeman_in_SC=False, SOC_in_SC=False) #Matrix that consists of everything in the Hamiltonian except for the Zeeman energy in the x-direction
H_G1 = spop.HBDG(coor, ax, ay, NN, NNb=NNb, Wj=Wj, cutx=cutx, cuty=cuty, gammaz=1, qx=0, Tesla = False, Zeeman_in_SC=False, SOC_in_SC=False) #Hamiltonian with ones on Zeeman energy along x-direction sites

HG = H_G1 - H_G0 #the proporitonality matrix for gamma-x, it is ones along the sites that have a gamma value
print()
spop.print_matrix(I_Zeeman.toarray())
spop.print_matrix(HG.toarray())

plots.potential_profile(coor, I_Zeeman.toarray())
