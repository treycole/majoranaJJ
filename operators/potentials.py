import scipy.sparse as sparse
import sys
from majoranaJJ.modules.checkers import junction_geometry_check as jgc
from majoranaJJ.modules import checkers as check

def Vjj(coor, Wj, Vsc, Vj, cutxT = 0, cutyT = 0, cutxB = 0, cutyB = 0):
    N = coor.shape[0]
    Nx = int((max(coor[: , 0]) - min(coor[:, 0])) + 1) #number of lattice sites in x-direction, parallel to junction
    Ny = int((max(coor[: , 1]) - min(coor[:, 1])) + 1) #number of lattice sites in y-direction, perpendicular to junction
    row = []; col = []; data = []

    Wsc = int((Ny - Wj)/2) #width of single superconductor

    for i in range(N):
        row.append(i); col.append(i)
        bool_inSC, which = check.is_in_SC(i, coor, Wsc, Wj, cutxT=cutxT, cutyT=cutyT, cutxB=cutxB, cutyB=cutyB)
        if bool_inSC:
            data.append(Vsc)
        else:
            data.append(Vj)

    V = sparse.csc_matrix((data, (row, col)), shape = (N,N))
    return V
