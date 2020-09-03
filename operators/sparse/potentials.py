import scipy.sparse as sparse
import sys

def junction_geometry_check(Ny, Nx, Wj, cutx, cuty):
    while Wj >= Ny: #if juntion width is larger than the total size of unit cell then we must decrease it until it is smaller
        Ny -= 1

    if (Ny-Wj)%2 != 0: #Cant have even Ny and odd Wj, the top and bottom superconductors would then be of a different size
        if Ny - 1 > Wj:
            Ny -= 1
        else:
            Ny += 1

    if (Nx-cutx)%2 != 0: #Sx must be equal lengths on both sides
        if Nx - 1 > cutx:
            Nx -= 1
        else:
            Nx += 1

    while (2*cuty) >= Wj: #height of nodule cant be bigger than junction width
        cuty -= 1

    return Nx, Ny, cutx, cuty, Wj

def Vjj(coor, Wj, Vsc, Vj, cutx = 0, cuty = 0):

    N = coor.shape[0]
    Ny = int((max(coor[: , 1]) - min(coor[:, 1])) + 1) #number of lattice sites in y-direction, perpendicular to junction
    Nx = int((max(coor[: , 0]) - min(coor[:, 0])) + 1) #number of lattice sites in x-direction, parallel to junction
    row = []; col = []; data = []

    Nx, Ny, cutx, cuty, Wj = junction_geometry_check(Ny, Nx, Wj, cutx, cuty)

    Sx = int((Nx - cutx)/2) #length of either side of nodule, leftover length after subtracted nodule length divided by two
    Wsc = int((Ny - Wj)/2) #width of single superconductor

    for i in range(N):
        y = coor[i, 1]
        x = coor[i, 0]

        if y < Wsc: #if in bottom SC
            row.append(i); col.append(i)
            data.append(Vsc)

        if y >= (Wsc+Wj): #if in top SC
            row.append(i); col.append(i)
            data.append(Vsc)

        if y >= Wsc and y < (Wsc+Wj): #if coordinates in junction region
            if cuty != 0 and cutx != 0: #if there is a nodule present
                if (x >= Sx and x < (Sx + cutx)): #in x range of cut
                    if y >= ((Wsc + Wj) - cuty): #if in y range of cut along top interface, in top SC
                        row.append(i); col.append(i)
                        data.append(Vsc)
                    elif  y < (Wsc + cuty): #if in y range of cut along bottom interface, in bottom SC
                        row.append(i); col.append(i)
                        data.append(Vsc)
                    else: #site is in junction, out of y range
                        row.append(i); col.append(i)
                        data.append(Vj)
                else: #lattice site is in junction, out of x range
                    row.append(i); col.append(i)
                    data.append(Vj)
            else: #lattice site is in junction, no nodule
                row.append(i); col.append(i)
                data.append(Vj)
    #row.append(int(coor.shape[0]/4)); col.append(int(coor.shape[0])/4); data.append(V0) #random potential to break symmetry
    V = sparse.csc_matrix((data, (row, col)), shape = (N,N))
    return V
