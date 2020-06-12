import scipy.sparse as sparse
def V_BL(
    coor, Wj, V0, cutx = 0, cuty = 0
    ):

    N = coor.shape[0]
    Ny = (max(coor[: , 1]) - min(coor[:, 1])) + 1 #number of lattice sites in y-direction, perpendicular to junction
    Nx = (max(coor[: , 0]) - min(coor[:, 0])) + 1 #number of lattice sites in x-direction, parallel to junction
    row = []; col = []; data = []

    if Wj == 0: #If no junction, every site is superconducting, no phase diff
        print("No junction, this potential function will not work")
        return

    if (Ny-Wj)%2 != 0: #Cant have even Ny and odd Wj, the top and bottom superconductors would then be of a different size
        if Wj - 1 > 0:
            Wj -= 1
        else:
            Wj +=1

    if (Nx-cutx)%2 != 0: #Sx must be equal lengths on both sides
        if cutx - 1 > 0:
            cutx -= 1
        else:
            cutx += 1

    while (2*cuty) >= Wj: #height of nodule cant be bigger than junction width
        cuty -= 1

    while Wj >= Ny: #if juntion width is larger than the total size of unit cell then we must decrease it until it is smaller
        Wj -= 1

    Sx = int((Nx - cutx)/2) #length of either side of nodule, leftover length after subtracted nodule length divided by two
    Wsc = int((Ny - Wj)/2) #width of single superconductor

    for i in range(N):
        y = coor[i, 1]
        x = coor[i, 0]

        if y < Wsc: #if in bottom SC
            row.append(i); col.append(i)
            data.append(V0)

        if y >= (Wsc+Wj): #if in top SC
            row.append(i); col.append(i)
            data.append(V0)

        if y >= Wsc and y < (Wsc+Wj): #if coordinates in junction region
            if cuty != 0 and cutx !=0: #if there is a nodule present
                if (x >= Sx and x < (Sx + cutx)): #in x range of cut
                    if y >= ((Wsc + Wj) - cuty): #if in y range of cut along bottom interface, in bottom SC
                        row.append(i); col.append(i)
                        data.append(V0)
                    if  y < (Wsc + cuty) :#if in y range of cut along top interface, in top SC
                        row.append(i); col.append(i)
                        data.append(V0)
                    else: #site is in junction, out of y range
                        row.append(i); col.append(i)
                        data.append(0)
                else: #lattice site is in junction, out of x range
                    row.append(i); col.append(i)
                    data.append(0)
            else: #lattice site is in junction, no nodule
                row.append(i); col.append(i)
                data.append(0)
    #row.append(int(coor.shape[0]/4)); col.append(int(coor.shape[0])/4); data.append(V0)
    V = sparse.csc_matrix((data, (row, col)), shape = (N,N))
    return V
