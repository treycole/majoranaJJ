import scipy.sparse as sparse
def V_BL(
    coor, Wj, V0, Sx = None, cutx = None, cuty = None
    ):

    row = []; col = []; data = []
    N = coor.shape[0]
    Ny = (max(coor[: , 1]) - min(coor[:, 1])) + 1 #number of lattice sites in y-direction, perp to junction

    while Wj > Ny:
        Wj -= 1
    if (Ny-Wj)%2 != 0: #Cant have even Ny and odd Wj, odd number of lattice sites for proximty superconductors
        if (Wj + 1) <= Ny:
            Wj += 1
        else:
            Wj -=1

    Wsc = int((Ny - Wj)/2)
    for i in range(N):
        y = coor[i, 1]
        x = coor[i, 0]

        if y < Wsc: #if in bottom SC
            row.append(i); col.append(i)
            data.append(0)

        if y >= Wsc and y < (Wsc+Wj):
            if Sx is not None: #if there is a cut present
                if (2*Sx + cutx) != (max(coor[:, 0]) + 1):
                    print("Dimensions of nodule do not add up to lattice size along x direction")
                    return
                if (2*cuty) >= Wj:
                    print("Nodule extends across junction")
                    return
                if (x >= Sx and x < (Sx + cutx)) and (y < (Wsc + cuty) or y >= ((Wsc + Wj) - cuty)): #if in range of cut
                    row.append(i); col.append(i)
                    data.append(0)
            else: #lattice site is in junction
                row.append(i); col.append(i)
                data.append(V0)

        if y >= (Wsc+Wj):
            row.append(i); col.append(i)
            data.append(0)

    V = sparse.csc_matrix((data, (row, col)), shape = (N,N))
    return V
