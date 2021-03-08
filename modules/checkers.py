import numpy as np

#returns 1 if there is a ZEC at a specific mu value within a
#relevant range of gammax
#used for checking whether a certain mu value has a boundary along gamma,
#saves from checking top gap at every slice along gamma wrt mu
def boundary_check(eig_arr, gx, max_gam = 1.0, tol = 0.004):
    #eig_array.size = gx.size
    val = 0
    for i in range(eig_arr.shape[0]):
        if gx[i] >= max_gam:
            return 0
        elif eig_arr[i] <= tol:
            return 1
    return val

def junction_geometry_check(Nx, Ny, cutx, cuty, Wj):
    while Wj >= Ny: #if juntion width is larger than the total size of unit cell then we must decrease it until it is smaller
        Ny -= 1

    if (Ny-Wj)%2 != 0 and Wj!=0: #Cant have even Ny and odd Wj, the top and bottom superconductors would then be of a different size
        if Ny - 1 > Wj:
            Ny -= 1
        else:
            Ny += 1

    if (Nx-cutx)%2 != 0 and cutx!=0: #Sx must be equal lengths on both sides
        if Nx - 1 > cutx:
            Nx -= 1
        else:
            Nx += 1

    while (2*cuty) > Wj: #height of nodule cant be bigger than junction width
        cuty -= 1

    return Nx, Ny, cutx, cuty, Wj

def is_in_SC(i, coor, Wsc, Wj, cutxT, cutyT, cutxB, cutyB):
    Nx = int((max(coor[: , 0]) - min(coor[:, 0])) + 1) #number of lattice sites in x-direction, parallel to junction
    SxT = int((Nx - cutxT)/2) #length of either side of nodule, leftover length after subtracted nodule length divided by two
    SxB = int((Nx - cutxB)/2) #length of either side of nodule, leftover length after subtracted nodule length divided by two
    x = coor[i, 0]
    y = coor[i, 1]

    if y < Wsc: #in bottom SC
        return [True, 'B']
    if y >= (Wsc+Wj): #in top SC
        return [True, 'T']
    if y >= Wsc and y < (Wsc+Wj): #if coordinates in junction region
        if (x >= SxT and x < (SxT + cutxT)) and cutxT*cutyT != 0: #in x range of cut at top and nodule present there
            if y >= ((Wsc + Wj) - cutyT): #if in y range of cut along top interface, in top SC
                return [True, 'T']
        if (x >= SxB and x < (SxB + cutxB)) and cutxB*cutyB != 0:
            if y < (Wsc + cutyB): #if in y range of cut along bottom interface, in bottom SC
                return [True, 'B']
            else: #site is in junction, out of y range
                return [False,  None]
        else: #site is in junction, out of y range
            return [False,  None]
