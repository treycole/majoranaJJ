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

def is_in_SC(x, y, Wsc, Wj, Sx, cutx, cuty):
    if y < Wsc: #in bottom SC
        return [True, 'B']

    if y >= (Wsc+Wj): #in top SC
        return [True, 'T']

    if y >= Wsc and y < (Wsc+Wj): #if coordinates in junction region
        if cuty != 0 and cutx !=0: #if there is a nodule present
            if (x >= Sx and x < (Sx + cutx)): #in x range of cut
                if y >= ((Wsc + Wj) - cuty): #if in y range of cut along top interface, in top SC
                    return [True, 'T']
                elif  y < (Wsc + cuty): #if in y range of cut along bottom interface, in bottom SC
                    return [True, 'B']
                else: #site is in junction, out of y range
                    return [False,  None]
            else: #lattice site is in junction, out of x range
                return [False, None]
        else: #lattice site is in junction, no nodule
            return [False, None]
