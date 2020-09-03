import numpy as np
#returns 1 if there is a ZEC
#used for checking whether a certain mu value has a boundary along gamma
def boundary_check(eig_arr, gx, max_gam = 1.0, tol = 0.004):
    #eig_array.size = gx.size
    val = 0
    for i in range(eig_arr.shape[0]):
        if gx[i] >= max_gam:
            return 0
        elif eig_arr[i] <= tol:
            return 1
    return val
