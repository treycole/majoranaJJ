import numpy as np
def boundary_check(eig_arr, gx, max_gam = 1.0, tol = 0.004):
    #eig_array.size = gx.size
    val = 0
    for i in range(eig_arr.shape[0]):
        if gx[i] >= max_gam:
            return 0
        elif eig_arr[i] <= tol:
            return 1
    return val
    #check for zero energy mode, set top array at that mu value 1 instead of 0 if boundary check shows zero energy state below max gamma
