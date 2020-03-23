from numpy import ones

"""
These neighbor arrays are implemented in such a way as to avoid double looping. This saves a significant ammount of time in large unit cells, as can be tested in the majoranaJJ/time_tsts/[bound_arr, nbr_arr]
"""
#Defining nearest neighbor array
#NN_arr is Nx4, the columns store the index of the 4 nearest neighbors for each
#lattice site
#Left: NN[n,0] = n-1
#Above: NN[n,1] = n+Nx
#Right: NN[n, 2] = n+1
#Down NN[n, 3] = n-Nx
#if there is no lattice site in nearest neighbor spot, value is -1
def NN_Arr(coor):
    N = coor.shape[0]
    NN = -1*ones((N,4), dtype = 'int')
    xmax = max(coor[:,0])
    ymax = max(coor[:,1])
    Lx = xmax + 1
    Ly = ymax + 1

    for i in range(N):
        xi = coor[i, 0]
        yi = coor[i, 1]

        if (i-1) >= 0 and abs(xi - 1) >= 0 and abs(xi - coor[i-1, 0]) == 1 and abs(yi - coor[i-1, 1]) == 0:
            NN[i, 0] = i - 1
        if (i+1) < N and abs(xi + 1) <= xmax and abs(xi - coor[i+1, 0]) == 1 and abs(yi - coor[i+1, 1]) == 0:
            NN[i, 2] = i + 1
        for j in range(0, int(Lx)+1):
            if (i + j) < N and abs(yi + 1) <= ymax and abs(yi - coor[int(i + j), 1]) == 1 and abs(xi - coor[int(i + j), 0]) == 0:
                NN[i, 1] = i + j
            if (i - j) >= 0 and abs(yi - 1) >= 0 and abs(yi - coor[int(i - j), 1]) == 1 and abs(xi - coor[int(i - j), 0]) == 0:
                NN[i, 3]= i - j
    return NN

#Periodic Boundary conditions
def NN_Bound(coor):
    xmin = int(min(coor[:, 0])) #1
    ymin = int(min(coor[:, 1])) #29
    xmax = int(max(coor[:, 0]))
    ymax = int(max(coor[:, 1]))

    N = coor.shape[0]
    NNb = -1*ones((N,4), dtype = 'int') #stores the values of the coordinates of each periodic neighbor

    #if statements:
    #if the x-coordinate of the ith lattice site is the minimum value, it must be on the edge of the unit cell and therefore has a nearest neighbor in the neighboring unit cell. To find the lattice site that corresponds to this neighbor, we know it will be at most the (i + xmax)th site, if we are given a perfect square. In the case of the donut, this is not the case, so we iterate backwards until we find the site that is at the same height as the ith site, and has an x-coordinate that is the maximum value. The other statements follow similar logic.
    for i in range(N):
        if coor[i, 0] == xmin:
            for j in range(xmax, 0, -1):
                y = coor[i+j, 1]
                x = coor[i+j, 0]
                if y == coor[i,1] and x == xmax:
                    NNb[i, 0] = i+j
                    break
        if coor[i, 1] == ymax:
            for j in range(0, xmax+1):
                x = coor[j, 0]
                y = coor[j, 1]
                if x == coor[i,0] and y == ymin:
                    NNb[i, 1] = j
                    break
        if coor[i, 0] == xmax:
            for j in range(xmax+1, 0, -1):
                x = coor[i-j, 0]
                y = coor[i-j, 1]
                if y == coor[i,1] and x == xmin:
                    NNb[i, 2] = i-j
                    break
        if coor[i, 1] == ymin:
            for j in range(xmax+1, 0, -1):
                x = coor[N-j, 0]
                y = coor[N-j, 1]
                if x == coor[i,0] and y == ymax:
                    NNb[i, 3] = N-j
                    break
    return NNb
