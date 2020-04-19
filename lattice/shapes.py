import numpy as np
from numpy import sqrt

def square(Nx, Ny):
    N = Nx*Ny
    coor = np.zeros((N,2))
    for i in range(Nx):
        for j in range(Ny):
            n = i + Nx * j
            #x = i, y = j
            coor[n, 0] = i
            coor[n, 1] = j
    return coor

#Disk with a hole, inner radius r, outer radius R
def donut(R, r):
    CAx = []
    CAy = []

    xmin = -R #Radius of disk
    ymin = -R

    for j in range(int(2*R) + 1):
        for i in range(int(2*R) + 1):
            x = xmin + i
            y = ymin + j
            #decide if x,y is inside shape
            r_ij = sqrt(x**2 + y**2)
            if r_ij < R and r_ij >= r:
                CAx.append(i)
                CAy.append(j)

    coor_arr = np.zeros((len(CAx), 2))
    coor_arr[:, 0] = CAx
    coor_arr[:, 1] = CAy
    return coor_arr

def halfdisk(R):
    CAx = []
    CAy = []

    xmin = -R
    ymin = -R

    for j in range(2*R + 1):
        for i in range(2*R + 1):
            x = xmin + i
            y = ymin + j
            if(x < 0 or sqrt(x**2+y**2) > R):
                continue
            else:
                CAx.append(i)
                CAy.append(j)

    coor_arr = np.zeros((len(CAx), 2))
    coor_arr[:, 0] = CAx
    coor_arr[:, 1] = CAy
    return coor_arr

def ibeam(xbase, xcut, y1, y2):
    CAx = []
    CAy = []
    ybase = int(2*y1+y2)

    for j in range(ybase+1):
        for i in range(xbase+1):
            if (j > y1 and j < y1+y2) and (i < xcut or i > xbase - xcut):
                continue
            else:
                CAx.append(i)
                CAy.append(j)

    coor_arr = np.zeros((len(CAx), 2))
    coor_arr[:, 0] = CAx
    coor_arr[:, 1] = CAy
    return coor_arr

def cross(x1, x2, y1, y2):
    CAx = []
    CAy = []
    xbase = int(x1 + 2*x2)
    ybase = int(y1 + 2*y2)

    for j in range(ybase+1):
        for i in range(xbase+1):
            if (i < x2 and (j < y2 or j > y2+y1)) or (i > x1+x2 and (j < y2 or j > y2+y1)):
                continue
            else:
                CAx.append(i)
                CAy.append(j)

    coor_arr = np.zeros((len(CAx), 2))
    coor_arr[:, 0] = CAx
    coor_arr[:, 1] = CAy
    return coor_arr

""" Neighbor Arrays:
These neighbor arrays are implemented in such a way as to avoid double looping. This saves a significant ammount of time in large unit cells, as can be tested in the majoranaJJ/time_tsts/[bound_arr, nbr_arr]

Defining nearest neighbor array
NN_arr is Nx4, the columns store the index of the 4 nearest neighbors for each
lattice site
Left: NN[n,0] = n-1
Above: NN[n,1] = n+Nx
Right: NN[n, 2] = n+1
Down NN[n, 3] = n-Nx
if there is no lattice site in nearest neighbor spot, value is -1
"""
def NN_Arr(coor):
    N = coor.shape[0]
    NN = -1*ones((N,4), dtype = 'int')
    xmax = max(coor[:, 0])
    ymax = max(coor[:, 1])
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

""" Periodic Boundary conditions
if statements:
if the x-coordinate of the ith lattice site is the minimum value, it must be on the edge of the unit cell and therefore has a nearest neighbor in the neighboring unit cell.
Ex: To find the lattice site that corresponds to the neighbor to the left in the neighboring unit cell, we know it will be at most the (i + xmax)th site. If we are given a perfect square, it is the (i+ xmax)th site. In the case of the donut, this is not the case, so we until we find the site that is at the same height as the ith site, and has an x-coordinate that is the maximum value. The other statements follow similar logic for other neighbors.
"""
def Bound_Arr(coor):
    xmin = int(min(coor[:, 0]))
    ymin = int(min(coor[:, 1]))
    xmax = int(max(coor[:, 0]))
    ymax = int(max(coor[:, 1]))

    N = coor.shape[0]
    NNb = -1*ones((N,4), dtype = 'int') #stores the values of the coordinates of each periodic neighbor, -1 means no neighbor

    for i in range(N):
        x_index = coor[i, 0]
        y_index = coor[i, 1]
        if x_index == xmin:
            for j in range(i, N):
                y = coor[j, 1]
                x = coor[j, 0]
                if y == y_index and x == xmax:
                    NNb[i, 0] = j
                    break
        if y_index == ymax:
            for j in range(0, int(coor[i, 0]) + 1):
                x = coor[j, 0]
                y = coor[j, 1]
                if x == x_index and y == ymin:
                    NNb[i, 1] = j
                    break
        if x_index == xmax:
            for j in range(i, -1, -1):
                x = coor[j, 0]
                y = coor[j, 1]
                if y == y_index and x == xmin:
                    NNb[i, 2] = j
                    break
        if y_index == ymin:
            for j in range(N-1, int(coor[i, 0]), -1):
                x = coor[j, 0]
                y = coor[j, 1]
                if x == x_index and y == ymax:
                    NNb[i, 3] = j
                    break
    return NNb
