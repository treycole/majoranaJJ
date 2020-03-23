import numpy as np

def square(Nx, Ny):
    N = Nx*Ny
    coor = np.zeros((N,2))
    for i in range(Nx):
        for j in range(Ny):
            n = i + Nx * j
            x = (i)
            y = (j)
            coor[n,0] = x
            coor[n,1] = y
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
            r_ij = np.sqrt((x)**2+(y)**2)
            if r_ij < R and r_ij >= r:
                CAx.append(i)
                CAy.append(j)

    coor_arr = np.zeros((len(CAx), 2))
    coor_arr[:, 0] = np.array(CAx)
    coor_arr[:, 1] = np.array(CAy)
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
            if(x < 0 or np.sqrt(x**2+y**2) > R):
                continue
            else:
                CAx.append(i)
                CAy.append(j)

    coor_arr = np.zeros((len(CAx), 2))
    coor_arr[:, 0] = np.array(CAx)
    coor_arr[:, 1] = np.array(CAy)
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
    coor_arr[:, 0] = np.array(CAx)
    coor_arr[:, 1] = np.array(CAy)
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
    coor_arr[:, 0] = np.array(CAx)
    coor_arr[:, 1] = np.array(CAy)
    return coor_arr

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
    NN = -1*np.ones((N,4), dtype = 'int')
    for n in range(N):
        for m in range(N):
            xn = coor[n, 0]
            xm = coor[m, 0]
            yn = coor[n, 1]
            ym = coor[m, 1]

            if abs((xn - xm) - 1) == 0 and abs(yn - ym) == 0:
                NN[n, 0] = m
            if abs((xn - xm) + 1) == 0 and abs(yn - ym) == 0:
                NN[n, 2] = m
            if abs((yn - ym) + 1) == 0 and abs(xn - xm) == 0:
                NN[n, 1] = m
            if abs((yn - ym) - 1) == 0 and abs(xn - xm) == 0:
                NN[n, 3]= m
    return NN

#Periodic Boundary conditions
def NN_Bound(NN, coor):
    xmin = min(coor[:, 0])
    ymin = min(coor[:, 1])
    xmax = max(coor[:, 0])
    ymax = max(coor[:, 1])

    N = NN.shape[0]
    NNb = -1*np.ones((N,4), dtype = 'int') #stores the values of the coordinate
                                           #of each periodic neighbor

    #if statements:
    #if i has no left nearest neighbor (NN[i,0] = -1)
    #and j has no right nearest neighbor
    #and the ith index is on the edge of the unit cell along the left wall (xmin)
    #and the jth index is on the edge of the unit cell along the right wall(xmax)
    #then i and j are periodic neighbors
    for i in range(N):
        for j in range(N):
            if NN[i, 0] == -1 and NN[j, 2] == -1 and coor[i, 1] == coor[j, 1] and coor[i, 0] == xmin and coor[j, 0] == xmax:
                NNb[i, 0] = j
            if NN[i, 1] == -1 and NN[j, 3] == -1 and coor[i, 0] == coor[j, 0] and coor[i, 1] == ymax and coor[j, 1] == ymin:
                NNb[i, 1] = j
            if NN[i, 2] == -1 and NN[j, 0] == -1 and coor[i, 1] == coor[j, 1] and coor[i, 0] == xmax and coor[j, 0] == xmin:
                NNb[i, 2] = j
            if NN[i, 3] == -1 and NN[j, 1] == -1 and coor[i, 0] == coor[j, 0] and coor[i, 1] == ymin and coor[j, 1] == ymax:
                NNb[i, 3] = j
    return NNb
