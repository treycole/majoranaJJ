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

#Disk with a hole
def donut(R, r):
    CAx = []
    CAy = []

    xmin = -R #Radius of disk
    ymin = -R

    Nx = int(2*R + 2)
    Ny = int(2*R + 2)

    for j in range(Ny):
        for i in range(Nx):
            x = xmin + i
            y = ymin + j
            #decide if x,y is inside shape
            r_ij = np.sqrt((x)**2+(y)**2)
            if r_ij < R and r_ij > r:
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

def Ibeam(xbase, xcut, y1, y2):
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

#defining nearest neighbor array
#NN_arr is Nx4, the columns store the index of the nearest neighbors.
#Left: NN[n,0] = (n-Nx), Above: NN[n,1] = n+1, Right: NN[n, 2] = n+1, Down NN[n, 3] = n+Nx
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
#     left_reg = []
#     top_reg = []
#     right_reg = []
#     bottom_reg = []
#     xmin = np.min(coor[:,0])
    N = NN.shape[0]
    NNk = -1*np.ones((N,4), dtype = 'int')
    #if statements: if i has no left nearest neighbor and j has no right nearest neighbor and the ith index has the same y-value
    # as the jth index then the periodic boundary nearest neighbor for i is j
    for i in range(N):
        for j in range(N):
            if NN[i, 0] == -1 and NN[j, 2] == -1 and coor[i, 1] == coor[j, 1]:
                NNk[i, 0] = j
            if NN[i, 1] == -1 and NN[j, 3] == -1 and coor[i, 0] == coor[j, 0]:
                NNk[i, 1] = j
            if NN[i, 2] == -1 and NN[j, 0] == -1 and coor[i, 1] == coor[j, 1]:
                NNk[i, 2] = j
            if NN[i, 3] == -1 and NN[j, 1] == -1 and coor[i, 0] == coor[j, 0]:
                NNk[i, 3] = j
    return NNk
