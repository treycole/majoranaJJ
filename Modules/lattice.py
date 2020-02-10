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
def donut(R, r, ax, ay):
    CAx = []
    CAy = []

    xmin = -R #Radius of disk
    ymin = -R

    Nx = int(2*R/ay + 2)
    Ny = int(2*R/ax + 2)

    for j in range(Ny):
        for i in range(Nx):
            x = xmin + i*ax
            y = ymin + j*ay
            #decide if x,y is inside shape
            r_ij = np.sqrt((x)**2+(y)**2)
            if r_ij < R and r_ij > r:
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
