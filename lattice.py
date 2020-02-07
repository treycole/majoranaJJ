#defining nearest neighbor array
#NN_arr is Nx4, the columns store the index of the nearest neighbors.
#Left: NN[n,0] = (n-Nx), Above: NN[n,1] = n+1, Right: NN[n, 2] = n+1, Down NN[n, 3] = n+Nx
#if there is no lattice site in nearest neighbor spot, value is -1
def NN_Arr(coor, ax, ay):
    N = coor.shape[0]
    tol = 1e-8
    NN = -1*np.ones((N,4), dtype = 'int')
    for n in range(N):
        for m in range(N):
            xn = coor[n, 0]
            xm = coor[m, 0]
            yn = coor[n,1]
            ym = coor[m,1]

            if abs((xn - xm) - ax)< tol and abs(yn - ym) < tol:
                NN[n, 0] = m
                if abs((xn - xm) + ax) < tol and abs(yn - ym) < tol:
                    NN[n, 2] = m
                    if abs((yn - ym) + ay) < tol and abs(xn - xm) < tol:
                        NN[n, 1] = m
                        if abs((yn - ym) - ay) < tol and abs(xn - xm) < tol:
                            NN[n, 3]= m
                            return NN
#defining lattice, numbered 0->N
def lattice(Nx, Ny):
    lattice = np.zeros((Nx, Ny))
    for i in range(Ny):
        for j in range(Nx):
            lattice[i, j] = j + i*Ny
    return lattice
#defining coordinate array
#coordinate array is Nx2, first column is array of x values in units of [A], second column is y values in units of [A]
#creates a square or rectangular coordinate array, it stores i*ax as x coordinate, n = nx + nx*ny means ...
# the lattice reference number is the x-value of the loop plus the y-value times the x-value
def square(Nx, Ny, ax, ay):
    N = Nx*Ny
    coor = np.zeros((N,2))
    for i in range(Nx):
        for j in range(Ny):
            n = i + Nx * j
            x = (i) * ax
            y = (j) * ay
            coor[n,0] = x
            coor[n,1] = y
    return coor
#Disk with a hole
def donut(R, r, ax, ay):
    CAx = []
    CAy = []

    xmin = -R #Radius of disk
    ymin = -R

    Nx = int(2*R/ax + 2)
    Ny = int(2*R/ay + 2)

    for j in range(Ny):
        for i in range(Nx):
            x = xmin + i*ax
            y = ymin + j*ay
            #decide if x,y is inside shape
            r_ij = np.sqrt(x**2+y**2)
            if r_ij < R and r_ij > r:
                CAx.append(x)
                CAy.append(y)

    coor_arr = np.zeros((len(CAx), 2))
    coor_arr[:, 0] = np.array(CAx)
    coor_arr[:, 1] = np.array(CAy)
    return coor_arr
