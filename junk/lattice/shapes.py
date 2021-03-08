#Used if we are defining each x and y coordinate in terms of the lattice spacing ax and ay

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
