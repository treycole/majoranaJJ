#Used if the coordinate array (donut, square etc.) takes values in units ..
#of ax and ay instead of 1 --> N
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
