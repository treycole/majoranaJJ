import scipy.sparse as sparse
import numpy as np
import majoranaJJ.constants as const

def kx(coor, ax, ay, NN, NNb = None, qx = 0):
    row = []; col = []; data = []
    N = coor.shape[0]

    xmax = max(coor[:, 0])
    xmin = max(coor[:, 0])

    Lx = (xmax - xmin + 1)*ax
    tx = 1j/2*ax

    for i in range(N):
        if NN[i,0] != -1:
            row.append(NN[i,0]); col.append(i); data.append(-tx)
        if NN[i,2] != -1:
            row.append(NN[i,2]); col.append(i); data.append(tx)
        if NNb != None and NNb[i,0] != -1:
            row.append(NNb[i,0]); col.append(i); data.append(-tx*np.exp(-1j*qx*Lx))
        if NNb != None and NNb[i,2] != -1:
            row.append(NNb[i,2]); col.append(i); data.append(tx*np.exp(1j*qx*Lx))

    ksq = sparse.csc_matrix((data, (row,col)), shape = (N,N), dtype = 'complex')
    return ksq

def kx2(coor, ax, ay, NN, NNb = None, qx = 0):
    row = []; col = []; data = []
    N = coor.shape[0]

    xmax = max(coor[:, 0])
    xmin = max(coor[:, 0])

    Lx = (xmax - xmin + 1)*ax
    tx = 1/ax**2

    for i in range(N):
        row.append(i); col.append(i); data.append(2*tx)
        if NN[i,0] != -1:
            row.append(NN[i,0]); col.append(i); data.append(-tx)
        if NN[i,2] != -1:
            row.append(NN[i,2]); col.append(i); data.append(-tx)
        if NNb != None and NNb[i,0] != -1:
            row.append(NNb[i,0]); col.append(i); data.append(-tx*np.exp(-1j*qx*Lx))
        if NNb != None and NNb[i,2] != -1:
            row.append(NNb[i,2]); col.append(i); data.append(-tx*np.exp(1j*qx*Lx))


    ksq = sparse.csc_matrix((data, (row,col)), shape = (N,N), dtype = 'complex')
    return ksq

def ky(coor, ax, ay, NN, NNb = None, qy = 0):
    row = []; col = []; data = []
    N = coor.shape[0]

    ymax = max(coor[:, 1])
    ymin = max(coor[:, 1])

    Ly = (ymax - ymin + 1)*ay
    ty = 1j/2*ay

    for i in range(N):
        if NN[i,1] != -1:
            row.append(NN[i,1]); col.append(i); data.append(-ty)
        if NN[i,3] != -1:
            row.append(NN[i,3]); col.append(i); data.append(ty)
        if NNb != None and NNb[i,1] != -1:
            row.append(NNb[i,1]); col.append(i); data.append(-ty*np.exp(-1j*qy*Ly))
        if NNb != None and NNb[i,3] != -1:
            row.append(NNb[i,3]); col.append(i); data.append(ty*np.exp(1j*qy*Ly))

    ksq = sparse.csc_matrix((data, (row,col)), shape = (N,N), dtype = 'complex')
    return ksq

def ky2(coor, ax, ay, NN, NNb = None, qy = 0):
    row = []; col = []; data = []
    N = coor.shape[0]

    ymax = max(coor[:, 1])
    ymin = max(coor[:, 1])

    Ly = (ymax - ymin + 1)*ay
    ty = 1./ay**2

    for i in range(N):
        row.append(i); col.append(i); data.append(2*ty)
        if NN[i,1] != -1:
            row.append(NN[i,1]); col.append(i); data.append(-ty)
        if NN[i,3] != -1:
            row.append(NN[i,3]); col.append(i); data.append(-ty)
        if NNb != None and NNb[i,1] != -1:
            row.append(NNb[i,1]); col.append(i); data.append(-ty*np.exp(-1j*qy*Ly))
        if NNb != None and NNb[i,3] != -1:
            row.append(NNb[i,3]); col.append(i); data.append(-ty*np.exp(1j*qy*Ly))

    ksq = sparse.csc_matrix((data, (row,col)), shape= (N,N), dtype = 'complex')
    return ksq

def Delta(coor, delta, Wsc, Wj, phi = 0, Sx = 0, Sy = 0, cutx = 0, cuty = 0):
    N = coor.shape[0]
    row = []; col = []; data01 = []
    row = []; col = []; data10 = []

    for i in range(N):
        y = coor[i,1]
        x = coor[i,0]
        row.append(i); col.append(i); data10.append(0); data01.append(0)
        if y <= Wsc:
            row.append(i); col.append(i); data10.append( -delta*np.exp( -1j*phi/2 ) )
            data01.append(delta*np.exp( -1j*phi/2 ) )

        if y > Wsc and y <= (Wsc+Wj):
            row.append(i); col.append(i); data10.append(0); data01.append(0)

        if y > (Wsc+Wj):
            row.append(i); col.append(i); data10.append(-delta*np.exp( 1j*phi/2 ) )
            data01.append(delta*np.exp( 1j*phi/2 ))

    D01 = sparse.csc_matrix((data01, (row, col)), shape = (N,N), dtype = 'complex')
    D10 = sparse.csr_matrix((data10, (row, col)), shape = (N,N), dtype = 'complex')
    #D00 = sparse.csc_matrix((dataD, (row, col)), shape = (N,N), dtype = 'complex')
    #D11 = sparse.csc_matrix((dataD, (row, col)), shape = (N,N), dtype = 'complex')

    D = sparse.bmat([[None, D01], [D10, None]])
    return D

    #D00 = np.zeros((N,N))
    #D11 = np.zeros((N,N))
    #D01 = D
    #D10 = -D
    #D = np.block([[D00, D01], [D10, D11]])
