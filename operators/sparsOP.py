import scipy.sparse as sparse

import majoranaJJ.const as const

def kx(coor, ax, ay, NN, NNb = None, periodic = 'no'):
    row = []; col = []; data = []
    N = coor.shape[0]
    for i in range(N):
        if NN[i,0] != -1:
            row.append(NN[i,0]); col.append(i); data.append(-1j/2*ax)
        if NN[i,2] != -1:
            row.append(NN[i,2]); col.append(i); data.append(1j/2*ax)
        if peridic.lower() == 
    ksq = sparse.csr_matrix((data, (row,col)), shape = (N,N), dtype = 'complex')
    return ksq

def kx2(coor, ax, ay, NN, NNb = None, periodic = 'no'):
    row = []; col = []; data = []
    N = coor.shape[0]
    for i in range(N):
        row.append(i); col.append(i); data.append(2/ax**2)
        if NN[i,0] != -1:
            row.append(NN[i,0]); col.append(i); data.append(-1./ax**2)
        if NN[i,2] != -1:
            row.append(NN[i,2]); col.append(i); data.append(-1./ax**2)
        if

    ksq = sparse.csr_matrix((data, (row,col)), shape = (N,N), dtype = 'complex')
    return ksq

def ky(coor, ax, ay, NN, NNb = None, periodic = 'no'):
    row = []; col = []; data = []
    N = coor.shape[0]
    for i in range(N):
        if NN[i,1] != -1:
            row.append(NN[i,1]); col.append(i); data.append(-1j/2*ay)
        if NN[i,3] != -1:
            row.append(NN[i,3]); col.append(i); data.append(1j/2*ay)
    ksq = sparse.csr_matrix((data, (row,col)), shape = (N,N), dtype = 'complex')
    return ksq

def ky2(coor, ax, ay, NN, NNb = None, periodic = 'no'):
    row = []; col = []; data = []
    N = coor.shape[0]
    for i in range(N):
        row.append(i); col.append(i); data.append(2/ay**2)
        if NN[i,1] != -1:
            row.append(NN[i,1]); col.append(i); data.append(-1./ay**2)
        if NN[i,3] != -1:
            row.append(NN[i,3]); col.append(i); data.append(-1./ay**2)

    ksq = sparse.csr_matrix((data, (row,col)), shape= (N,N), dtype = 'complex')
    return ksq
