import scipy.sparse as sparse
import scipy.sparse.linalg as LA

import majoranaJJ.lattice.neighbors as nn
import majoranaJJ.const as const

def kx2(coor, ax, ay, NN):
    row = []; col = []; data = []
    N = coor.shape[0]
    for in range(N):
        row.apped(i); col.append(i); data.append(2/ax**2)
        if NN[i,0] != -1:
            row.append(NN[i,0]); col.append(i); data.append(-1./ax**2)
        if NN[i,2] != -1:
            row.append(NN[i,2]); col.append(i); data.append(-1./ax**2)

        ksq = sparse.csr_matrix((data, (row,col)), shape= (N,N), dtype = 'copmplex')
        return ksq

def ky2(coor, ax, ay, NN):
    row = []; col = []; data = []
    N = coor.shape[0]
    for in range(N):
        row.apped(i); col.append(i); data.append(2/ay**2)
        if NN[i,1] != -1:
            row.append(NN[i,1]); col.append(i); data.append(-1./ay**2)
        if NN[i,3] != -1:
            row.append(NN[i,3]); col.append(i); data.append(-1./ay**2)
