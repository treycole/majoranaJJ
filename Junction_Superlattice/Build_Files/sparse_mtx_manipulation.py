import scipy.sparse as Spar
import numpy as np
import sys

#for i,j,v in itertools.izip(mtx.row, mtx.col, mtx.data):
        #    print i,j, '            ', v


def append_mtx_block(mtx,row_n,col_n,data_n,N,m,n):
    """
    Appends mtx to the lists row_n, col_n, data_n
    that represent a new matrix. The matrix being appended
    is shifted to the m,n block of the new matrix, where
    each block is N x N in size
    """
    ii = N*m
    jj = N*n
    for i,j,v in zip(mtx.row, mtx.col, mtx.data):
        row_n.append(i+ii)
        col_n.append(j+jj)
        data_n.append(v)

def get_inverse_perm(perm):
    N = perm.size
    perm_inv = np.zeros(N,dtype = 'int')
    for i in range(0,N):
        idx = perm[i]
        perm_inv[idx] = i
    return perm_inv


def zero_csr_mtx(N,dtype = 'complex'):
    row = []
    col = []
    data = []
    mtx = Spar.csr_matrix((data,(row,col)), shape=(N, N),dtype = dtype)
    return mtx

def check_Hermitean(mtx,N,tol = 10**(-20)):
    mtxC = mtx.tocoo()
    for i, j, v in zip(mtxC.row,mtxC.col,mtxC.data):
        if v != np.conjugate(mtx[j,i]):
            if abs(v -np.conjugate(mtx[j,i]) ) > tol:
                bi = i / N; bj = j / N
                print ("Non Hermitian block is (%d,%d)" % (bi,bj))
                print (v,mtx[j,i],i,j)
                return False
    return True




"""
def inactive_dof_remover(mtx,NSM,NSC):
    ###Removes the valence band dof from the supercondcutor.
    ###This was used in the cylindrical model, so its of no use here
    N = NSM + NSC
    if 8*N != mtx.shape[1]:
        print "The number of degrees of freedom doesn't match with what NSM and NSC add up to"
        print NSM,NSC,8*N,mtx.shape[1]
        sys.exit()
    dofs = np.zeros(8*NSM+2*NSC,dtype = 'int')
    counter = 0
    for m in range(0,8):
        for i in range(0,N):
            if m == 0 or m == 1:
                dofs[counter] = i + m*N
                counter += 1
            else:
                if i < NSM:
                    dofs[counter] = i + m*N
                    counter += 1
    if isinstance(mtx,Spar.csr_matrix) == False:
        mtxCSR = mtx.tocsr()
        mtxCSR = (mtxCSR[dofs,:])[:,dofs]
    else:
        mtxCSR = (mtx[dofs,:])[:,dofs]
    return mtxCSR
"""
