import numpy as np
import scipy.sparse.linalg as spLA

def mu_finder(H):
    num = 1 # This is the number of eigenvalues and eigenvectors you want
    sigma = 0 # This is the eigenvalue we search around
    which = 'LM'
    eigs = np.sort(spLA.eigsh(H, k = num, sigma = sigma, which = which)[0])

    return eigs[0]
