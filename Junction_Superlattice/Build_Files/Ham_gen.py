import sys
import numpy as np
import scipy as SCI
import scipy.sparse as Spar
import sparse_mtx_manipulation as SMM
import parameters as par
np.set_printoptions(linewidth = 500)


def Ham_comp_gen(Diff_ops,m_eff,alp_x,alp_y):

    hbm = 1000. * par.hbm0 / (m_eff)
    N = SMM.zero_csr_mtx(Diff_ops.Diag.shape[0])

    for i in range(3):
        if i == 0:
            A = (hbm/2.)* (Diff_ops.kxSq + Diff_ops.kySq)
            B = -alp_y*Diff_ops.ky - 1j*alp_x*Diff_ops.kx
            C = -alp_y*Diff_ops.ky + 1j*alp_x*Diff_ops.kx
            D = (hbm/2.)* (Diff_ops.kxSq + Diff_ops.kySq)
        elif i == 1:
            A = hbm * Diff_ops.kx
            B = -1j*alp_x * Diff_ops.Diag
            C = 1j*alp_x * Diff_ops.Diag
            D = hbm * Diff_ops.kx
        elif i == 2:
            A = (hbm/2.)*Diff_ops.Diag
            B = 1.*N
            C = 1.*N
            D = (hbm/2.)*Diff_ops.Diag

        H_comp = Spar.bmat([
                           [A,B],
                           [C,D]
        ],format = 'csc')
        if i == 0:
            H_0 = H_comp
        elif i == 1:
            H_qx = H_comp
        elif i == 2:
            H_qxSq = H_comp
    M = Diff_ops.Diag
    OVERLAP = Spar.bmat([
                       [M,N],
                       [N,M]
    ],format = 'csc')
    GAM = Spar.bmat([
                       [N,M],
                       [M,N]
    ],format = 'csc')

    M = Diff_ops.Diag_1
    DELTA_B = Spar.bmat([
                       [N,M],
                       [-M,N]
    ],format = 'csc')

    M = Diff_ops.Diag_3
    DELTA_T = Spar.bmat([
                       [N,M],
                       [-M,N]
    ],format = 'csc')

    M = Diff_ops.Diag_2
    V_J = Spar.bmat([
                       [M,N],
                       [N,M]
    ],format = 'csc')

    M = Diff_ops.Diag_1 + Diff_ops.Diag_3
    V_SC = Spar.bmat([
                       [M,N],
                       [N,M]
    ],format = 'csc')

    return H_0,H_qx,H_qxSq,OVERLAP,GAM,DELTA_B,DELTA_T,V_J, V_SC 
