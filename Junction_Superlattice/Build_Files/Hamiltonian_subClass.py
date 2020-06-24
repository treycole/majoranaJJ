import sys
import numpy as np
import scipy as SCI
import scipy.sparse as Spar
import scipy.sparse.linalg as SparLinalg
import scipy.linalg as linalg
import matplotlib.pyplot as plt
import parameters as par
import Ham_gen as HG
np.set_printoptions(linewidth = 500)


class HAM_subClass:

    def __init__(self,Junc_obj):
        self.Junc_obj = Junc_obj # parent instance of Junction_Model
        self.gen_Ham_components() # Generates the various Hamiltonian componenets

    def gen_Ham_components(self):
        self.H_0, self.H_qx, self.H_qxSq, \
        self.S,self.GAM, \
        self.DELTA_B,self.DELTA_T, \
        self.V_J,self.V_SC = \
        HG.Ham_comp_gen(self.Junc_obj.MESH.DIFF_OPS,self.Junc_obj.m_eff,self.Junc_obj.alpha,self.Junc_obj.alpha)
        self.DELTA_B_hc = self.DELTA_B.conjugate().transpose()
        self.DELTA_T_hc = self.DELTA_T.conjugate().transpose()

    def compile_Ham(self,qx,mu,Gam,V_j,V_sc):
        H = self.H_0 + qx*self.H_qx + (qx * qx) * self.H_qxSq \
           - mu*self.S + Gam * self.GAM \
           + V_j * self.V_J + V_sc * self.V_SC
        return H, self.S

    def compile_Ham_BdG(self,qx,mu,Gam,Delta,phi,V_j,V_sc):
        Ham_p, S = self.compile_Ham(qx,mu,Gam,V_j,V_sc)
        Ham_h, S = (self.compile_Ham(-qx,mu,Gam,V_j,V_sc))
        Ham_h = -(Ham_h).conjugate()
        Ham_BdG = Spar.bmat([
                            [Ham_p,Delta*self.DELTA_B + Delta*np.exp(1j*phi)*self.DELTA_T],
                            [np.conjugate(Delta)*self.DELTA_B_hc + np.conjugate(Delta)*np.exp(-1j*phi)*self.DELTA_T_hc,Ham_h]
        ],format = 'csc')
        S_BdG = Spar.bmat([
                          [S,None],
                          [None,S]
        ],format = 'csc')
        return Ham_BdG, S_BdG

    def solve_Ham(self,Ham,S,num,sigma,which = 'LM',Return_vecs = False,reverse = False):
        ### Finding "num" eigenvalues near E = sigma
        eigs,vecs = SparLinalg.eigsh(Ham,M=S,k=num,sigma = sigma, which = which)
        idx = np.argsort(eigs)
        if reverse:
            idx = idx[::-1]
        if Return_vecs:
            return eigs[idx], vecs[:,idx]
        else:
            return eigs[idx]

    def generate_lNRG_subspace(self,qx_knot,V_j,V_sc,num):
        lNRG_subObj = lNRG_subClass(self)
        lNRG_subObj.gen_lNRG(qx_knot,V_j,V_sc,num)
        return lNRG_subObj

    def generate_lNRG_BdG_subspace(self,qx_knot,V_j,V_sc,mu,num):
        lNRG_subObj = lNRG_BdG_subClass(self)
        lNRG_subObj.gen_lNRG(qx_knot,V_j,V_sc,mu,num)
        return lNRG_subObj


class lNRG_subClass:

    def __init__(self,Ham_obj):
        self.Ham_obj = Ham_obj
        self.Junc_obj = Ham_obj.Junc_obj

    def gen_lNRG(self,qx_knot,V_j,V_sc,num):
        self.qx_knot = qx_knot
        self.V_j = V_j
        self.V_sc = V_sc

        Ham, S = self.Ham_obj.compile_Ham(qx_knot,0.,1.e-3,V_j,V_sc)
        eigs, U  = self.Ham_obj.solve_Ham(Ham,S,num,0.,Return_vecs = True)
        U_hc = np.conjugate(np.transpose(U))
        self.U = U
        self.H_0 = np.dot(U_hc, self.Ham_obj.H_0.dot(U))
        self.H_qx = np.dot(U_hc, self.Ham_obj.H_qx.dot(U))
        self.H_qxSq = np.dot(U_hc, self.Ham_obj.H_qxSq.dot(U))
        self.GAM = np.dot(U_hc, self.Ham_obj.GAM.dot(U))
        self.V_J = np.dot(U_hc, self.Ham_obj.V_J.dot(U))
        self.V_SC = np.dot(U_hc, self.Ham_obj.V_SC.dot(U))
        self.S = np.eye(self.H_0.shape[0])

    def compile_Ham(self,qx,mu,Gam,V_j,V_sc):
        H = self.H_0 + qx*self.H_qx + (qx * qx) * self.H_qxSq \
           - mu*self.S + Gam * self.GAM \
           + V_j * self.V_J + V_sc * self.V_SC
        #print (H.shape)
        return H

    def solve_Ham(self,Ham,num = -1):
        #print (Ham.shape)
        if num == -1:
            eigs, U = linalg.eigh(Ham)
        else:
            eigs, U = linalg.eigh(Ham,eigvals = (0,num-1))
        return eigs, U


class lNRG_BdG_subClass:

    def __init__(self,Ham_obj):
        self.Ham_obj = Ham_obj
        self.Junc_obj = Ham_obj.Junc_obj

    def gen_lNRG(self,qx_knot,V_j,V_sc,mu,num):
        self.qx_knot = qx_knot
        self.V_j = V_j
        self.V_sc = V_sc
        self.mu = mu

        ### Diagonalize Hamiltonian at qx_knot
        Gam = 1.e-4
        Ham, S = self.Ham_obj.compile_Ham(qx_knot,mu,Gam,V_j,V_sc)
        print ("Ham Shape: ", Ham.shape)
        eigs, U  = self.Ham_obj.solve_Ham(Ham,S,num,0.,Return_vecs = True)
        #print (eigs)
        #sys.exit()
        U_hc = np.conjugate(np.transpose(U))
        U_c = np.conjugate(U)

        ### Testing that spectrum is particle-hole symmetric, otherwise shows a warning

        ### Transform the components of the Hamiltonian into the new basis
        self.U = U
        self.H_0 = np.dot(U_hc, self.Ham_obj.H_0.dot(U))
        self.H_qx = np.dot(U_hc, self.Ham_obj.H_qx.dot(U))
        self.H_qxSq = np.dot(U_hc, self.Ham_obj.H_qxSq.dot(U))
        self.GAM = np.dot(U_hc, self.Ham_obj.GAM.dot(U))
        self.V_J = np.dot(U_hc, self.Ham_obj.V_J.dot(U))
        self.V_SC = np.dot(U_hc, self.Ham_obj.V_SC.dot(U))
        self.S = np.dot(U_hc, self.Ham_obj.S.dot(U))
        #self.DELTA_B = np.dot(U_hc, self.Ham_obj.DELTA_B.dot(U_tr))
        #self.DELTA_T = np.dot(U_hc, self.Ham_obj.DELTA_T.dot(U_tr))
        self.DELTA_B = np.dot(U_hc, self.Ham_obj.DELTA_B.dot(U_c))
        self.DELTA_T = np.dot(U_hc, self.Ham_obj.DELTA_T.dot(U_c))
        self.DELTA_B_hc = self.DELTA_B.conjugate().transpose()
        self.DELTA_T_hc = self.DELTA_T.conjugate().transpose()

    def compile_Ham(self,qx,mu,Gam,V_j,V_sc):
        H = self.H_0 + qx*self.H_qx + (qx * qx) * self.H_qxSq \
           - mu*self.S + Gam * self.GAM \
           + V_j * self.V_J + V_sc * self.V_SC
        #print (H.shape)
        return H

    def compile_Ham_BdG(self,qx,mu,Gam,Delta,phi,V_j,V_sc):
        Ham_p = self.compile_Ham(qx,mu,Gam,V_j,V_sc)
        Ham_h = -(self.compile_Ham(-qx,mu,Gam,V_j,V_sc)).conjugate()
        Ham_BdG = np.bmat([
                            [Ham_p,Delta*self.DELTA_B + Delta*np.exp(1j*phi)*self.DELTA_T],
                            [np.conjugate(Delta)*self.DELTA_B_hc + np.conjugate(Delta)*np.exp(-1j*phi)*self.DELTA_T_hc,Ham_h]
        ])
        return Ham_BdG

    def solve_Ham(self,Ham,num = -1):
        #print (Ham.shape)
        if num == -1:
            eigs, U = linalg.eigh(Ham)
        else:
            eigs, U = linalg.eigh(Ham,eigvals = (0,num-1))
        return eigs, U
