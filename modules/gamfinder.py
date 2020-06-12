import majoranaJJ.operators.sparse.qmsops as spop #sparse operators
import numpy as np
import scipy.linalg as LA
import scipy.sparse.linalg as spLA
from scipy.signal import argrelextrema
import sys
import matplotlib.pyplot as plt

def gamfinder(
    coor, ax, ay, NN, mu,
    NNb = None, Wj = 0, cutx = 0, cuty = 0, V = 0,
    gammax = 0, gammay = 0, gammaz = 0,
    alpha = 0, delta = 0 , phi = 0,
    qx = 0, qy = 0, periodicX = True, periodicY = False,
    k = 20, sigma = 0, which = 'LM', tol = 0, maxiter = None
    ):

    #saving the particle energies, all energies above E=0
    Ei = spop.EBDG(
        coor, ax, ay, NN, NNb = NNb, Wj = Wj,
        cutx = cutx, cuty = cuty,
        V = V, mu = mu,
        gammax = gammax, gammay = gammay, gammaz = gammaz,
        alpha = alpha, delta = delta, phi = phi,
        qx = qx, qy = qy,
        periodicX = periodicX, periodicY = periodicY,
        k = k, sigma = sigma, which = which, tol = tol, maxiter = maxiter
        )[int(k/2):][::2]
    #print(Ei)

    deltaG = 0.00001
    gammanew = gammax + deltaG

    #saving the particle energies, all energies above E=0
    Ef = spop.EBDG(
        coor, ax, ay, NN, NNb = NNb, Wj = Wj,
        cutx = cutx, cuty = cuty,
        V = V, mu = mu,
        gammax = gammanew, gammay = gammay, gammaz = gammaz,
        alpha = alpha, delta = delta, phi = phi,
        qx = qx, qy = qy,
        periodicX = periodicX, periodicY = periodicY,
        k = k, sigma = sigma, which = which, tol = tol, maxiter = maxiter
        )[int(k/2):][::2]
    #print(Ef)

    m = np.array((Ef - Ei)/(gammanew - gammax)) #slope, linear dependence on gamma
    #print(m)
    b = np.array(Ei - m*gammax) #y-intercept
    G_crit = np.array(-b/m) #gamma value that E=0 for given mu value
    #print(G_crit)

    return G_crit

"""
This function calculates the phase transition points. To work it needs energy eigenvalues and eigenvectors of unperturbed Hamiltonian, or a Hamiltonian without any Zeeman field. When a Zeeman field is turned on, perturbation theory can be applied to calculate the new energy eigenvalues and eigenvectors.
"""

def gamfinder_lowE(
    coor, ax, ay, NN, mu, gi, gf,
    NNb = None, Wj = 0, cutx = 0, cuty = 0,
    V = 0, gammax = 0,  gammay = 0, gammaz = 0,
    alpha = 0, delta = 0, phi = 0,
    qx = 0, qy = 0, periodicX = True, periodicY = False,
    k = 20, tol = 0.01, steps=2000
    ):

    MU = mu #fixed mu value
    Lx = (max(coor[:, 0]) - min(coor[:, 0]) + 1)*ax #Unit cell size in x-direction

    H0 = spop.HBDG(coor, ax, ay, NN, NNb=NNb, Wj=Wj, cutx=cutx, cuty=cuty, V=V, mu=MU, gammaz=1e-5, alpha=alpha, delta=delta, phi=phi, qx=0.0001*(np.pi/Lx), qy=qy, periodicX=periodicX, periodicY=periodicY) #gives low energy basis

    eigs_0, vecs_0 = spLA.eigsh(H0, k=k, sigma=0, which='LM')
    vecs_0_hc = np.conjugate(np.transpose(vecs_0)) #hermitian conjugate

    H_G0 =  spop.HBDG(coor, ax, ay, NN, NNb = NNb, Wj = Wj, cutx = cutx, cuty = cuty, V = V, mu = MU, gammax = 0, alpha = alpha, delta = delta, phi = phi, qx = qx, qy=qy, periodicX = periodicX, periodicY=periodicY) #Matrix that consists of everything in the Hamiltonian except for the Zeeman energy in the x-direction

    H_G1 = spop.HBDG(coor, ax, ay, NN, NNb = NNb, Wj = Wj, cutx = cutx, cuty = cuty, V = V, mu = MU, gammax = 1, alpha = alpha, delta = delta, phi = phi, qx = qx, qy=qy, periodicX = periodicX, periodicY=periodicY) #Hamiltonian with ones on Zeeman energy along x-direction sites

    HG = H_G1 - H_G0    #the proporitonality matrix for gamma-x, it is ones along the sites that have a gamma value

    HG0_DB = np.dot(vecs_0_hc, H_G0.dot(vecs_0))
    HG_DB = np.dot(vecs_0_hc, HG.dot(vecs_0))

    gx = np.linspace(gi, gf, steps)
    eig_arr = np.zeros((gx.shape[0]))

    for i in range(gx.shape[0]):
        #H = H_G0 + gx[i]*HG
        H_DB = HG0_DB + gx[i]*HG_DB
        #H_DB = np.dot(vecs_0_hc, H.dot(vecs_0))
        eigs_DB, U_DB = LA.eigh(H_DB)

        eig_arr[i] = eigs_DB[int(k/2)]

    eig_min_idx = np.array(argrelextrema(eig_arr, np.less)[0]) #local minima indices

    G_crit = gx[eig_min_idx[:]]
    return G_crit

"""
    G_crit = []
    for j in range(eig_min_idx.size):
        gx_c = gx[eig_min_idx[j]] #gamma value at local minima, first approx
        gx_c_lower = gx[eig_min_idx[j]-1] #gamma value one step behind minima
        gx_c_higher = gx[eig_min_idx[j]+1] #gamma value one step in front of minima
        gx_finer = np.linspace(gx_c_lower, gx_c_higher, steps) #refined gamma range

        eig_arr_finer = np.zeros(gx_finer.size) #new eigen value array that is higher resolution around local minima in first approximation
        for i in range(gx_finer.shape[0]):
            #H = H_G0 + gx[i]*H_G1 #Hamiltonian
            H_DB = HG0_DB + gx_finer[i]*HG_DB
            #H_DB = np.dot(vecs_0_hc, H.dot(vecs_0)) #change of basis, diff basis
            eigs_DB, U_DB = LA.eigh(H_DB)

            eig_arr_finer[i] = eigs_DB[int(k/2)] #k/2 -> lowest postive energy state

        eig_min_idx_finer = np.array(argrelextrema(eig_arr_finer, np.less)[0]) #new local minima indices
        eigs_local_min_finer = eig_arr_finer[eig_min_idx_finer] #isolating local minima
        #G_crit = np.ones((n_boundry, eigs_local_min_finder.size))
        for k in range(eigs_local_min_finer.size):
            if eigs_local_min_finer[k] < tol: #if effectively zero crossing
                G_crit.append(gx_finer[eig_min_idx_finer[k]]) #append critical gamma
                #print(gx_finer[eig_min_idx_finer[k]])

    G_crit = np.array(G_crit)

    #eigs_local_min = eig_arr[eig_min_idx]

    #G_crit = [] #np.empty((num_bound), dtype = 'object')

    #for j in range(eigs_local_min.size):
    #    if eigs_local_min[j] < tol:
    #        G_crit.append(gx[eig_min_idx[j]])

    #for i in range(eig_arr.shape[0]):
    #    if eig_arr[i] < tol:
    #        G_crit[idx] = gx[i]
    #        idx += 1

    #G_crit = np.array(G_crit)
    return G_crit
"""
