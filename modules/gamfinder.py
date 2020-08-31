import majoranaJJ.operators.sparse.qmsops as spop #sparse operators
import numpy as np
import scipy.linalg as LA
import scipy.sparse.linalg as spLA
from scipy.signal import argrelextrema
import sys
import matplotlib.pyplot as plt

#assuming linear behavior of the E vs gamma energy dispersion
def linear(
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
This function calculates the phase transition points. To work it needs energy eigenvalues and eigenvectors of unperturbed Hamiltonian, or a Hamiltonian without any Zeeman field. When a Zeeman field is turned on, perturbation theory can be used to calculate the new energy eigenvalues and eigenvectors.
"""

def lowE(
    coor, ax, ay, NN, mu, gi, gf,
    NNb = None, Wj = 0, cutx = 0, cuty = 0,
    V = 0, gammax = 0,  gammay = 0, gammaz = 0,
    alpha = 0, delta = 0, phi = 0,
    qx = 0, qy = 0, periodicX = True, periodicY = False,
    k = 20, tol = 0.005, n_bounds = 2
    ):

    Lx = (max(coor[:, 0]) - min(coor[:, 0]) + 1)*ax #Unit cell size in x-direction
    H0 = spop.HBDG(coor, ax, ay, NN, NNb=NNb, Wj=Wj, cutx=cutx, cuty=cuty, V=V, mu=mu, gammaz=1e-5, alpha=alpha, delta=delta, phi=phi, qx=1e-4*(np.pi/Lx), qy=qy, periodicX=periodicX, periodicY=periodicY) #gives low energy basis

    eigs_0, vecs_0 = spLA.eigsh(H0, k=k, sigma=0, which='LM')
    vecs_0_hc = np.conjugate(np.transpose(vecs_0)) #hermitian conjugate

    H_G0 = spop.HBDG(coor, ax, ay, NN, NNb=NNb, Wj=Wj, cutx=cutx, cuty=cuty, V=V, mu=mu, gammax=0, alpha=alpha, delta=delta, phi=phi, qx=qx, qy=qy, periodicX=periodicX, periodicY=periodicY) #Matrix that consists of everything in the Hamiltonian except for the Zeeman energy in the x-direction
    H_G1 = spop.HBDG(coor, ax, ay, NN, NNb=NNb, Wj=Wj, cutx=cutx, cuty=cuty, V=V, mu=mu, gammax=1, alpha=alpha, delta=delta, phi=phi, qx=qx, qy=qy, periodicX=periodicX, periodicY=periodicY) #Hamiltonian with ones on Zeeman energy along x-direction sites

    HG = H_G1 - H_G0 #the proporitonality matrix for gamma-x, it is ones along the sites that have a gamma value

    HG0_DB = np.dot(vecs_0_hc, H_G0.dot(vecs_0))
    HG_DB = np.dot(vecs_0_hc, HG.dot(vecs_0))

    delta_gam = abs(gf - gi)
    steps = int((delta_gam/(0.5*tol))) +1
    gx = np.linspace(gi, gf, steps)
    eig_arr = np.zeros((gx.shape[0]))
    G_crit = []

    for i in range(gx.shape[0]):
        H_DB = HG0_DB + gx[i]*HG_DB
        eigs_DB, U_DB = LA.eigh(H_DB)

        eig_arr[i] = eigs_DB[int(k/2)]

    #checking edge cases
    if eig_arr[0] < tol:
        G_crit.append(gx[0])
    if eig_arr[-1] < tol:
        G_crit.append(gx[-1])

    local_min_idx = np.array(argrelextrema(eig_arr, np.less)[0]) #local minima indices in the E vs gamma plot

    """could be used for a first approx of gamma values for zero energy crossings (below tolerance):
    eigs_local_min = eig_arr[local_min_idx] #eigenvalues at local min
    ZEC_idx = [] #not all local min are zero energy crossings
    for i in range(len(local_min_idx)):
        if eigs_local_min[i] < tol:
            ZEC_idx.append(local_min_idx[i]) #only append indices that have local min below tolerance
    eigs_local_min_below_tol = eig_arr[ZEC_idx] #eigenvalues under tolerance

    #plt.plot(gx, eig_arr, c='b')
    #plt.scatter(gx[ZEC_idx], eig_arr[ZEC_idx], c='r', marker = 'X')
    #plt.show()

    print(eigs_min.size, "local minima")
    """
    print(local_min_idx.size, "local minima found")
    tol = tol/100
    for i in range(0, local_min_idx.size): #eigs_min.size
        gx_c = gx[local_min_idx[i]] #gx[ZEC_idx[i]]""" #first approx g_critical
        print("Checking for ZEC around gamma = {}, energy = {}".format(gx_c, eig_arr[i]))
        gx_lower = gx[local_min_idx[i]-1]#gx[ZEC_idx[i]-1]""" #one step back
        gx_higher = gx[local_min_idx[i]+1]#gx[ZEC_idx[i]+1]""" #one step forward

        delta_gam = (gx_higher - gx_lower)
        n_steps = int((delta_gam/(0.5*tol))) + 1
        gx_finer = np.linspace(gx_lower, gx_higher, n_steps) #high res gamma around supposed zero energy crossing (local min)
        eig_arr_finer = np.zeros((gx_finer.size)) #new eigenvalue array
        for j in range(gx_finer.shape[0]):
            H_DB = HG0_DB + gx_finer[j]*HG_DB
            eigs_DB, U_DB = LA.eigh(H_DB)
            eig_arr_finer[j] = eigs_DB[int(k/2)] #k/2 -> lowest postive energy state

        min_idx_finer = np.array(argrelextrema(eig_arr_finer, np.less)[0]) #new local minima indices
        eigs_min_finer = eig_arr_finer[min_idx_finer] #isolating local minima
        for m in range(min_idx_finer.shape[0]):
            if eigs_min_finer[m] < tol:
                crossing_gamma = gx_finer[min_idx_finer]
                G_crit.append(crossing_gamma)
                #ZEC_idx_HR = np.where(eig_arr_finer == eigs_min_finer[m])
                #G_crit.append(gx_finer[ZEC_idx_HR[0][0]]) #append critical gamma
                #print("crossing found at gx = ", gx_finer[ZEC_idx_HR[0][0]])
                print("crossing found at gx = {} and E = {}".format(crossing_gamma, eigs_min_finer[m]))
        #if min(eigs_min_finer) < tol: #if effectively zero crossing
        #    ZEC_idx_HR = np.where(eig_arr_finer == min(eigs_min_finer)) #where is the absolute minimum around approximate minumum, don't want more than one
        #    G_crit.append(gx_finer[ZEC_idx_HR[0][0]]) #append critical gamma
        #    print("crossing found at gx = ", gx_finer[ZEC_idx_HR[0][0]])

            #plt.plot(gx_finer, eig_arr_finer, c = 'b')
            #plt.scatter(gx_finer[ZEC_idx_HR[0][0]], eig_arr_finer[ZEC_idx_HR[0][0]])
            #plt.show()
    G_crit = np.array(G_crit)
    return G_crit
