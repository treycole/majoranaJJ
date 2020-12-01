import majoranaJJ.operators.sparse_operators as spop #sparse operators
from majoranaJJ.modules import constants as const
import numpy as np
import scipy.linalg as LA
import scipy.sparse.linalg as spLA
from scipy.signal import argrelextrema
import sys
import matplotlib.pyplot as plt

#Assuming linear behavior of the E vs gamma energy dispersion
#Taking the slope and the initial points in the energy vs gamma plot
#Extrapolate to find zero energy crossing
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
This function calculates the topological phase transition points.

To work it needs to calculate energy eigenvalues and eigenvectors of unperturbed Hamiltonian, or a Hamiltonian without any Zeeman field.

When a Zeeman field is turned on, a perturbed Hamiltonian can be used to calculate the new energy eigenvalues and eigenvectors from a reduced subspace of the initial Hilbert space.

This function also assumes that the phase transition points occur at kx=0
This will need to be reproduced to find energy minima in phase space for topological gap size

To have any meaning behind a phase boundary, the system must be perioidic and thus requires a neighbor boundary array. again, qx is assumed to be zero
"""
def lowE(
    coor, NN, NNb, ax, ay, mu, gi, gf,
    Wj = 0, cutx = 0, cuty = 0,
    V = 0, meff_normal = 0.026*const.m0, meff_sc = 0.026*const.m0,
    g_normal = 26, g_sc = 26,
    alpha = 0, delta = 0, phi = 0,
    Tesla = False, diff_g_factors = True,  Rfactor = 0, diff_alphas = False, diff_meff = False,
    k = 20, tol = 0.005
    ):

    Lx = (max(coor[:, 0]) - min(coor[:, 0]) + 1)*ax #Unit cell size in x-direction

    #gamz and qx are finite in order to avoid degneracy issues
    H0 = spop.HBDG(coor, ax, ay, NN, NNb=NNb, Wj=Wj, cutx=cutx, cuty=cuty, V=V, mu=mu, gamz=1e-4, alpha=alpha, delta=delta, phi=phi, qx=0*1e-5*(np.pi/Lx), Tesla=Tesla, diff_g_factors=diff_g_factors, Rfactor=Rfactor, diff_alphas=diff_alphas, diff_meff=diff_meff) #gives low energy basis
    eigs_0, vecs_0 = spLA.eigsh(H0, k=k, sigma=0, which='LM')
    vecs_0_hc = np.conjugate(np.transpose(vecs_0)) #hermitian conjugate

    H_G0 = spop.HBDG(coor, ax, ay, NN, NNb=NNb, Wj=Wj, cutx=cutx, cuty=cuty, V=V, mu=mu, alpha=alpha, delta=delta, phi=phi, qx=0, Tesla=Tesla, diff_g_factors=diff_g_factors, Rfactor=Rfactor, diff_alphas=diff_alphas, diff_meff=diff_meff) #Matrix that consists of everything in the Hamiltonian except for the Zeeman energy in the x-direction
    H_G1 = spop.HBDG(coor, ax, ay, NN, NNb=NNb, Wj=Wj, cutx=cutx, cuty=cuty, V=V, mu=mu, gamx=1, alpha=alpha, delta=delta, phi=phi, qx=0, Tesla=Tesla, diff_g_factors=diff_g_factors, Rfactor=Rfactor, diff_alphas=diff_alphas, diff_meff=diff_meff) #Hamiltonian with ones on Zeeman energy along x-direction sites
    HG = H_G1 - H_G0 #the proporitonality matrix for gam-x, it is ones along the sites that have a gam value
    HG0_DB = np.dot(vecs_0_hc, H_G0.dot(vecs_0))
    HG_DB = np.dot(vecs_0_hc, HG.dot(vecs_0))

    G_crit = []
    delta_gam = abs(gf - gi)
    steps = int((delta_gam/(0.5*tol))) + 1
    gx = np.linspace(gi, gf, steps)
    eig_arr = np.zeros((gx.shape[0]))
    for i in range(gx.shape[0]):
        H_DB = HG0_DB + gx[i]*HG_DB
        eigs_DB, U_DB = LA.eigh(H_DB)
        eig_arr[i] = eigs_DB[int(k/2)]

    #checking edge cases
    if eig_arr[0] < tol:
        G_crit.append(gx[0])
    if eig_arr[-1] < tol:
        G_crit.append(gx[-1])

    local_min_idx = np.array(argrelextrema(eig_arr, np.less)[0]) #local minima indices in the E vs gam plot
    print(local_min_idx.size, "Energy local minima found at gx = ", gx[local_min_idx])
    #plt.plot(gx, eig_arr, c='b')
    #plt.scatter(gx[local_min_idx], eig_arr[local_min_idx], c='r', marker = 'X')
    #plt.show()

    tol = tol/10
    for i in range(0, local_min_idx.size): #eigs_min.size
        gx_c = gx[local_min_idx[i]] #gx[ZEC_idx[i]]""" #first approx g_critical
        gx_lower = gx[local_min_idx[i]-1]#gx[ZEC_idx[i]-1]""" #one step back
        gx_higher = gx[local_min_idx[i]+1]#gx[ZEC_idx[i]+1]""" #one step forward

        delta_gam = (gx_higher - gx_lower)
        n_steps = (int((delta_gam/(0.5*tol))) + 1)
        gx_finer = np.linspace(gx_lower, gx_higher, n_steps) #high res gam around supposed zero energy crossing (local min)
        eig_arr_finer = np.zeros((gx_finer.size)) #new eigenvalue array
        for j in range(gx_finer.shape[0]):
            H_DB = HG0_DB + gx_finer[j]*HG_DB
            eigs_DB, U_DB = LA.eigh(H_DB)
            eig_arr_finer[j] = eigs_DB[int(k/2)] #k/2 -> lowest postive energy state

        min_idx_finer = np.array(argrelextrema(eig_arr_finer, np.less)[0]) #new local minima indices
        eigs_min_finer = eig_arr_finer[min_idx_finer] #isolating local minima
        #plt.plot(gx_finer, eig_arr_finer, c = 'b')
        #plt.scatter(gx_finer[min_idx_finer], eig_arr_finer[min_idx_finer], c='r', marker = 'X')
        #plt.plot(gx_finer, 0*gx_finer, c='k', lw=1)
        for m in range(eigs_min_finer.shape[0]):
            if eigs_min_finer[m] < tol:
                crossing_gam = gx_finer[min_idx_finer[m]]
                G_crit.append(crossing_gam)
                print("Crossing found at Gx = {} | E = {} meV".format(crossing_gam, eigs_min_finer[m]))
                #plt.scatter(G_crit, eigs_min_finer[m], c= 'r', marker = 'X')
            #plt.show()
    G_crit = np.array(G_crit)
    return G_crit
