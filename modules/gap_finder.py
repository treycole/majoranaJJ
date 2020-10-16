import majoranaJJ.operators.sparse_operators as spop #sparse operators
from majoranaJJ.modules import constants as const
import numpy as np
import scipy.linalg as LA
import scipy.sparse.linalg as spLA
from scipy.signal import argrelextrema
import sys
import matplotlib.pyplot as plt

def minima(arr):
    abs_min = min(arr)
    for i in range(arr.shape[0]):
        min_temp = arr[i]
        if min_temp <= abs_min:
            abs_min = min_temp
            idx = i
    return abs_min, idx

def gap_finder(
    coor, NN, NNb, ax, ay, mu, gx,
    Wj = 0, cutx = 0, cuty = 0,
    V = 0, meff_normal = 0.026*const.m0, meff_sc = 0.026*const.m0,
    g_normal = 26, g_sc = 26,
    alpha = 0, delta = 0, phi = 0,
    Tesla = False, diff_g_factors = True,  Rfactor = 0, diff_alphas = False, diff_meff = False,
    k = 4, steps = 21
    ):

    Lx = (max(coor[:, 0]) - min(coor[:, 0]) + 1)*ax #Unit cell size in x-direction
    qx = np.linspace(0, np.pi/Lx, steps) #kx in the first Brillouin zone

    bands = np.zeros((steps, k))
    for i in range(steps):
        print(steps - i)
        H = spop.HBDG(coor, ax, ay, NN, NNb=NNb, Wj=Wj, cutx=cutx, cuty=cuty, mu=mu, V=V, alpha=alpha, delta=delta, phi=phi, gamx=gx, qx=qx[i], Tesla=Tesla, diff_g_factors=diff_g_factors, diff_alphas=diff_alphas, diff_meff=diff_meff )
        eigs, vecs = spLA.eigsh(H, k=k, sigma=0, which='LM')
        idx_sort = np.argsort(eigs)
        eigs = eigs[idx_sort]
        bands[i, :] = eigs

    min_energy = []
    qx_crit_arr = []

    lowest_energy_band = bands[:, int(k/2)]
    local_min_idx = np.array(argrelextrema(lowest_energy_band, np.less)[0])

    #checking edge cases
    min_energy.append(bands[0, int(k/2)])
    qx_crit_arr.append(qx[0])
    min_energy.append(bands[-1, int(k/2)])
    qx_crit_arr.append(qx[-1])

    print(local_min_idx.size, "Energy local minima found at kx = ", qx[local_min_idx])

    #for i in range(bands.shape[1]):
    #    plt.plot(qx, bands[:, i], c ='mediumblue', linestyle = 'solid')
    #    plt.plot(-qx, bands[:, i], c ='mediumblue', linestyle = 'solid')
    #plt.scatter(qx[local_min_idx], lowest_energy_band[local_min_idx], c='r', marker = 'X')
    #plt.show()
    #sys.exit()

    for i in range(0, local_min_idx.size): #eigs_min.size
        qx_c = qx[local_min_idx[i]] #first approx g_critical
        qx_lower = qx[local_min_idx[i]-1] #one step back
        qx_higher = qx[local_min_idx[i]+1] #one step forward

        n_steps = 21
        qx_finer = np.linspace(qx_lower, qx_higher, n_steps) #high res gam around supposed zero energy crossing (local min)
        bands_finer = np.zeros((qx_finer.size, k)) #new eigenvalue array
        for j in range(qx_finer.shape[0]):
            H = spop.HBDG(coor, ax, ay, NN, NNb=NNb, Wj=Wj, cutx=cutx, cuty=cuty, mu=mu, V=V, alpha=alpha, delta=delta, phi=phi, gamx=gx, qx=qx_finer[j], Tesla=Tesla, diff_g_factors=diff_g_factors, diff_alphas=diff_alphas, diff_meff=diff_meff )
            eigs, vecs = spLA.eigsh(H, k=k, sigma=0, which='LM')
            idx_sort = np.argsort(eigs)
            eigs = eigs[idx_sort]
            bands_finer[j, :] = eigs

        leb_finer = bands_finer[:, int(k/2)]
        min_idx_finer = np.array(argrelextrema(leb_finer, np.less)[0]) #new local minima indices
        leb_min_finer = leb_finer[min_idx_finer] #isolating local minima
        qx_crit = qx_finer[min_idx_finer]
        #for b in range(bands.shape[1]):
        #    plt.plot(qx_finer, bands_finer[:, b], c ='mediumblue', linestyle = 'solid')
            #plt.plot(-qx_finer, bands_finer[:, b], c ='mediumblue', linestyle = 'solid')
        #plt.scatter(qx_finer[min_idx_finer], leb_min_finer, c='r', marker = 'X')
        #plt.show()
        leb_min_finer = np.array(leb_min_finer)
        GAP, IDX = minima(leb_min_finer)
        min_energy.append(GAP)
        qx_crit_arr.append(qx_crit[IDX])

    qx_crit_arr.append(np.pi/Lx)
    qx_crit_arr
    min_energy = np.array(min_energy)
    gap, idx = minima(min_energy)
    qx_crit = qx_crit_arr[idx]
    return [gap, qx_crit]
