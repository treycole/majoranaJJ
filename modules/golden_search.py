#import meff, hbar
import numpy as np
import scipy.linalg as LA
import scipy.sparse.linalg as spLA
from scipy.signal import argrelextrema
import sys
import matplotlib.pyplot as plt
import majoranaJJ.operators.sparse_operators as spop #sparse operators
from majoranaJJ.operators.potentials import Vjj as Vjj #potential JJ
from majoranaJJ.modules import constants as const

m_eff = const.meff
h_bar = const.hbar*10**3

def mu_scan(coor, ax, ay, NN, mui, muf, NNb=None, Wj=0, cutx=0, cuty=0, gx=0, alpha=0, delta=0, phi=0, Vj=0):
    V = Vjj(coor, Wj = Wj, Vsc = 0, Vj = Vj, cutx = cutx, cuty = cuty)
    k_sq = 0
    k = 0
    k_step = (0.001/Wj)#0.022/(Wj)
    #k_sq_step = 0.0005/(Wj**2)
    k_sq_list = []
    k_list = []
    E_list = []
    tol = 5
    k_sq_max = max([4*muf/(const.xi), 4*(muf-Vj)/const.xi])
    while True:
        #k_sq = k_sq + k_sq_step
        #k = np.sqrt(k_sq)
        k_sq = k**2
        print(k, k_sq)
        H = spop.HBDG(coor, ax, ay, NN, NNb=NNb, Wj=Wj, cutx=cutx, cuty=cuty, mu=muf, V=V, alpha=alpha, delta=delta, phi=phi, gamx=gx, qx=k)
        eigs, vecs = spLA.eigsh(H, k=4, sigma=0, which='LM')
        idx_sort = np.argsort(eigs)
        eigs = eigs[idx_sort]

        k_sq_list.append(k_sq)
        k_list.append(k)
        E_list.append(np.min(np.abs(eigs)))

        if k_sq > k_sq_max:
            break
        k += k_step

    k_arr = np.array(k_list)
    E_arr = np.array(E_list)
    local_min_idx = np.array(argrelextrema(E_arr, np.less)[0])
    local_min_idx = np.concatenate((np.array([0]), local_min_idx))
    min_energy = E_arr[local_min_idx]
    k_min_arr = k_arr[local_min_idx]
    print(k_min_arr[0])
    #sys.exit()

    #plt.scatter(k_min_arr, min_energy, c='r', marker = 'X')
    #plt.plot(k_arr, E_arr)
    #plt.show()
    #delta_k_sq = 2*m_eff*delta_mu/(hbar**2)
    #delta_k = np.sqrt(delta_k_sq)*10
    delta_k_arr = k_min_arr[1:] - k_min_arr[:-1]
    delta_k = min(delta_k_arr)
    #print(delta_k)
    delta_k = k_step

    delta_mu = (const.xi/2)*(delta_k/4)**2
    delta_mu = 0.01
    #print(delta_mu)
    Nmu = int( (muf-mui)/delta_mu + 1 )
    mu_arr = np.linspace(muf, mui, Nmu)

    E_min_arr = np.zeros((mu_arr.shape[0], k_min_arr.shape[0]))
    k_MIN_arr = np.zeros((mu_arr.shape[0], k_min_arr.shape[0]))
    E_min_Global_arr = np.zeros(mu_arr.shape[0])

    for j in range(mu_arr.shape[0]):
        for i in range(k_min_arr.shape[0]):
            E_min_i, k_min_i = GoldenSearch(coor, ax, ay, NN, mu_arr[j], k_min_arr[i], delta_k, NNb=NNb, Wj=Wj, cutx=cutx, cuty=cuty, gx=gx, delta=delta, alpha=alpha, phi=phi, Vj=Vj)
            k_MIN_arr[j,i] = k_min_i
            E_min_arr[j,i] = E_min_i
            k_min_arr[i] = k_min_i
            print(mu_arr.shape[0]-j, k_min_arr.shape[0]-i, k_min_i, E_min_i)
        E_min_Global_arr[j] = min(E_min_arr[j, :])

    return E_min_Global_arr, mu_arr

def GoldenSearch(coor, ax, ay, NN, mu, kcenter, deltak, NNb=None, Wj=0, cutx=0, cuty=0, gx=0, delta=0, alpha=0, phi=0, Vj=0):
    V = Vjj(coor, Wj = Wj, Vsc = 0, Vj = Vj, cutx = cutx, cuty = cuty)
    ka = kcenter - (deltak/2)
    kb = kcenter + (deltak/2)
    GR = 0.618
    d = GR*(kb-ka)
    k1 = kb - d
    k2 = ka + d

    H1 = spop.HBDG(coor, ax, ay, NN, NNb=NNb, Wj=Wj, cutx=cutx, cuty=cuty, mu=mu, V=V, alpha=alpha, delta=delta, phi=phi, gamx=gx, qx=k1)
    eigs1, vecs1 = spLA.eigsh(H1, k=4, sigma=0, which='LM')
    idx_sort1 = np.argsort(eigs1)
    eigs1 = eigs1[idx_sort1]
    f1 = min(np.abs(eigs1))

    H2 = spop.HBDG(coor, ax, ay, NN, NNb=NNb, Wj=Wj, cutx=cutx, cuty=cuty, mu=mu, V=V, alpha=alpha, delta=delta, phi=phi, gamx=gx, qx=k2)
    eigs2, vecs2 = spLA.eigsh(H2, k=4, sigma=0, which='LM')
    idx_sort2 = np.argsort(eigs2)
    eigs2 = eigs2[idx_sort2]
    f2 = min(np.abs(eigs2))

    #Ha = spop.HBDG(coor, ax, ay, NN, NNb=NNb, Wj=Wj, cutx=cutx, cuty=cuty, mu=mu, V=V, alpha=alpha, delta=delta, phi=phi, gamx=gx, qx=ka)
    #eigsa, vecsa = spLA.eigsh(Ha, k=4, sigma=0, which='LM')
    #idx_sorta = np.argsort(eigsa)
    #eigsa = eigsa[idx_sorta]
    #fa = min(np.abs(eigsa))

    #Hb = spop.HBDG(coor, ax, ay, NN, NNb=NNb, Wj=Wj, cutx=cutx, cuty=cuty, mu=mu, V=V, alpha=alpha, delta=delta, phi=phi, gamx=gx, qx=kb)
    #eigsb, vecsb = spLA.eigsh(Hb, k=4, sigma=0, which='LM')
    #idx_sortb = np.argsort(eigsb)
    #eigsb = eigsb[idx_sortb]
    #fb = min(np.abs(eigsb))

    #plt.scatter(k1, f1, c='blue')
    #plt.scatter(k2, f2, c='red')
    #plt.scatter(ka, fa, c= 'green')
    #plt.scatter(kb, fb, c = 'orange')
    #plt.show()
    #print(k1, f1)
    #print(k2, f2)
    #print(kcenter, fc)
    #print(ka, fa)
    #print(kb, fb)

    while True:
        if f1 < f2:
            kb = k2
            k2 = k1
            f2 = f1
            d = GR*(kb-ka)
            k1 = kb-d

            H = spop.HBDG(coor, ax, ay, NN, NNb=NNb, Wj=Wj, cutx=cutx, cuty=cuty, mu=mu, V=V, alpha=alpha, delta=delta, phi=phi, gamx=gx, qx=k1)
            eigs1, vecs1 = spLA.eigsh(H, k=4, sigma=0, which='LM')
            idx_sort1 = np.argsort(eigs1)
            eigs1 = eigs1[idx_sort1]
            f1 = min(np.abs(eigs1))

        else:
            ka = k1
            k1 = k2
            f1 = f2
            d = GR*(kb-ka)
            k2 = ka + d
            H = spop.HBDG(coor, ax, ay, NN, NNb=NNb, Wj=Wj, cutx=cutx, cuty=cuty, mu=mu, V=V, alpha=alpha, delta=delta, phi=phi, gamx=gx, qx=k2)
            eigs2, vecs2 = spLA.eigsh(H, k=4, sigma=0, which='LM')
            idx_sort2 = np.argsort(eigs2)
            eigs2 = eigs1[idx_sort2]
            f2 = min(np.abs(eigs2))

        tol = np.sqrt((delta/1000000)*(2/const.xi))
        #print(tol)
        if (kb-ka) < tol:
            if f1 < f2:
                return f1, k1
            else:
                return f2, k2
