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

def mu_scan(coor, ax, ay, NN, mui, muf, NNb=None, Wj=0, cutx=0, cuty=0, gamx=0, alpha=0, delta=0, phi=0, Vj=0):
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
        H = spop.HBDG(coor, ax, ay, NN, NNb=NNb, Wj=Wj, cutx=cutx, cuty=cuty, mu=muf, V=V, alpha=alpha, delta=delta, phi=phi, gamx=gamx, qx=k)
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
            E_min_i, k_min_i = GoldenSearch(coor, ax, ay, NN, mu_arr[j], k_min_arr[i], delta_k, NNb=NNb, Wj=Wj, cutx=cutx, cuty=cuty, gamx=gamx, delta=delta, alpha=alpha, phi=phi, Vj=Vj)
            k_MIN_arr[j,i] = k_min_i
            E_min_arr[j,i] = E_min_i
            k_min_arr[i] = k_min_i
            print(mu_arr.shape[0]-j, k_min_arr.shape[0]-i, k_min_i, E_min_i)
        E_min_Global_arr[j] = min(E_min_arr[j, :])

    return E_min_Global_arr, mu_arr

def first_scan(coor, ax, ay, NN, qx, k_sq_max, k_step, mu, NNb=None, Wj=0, cutx=0, cuty=0, V=0, alpha=0, delta=0, phi=0, gamx=0):
    k_sq = 0
    k = 0

    k_sq_list = []
    k_list = []
    E_list = []

    while True:
        #k_sq = k_sq + k_sq_step
        #k = np.sqrt(k_sq)
        k_sq = k**2
        print(k, k_sq)
        H = spop.HBDG(coor, ax, ay, NN, NNb=NNb, Wj=Wj, cutx=cutx, cuty=cuty, mu=mu, V=V, alpha=alpha, delta=delta, phi=phi, gamx=gamx, qx=k)
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
    return E_arr, k_arr

def mu_scan_2(coor, ax, ay, NN, mui, muf, NNb=None, Wj=0, cutx=0, cuty=0, gamx=0, alpha=0, delta=0, phi=0, Vj=0):
    V = Vjj(coor, Wj = Wj, Vsc = 0, Vj = Vj, cutx = cutx, cuty = cuty)
    k_sq = 0
    k = 0
    k_step = (0.001/Wj)#0.022/(Wj)
    tol = 5
    k_sq_max = max([4*muf/(const.xi), 4*(muf-Vj)/const.xi])

    #getting lowest energy band as a first approximation for the local minima
    E_arr, k_arr = first_scan(coor, ax, ay, NN, k, k_sq_max, k_step, muf, Wj=Wj, NNb=NNb, cutx=cutx, cuty=cuty, gamx=gamx, alpha=alpha, delta=delta, phi=phi, V=V)

    #local minima indices, concatenating index 0 for edge case
    local_min_idx = np.array(argrelextrema(E_arr, np.less)[0])
    local_min_idx = np.concatenate((np.array([0]), local_min_idx))

    #local minima energies along with the corresponding k-values
    min_energy = E_arr[local_min_idx]
    k_min_arr = k_arr[local_min_idx]

    #plt.plot(k_arr, E_arr)
    #plt.scatter(k_min_arr, min_energy, c='r', marker = 'X')

    delta_k = k_step #how much k is changed in golden search
    delta_mu = 0.01 #changing mu a small ammount
    mu_steps = int((muf-mui)/delta_mu + 1)
    mu_arr = np.linspace(muf, mui, mu_steps)

    #E_arr2, k_arr2 = first_scan(coor, ax, ay, NN, k, k_sq_max, k_step, mu_arr[1], NNb=NNb, cutx=cutx, cuty=cuty, Wj=Wj, gamx=gamx, alpha=alpha, delta=delta, phi=phi, V=V)
    #local_min_idx2 = np.array(argrelextrema(E_arr2, np.less)[0])
    #local_min_idx2 = np.concatenate((np.array([0]), local_min_idx2))

    #min_energy2 = E_arr2[local_min_idx2]
    #k_min_arr2 = k_arr2[local_min_idx2]

    #plt.plot(k_arr2, E_arr2, c = 'r', ls='--')
    #plt.scatter(k_min_arr2, min_energy2, c='b', marker = 'o')
    #plt.show()
    #sys.exit()

    #array for the new local minima in energy found by GoldenSearch
    E_min_GS = np.zeros((mu_arr.shape[0], k_min_arr.shape[0]))
    k_min_GS = np.zeros((mu_arr.shape[0], k_min_arr.shape[0]))
    E_min_Global = np.zeros(mu_arr.shape[0])

    #first step in mu
    for i in range(k_min_arr.shape[0]):
        E_min_GS[0, i], k_min_GS[0, i] = GoldenSearch(coor, ax, ay, NN, muf, k_min_arr[i], delta_k, NNb=NNb, Wj=Wj, cutx=cutx, cuty=cuty, gamx=gamx, delta=delta, alpha=alpha, phi=phi, V=V)

        k_min_arr[i] = k_min_GS[0, i]

    #plt.scatter(k_arr, E_arr, s=2, c='r')
    #plt.scatter(k_min_GS[0, :], E_min_GS[0, :], c='b', marker='x')
    #plt.show()
    #sys.exit()
    #print("kmin", k_min_arr)
    #for i in range(k_min_arr.shape[0]):
    #    print("Emin_i", E_min_GS[0, i])

    #finding how much the minima moves after changing mu
    #find derivative, determines direction of change
    #increment to left or right until minima is found
    change_in_k = np.zeros(k_min_arr.shape[0])
    for i in range(k_min_arr.shape[0]):
        H = spop.HBDG(coor, ax, ay, NN, NNb=NNb, Wj=Wj, cutx=cutx, cuty=cuty, mu=mu_arr[1], V=V, alpha=alpha, delta=delta, phi=phi, gamx=gamx, qx=k_min_arr[i])
        eigs, vecs = spLA.eigsh(H, k=4, sigma=0, which='LM')
        idx_sort = np.argsort(eigs)
        eigs = eigs[idx_sort]
        E1 = eigs[2]

        dk = (delta_k/100)

        H = spop.HBDG(coor, ax, ay, NN, NNb=NNb, Wj=Wj, cutx=cutx, cuty=cuty, mu=mu_arr[1], V=V, alpha=alpha, delta=delta, phi=phi, gamx=gamx, qx=k_min_arr[i]+dk)
        eigs, vecs = spLA.eigsh(H, k=4, sigma=0, which='LM')
        idx_sort = np.argsort(eigs)
        eigs = eigs[idx_sort]
        E2 = eigs[2]

        dE_dK = (E2-E1)/dk
        krange = [None, None]
        if dE_dK > 0:
            krange[1] = k_min_arr[i]+dk
            El = E1
            ER = E2
            while El < ER:
                knew = k_min_arr[i] - dk
                H = spop.HBDG(coor, ax, ay, NN, NNb=NNb, Wj=Wj, cutx=cutx, cuty=cuty, mu=mu_arr[1], V=V, alpha=alpha, delta=delta, phi=phi, gamx=gamx, qx=knew)
                eigs, vecs = spLA.eigsh(H, k=4, sigma=0, which='LM')
                idx_sort = np.argsort(eigs)
                eigs = eigs[idx_sort]
                ER = El
                El = eigs[2]
            krange[0] = knew - dk
            k_center = knew - dk
        else:
            krange[0] = k_min_arr[i]-dk
            El = E1
            ER = E2
            while El > ER:
                knew = k_min_arr[i] + dk
                H = spop.HBDG(coor, ax, ay, NN, NNb=NNb, Wj=Wj, cutx=cutx, cuty=cuty, mu=mu_arr[1], V=V, alpha=alpha, delta=delta, phi=phi, gamx=gamx, qx=knew)
                eigs, vecs = spLA.eigsh(H, k=4, sigma=0, which='LM')
                idx_sort = np.argsort(eigs)
                eigs = eigs[idx_sort]
                ER = El
                El = eigs[2]
            krange[1] = knew + dk
            k_center = knew + dk

        change_in_k[i] = krange[1]-krange[0]
        print("dE_dK", dE_dK, k_min_arr[i])

    E_min_arr = np.zeros((mu_arr.shape[0], k_min_arr.shape[0]))
    k_MIN_arr = np.zeros((mu_arr.shape[0], k_min_arr.shape[0]))
    E_min_Global_arr = np.zeros(mu_arr.shape[0])
    for j in range(mu_arr.shape[0]):
        for i in range(k_min_arr.shape[0]):
            E_min_i, k_min_i = GoldenSearch(coor, ax, ay, NN, mu_arr[j], k_min_arr[i], change_in_k[i], NNb=NNb, Wj=Wj, cutx=cutx, cuty=cuty, gamx=gamx, delta=delta, alpha=alpha, phi=phi, V=V)
            k_MIN_arr[j,i] = k_min_i
            E_min_arr[j,i] = E_min_i
            k_min_arr[i] = k_min_i
            print(mu_arr.shape[0]-j, k_min_arr.shape[0]-i, k_min_i, E_min_i)
        E_min_Global_arr[j] = min(E_min_arr[j, :])

    return E_min_Global_arr, mu_arr, E_min_arr, k_MIN_arr

def GoldenSearch(coor, ax, ay, NN, mu, kcenter, deltak, NNb=None, Wj=0, cutx=0, cuty=0, gamx=0, delta=0, alpha=0, phi=0, V=0):
    ka = kcenter - (deltak)
    kb = kcenter + (deltak)
    GR = 0.6180339887 #(np.sqrt(5) + 1.0)/2.0
    omgr = 1 - GR
    d = GR*(kb-ka)
    k1 = ka + omgr*(kb-ka)
    k2 = ka + GR*(kb-ka)

    if kcenter == 0:
        H1 = spop.HBDG(coor, ax, ay, NN, NNb=NNb, Wj=Wj, cutx=cutx, cuty=cuty, mu=mu, V=V, alpha=alpha, delta=delta, phi=phi, gamx=gamx, qx=0)
        eigs1, vecs1 = spLA.eigsh(H1, k=4, sigma=0, which='LM')
        idx_sort1 = np.argsort(eigs1)
        eigs1 = eigs1[idx_sort1]
        f1 = min(np.abs(eigs1))
        return f1, kcenter


    H1 = spop.HBDG(coor, ax, ay, NN, NNb=NNb, Wj=Wj, cutx=cutx, cuty=cuty, mu=mu, V=V, alpha=alpha, delta=delta, phi=phi, gamx=gamx, qx=k1)
    eigs1, vecs1 = spLA.eigsh(H1, k=4, sigma=0, which='LM')
    idx_sort1 = np.argsort(eigs1)
    eigs1 = eigs1[idx_sort1]
    f1 = min(np.abs(eigs1))

    H2 = spop.HBDG(coor, ax, ay, NN, NNb=NNb, Wj=Wj, cutx=cutx, cuty=cuty, mu=mu, V=V, alpha=alpha, delta=delta, phi=phi, gamx=gamx, qx=k2)
    eigs2, vecs2 = spLA.eigsh(H2, k=4, sigma=0, which='LM')
    idx_sort2 = np.argsort(eigs2)
    eigs2 = eigs2[idx_sort2]
    f2 = min(np.abs(eigs2))

    k1_list = [ka]
    k2_list = [kb]
    eigs1_list = [f1]
    eigs2_list = [f2]
    #Ha = spop.HBDG(coor, ax, ay, NN, NNb=NNb, Wj=Wj, cutx=cutx, cuty=cuty, mu=mu, V=V, alpha=alpha, delta=delta, phi=phi, gamx=gamx, qx=ka)
    #eigsa, vecsa = spLA.eigsh(Ha, k=4, sigma=0, which='LM')
    #idx_sorta = np.argsort(eigsa)
    #eigsa = eigsa[idx_sorta]
    #fa = min(np.abs(eigsa))

    #Hb = spop.HBDG(coor, ax, ay, NN, NNb=NNb, Wj=Wj, cutx=cutx, cuty=cuty, mu=mu, V=V, alpha=alpha, delta=delta, phi=phi, gamx=gamx, qx=kb)
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
            k1 = ka + omgr*(kb-ka)

            H1 = spop.HBDG(coor, ax, ay, NN, NNb=NNb, Wj=Wj, cutx=cutx, cuty=cuty, mu=mu, V=V, alpha=alpha, delta=delta, phi=phi, gamx=gamx, qx=k1)
            eigs1, vecs1 = spLA.eigsh(H1, k=4, sigma=0, which='LM')
            idx_sort1 = np.argsort(eigs1)
            eigs1 = eigs1[idx_sort1]

            f1 = min(np.abs(eigs1))

            k1_list.append(ka)
            k2_list.append(kb)
            eigs1_list.append(f1)
            eigs2_list.append(f2)
        else:
            ka = k1
            k1 = k2
            f1 = f2
            d = GR*(kb-ka)
            k2 = ka + GR*(kb-ka)

            H2 = spop.HBDG(coor, ax, ay, NN, NNb=NNb, Wj=Wj, cutx=cutx, cuty=cuty, mu=mu, V=V, alpha=alpha, delta=delta, phi=phi, gamx=gamx, qx=k2)
            eigs2, vecs2 = spLA.eigsh(H2, k=4, sigma=0, which='LM')
            idx_sort2 = np.argsort(eigs2)
            eigs2 = eigs1[idx_sort2]

            f2 = min(np.abs(eigs2))

            k1_list.append(ka)
            k2_list.append(kb)
            eigs1_list.append(f1)
            eigs2_list.append(f2)

        tol = 1e-8 #np.sqrt((delta/1000000)*(2/const.xi))
        if abs(ka-kb) < tol:
            if f1 < f2:
                return f1, k1
            else:
                return f2, k2
