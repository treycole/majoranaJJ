import sys
import time
import os
import gc

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.signal import argrelextrema
import scipy.linalg as LA
import scipy.sparse.linalg as spLA

import majoranaJJ.operators.sparse_operators as spop #sparse operators
from majoranaJJ.operators.potentials import Vjj #potential JJ
import majoranaJJ.lattice.nbrs as nb #neighbor arrays
import majoranaJJ.lattice.shapes as shps #lattice shapes
import majoranaJJ.modules.plots as plots #plotting functions
import majoranaJJ.modules.gamfinder as gamfinder
from majoranaJJ.modules.checkers import boundary_check as bc
import majoranaJJ.modules.checkers as check
###################################################
#Defining System
Nx = 3 #Number of lattice sites along x-direction
Ny = 290 #Number of lattice sites along y-direction
ax = 50 #lattice spacing in x-direction: [A]
ay = 50 #lattice spacing in y-direction: [A]
Wj = 40 #Junction region
cutx = 0 #width of nodule
cuty = 0 #height of nodule
Nx, Ny, cutx, cuty, Wj = check.junction_geometry_check(Nx, Ny, cutx, cuty, Wj)
print("Nx = {}, Ny = {}, cutx = {}, cuty = {}, Wj = {}".format(Nx, Ny, cutx, cuty, Wj))

Junc_width = Wj*ay*.10 #nm
SC_width = ((Ny - Wj)*ay*.10)/2 #nm
Nod_widthx = cutx*ax*.1 #nm
Nod_widthy = cuty*ay*.1 #nm
print("Nodule Width in x-direction = ", Nod_widthx, "(nm)")
print("Nodule Width in y-direction = ", Nod_widthy, "(nm)")
print("Junction Width = ", Junc_width, "(nm)")
print("Supercondicting Lead Width = ", SC_width, "(nm)")
###################################################coor = shps.square(Nx, Ny) #square lattice
coor = shps.square(Nx, Ny) #square lattice
NN = nb.NN_sqr(coor)
NNb = nb.Bound_Arr(coor)
lat_size = coor.shape[0]

Lx = (max(coor[:, 0]) - min(coor[:, 0]) + 1)*ax #Unit cell size in x-direction
Ly = (max(coor[:, 1]) - min(coor[:, 1]) + 1)*ay #Unit cell size in y-direction
###################################################
#Defining Hamiltonian parameters
alpha = 100 #Spin-Orbit Coupling constant: [meV*A]
phi = np.pi #SC phase difference
delta = 1 #Superconducting Gap: [meV]
Vsc = 0 #Amplitude of potential in SC region: [meV]
Vj = -5 #Amplitude of potential in junction region: [meV]
V = Vjj(coor, Wj = Wj, Vsc = Vsc, Vj = Vj, cutx = cutx, cuty = cuty)

mu_i = -5
mu_f = 10.0
res = 0.1
delta_mu = mu_f - mu_i
steps = int(delta_mu/(0.5*res)) + 1
mu = np.linspace(mu_i, mu_f, steps) #Chemical Potential: [meV]
dmu = -0.010347

k = 200
###################################################
#phase diagram mu vs gamx
H0 = spop.HBDG(coor, ax, ay, NN, NNb=NNb, Wj=Wj, cutx=cutx, cuty=cuty, V=V, mu=0, alpha=alpha, delta=delta, phi=phi, gamz=1e-5, qx=1e-5)
eigs_0, vecs_0 = spLA.eigsh(H0, k=k, sigma=0, which='LM')
vecs_0_hc = np.conjugate(np.transpose(vecs_0))

H_M0 = spop.HBDG(coor, ax, ay, NN, NNb=NNb, Wj=Wj, cutx=cutx, cuty=cuty, V=V, mu=0, alpha=alpha, delta=delta, phi=phi, qx=0)
H_M1 = spop.HBDG(coor, ax, ay, NN, NNb=NNb, Wj=Wj, cutx=cutx, cuty=cuty, V=V, mu=1, alpha=alpha, delta=delta, phi=phi, qx=0)
HM = H_M1 - H_M0
HM0_DB = np.dot(vecs_0_hc, H_M0.dot(vecs_0))
HM_DB = np.dot(vecs_0_hc, HM.dot(vecs_0))

eig_arr = np.zeros((mu.shape[0], k))
eig_arr_NB = np.zeros((mu.shape[0], k))
for i in range(mu.shape[0]):
    print(mu.shape[0]-i)
    H_DB = HM0_DB + (mu[i]+dmu)*HM_DB
    H = H_M0 + (mu[i]+dmu)*HM

    eigs_DB, U_DB = LA.eigh(H_DB)
    eig_arr_NB[i, :] = eigs_DB

    eigs, vecs = spLA.eigsh(H, k=k, sigma=0, which='LM')
    idx_sort = np.argsort(eigs)
    eigs = eigs[idx_sort]
    eig_arr[i, :] = eigs

for i in range(k):
    #if i % 2 == 0:
    plt.plot(mu, eig_arr[:, i], c = 'r')
    #else:
    plt.plot(mu, eig_arr_NB[:, i], c = 'b', ls = '--')

plt.xlabel(r'$\mu$ (meV)')
plt.ylabel("Energy (meV)")
plt.savefig("EvsGamx.png")
plt.show()
