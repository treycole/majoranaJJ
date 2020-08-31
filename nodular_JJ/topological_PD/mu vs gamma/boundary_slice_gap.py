import sys
import os
import gc

import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sparse
from scipy.signal import argrelextrema
import scipy.linalg as LA
import scipy.sparse.linalg as spLA

import majoranaJJ.operators.sparse.qmsops as spop #sparse operators
import majoranaJJ.lattice.nbrs as nb #neighbor arrays
import majoranaJJ.lattice.shapes as shps #lattice shapes
import majoranaJJ.modules.plots as plots #plotting functions
import majoranaJJ.modules.gamfinder as gamfinder #finds critical gamma
from majoranaJJ.operators.sparse.potentials import Vjj #potential JJ

###################################################
#Defining System
Nx = 3 #Number of lattice sites along x-direction
Ny = 60 #Number of lattice sites along y-direction
ax = 50 #lattice spacing in x-direction: [A]
ay = 50 #lattice spacing in y-direction: [A]
Wj = 8 #Junction region
cutx = 0 #width of nodule
cuty = 0 #height of nodule

Junc_width = Wj*ay*.10 #nm
SC_width = ((Ny - Wj)*ay*.10)/2 #nm
Nod_widthx = cutx*ax*.1 #nm
Nod_widthy = cuty*ay*.1 #nm
print("Nodule Width in x-direction = ", Nod_widthx, "(nm)")
print("Nodule Width in y-direction = ", Nod_widthy, "(nm)")
print("Junction Width = ", Junc_width, "(nm)")
print("Supercondicting Lead Width = ", SC_width, "(nm)")
###################################################
coor = shps.square(Nx, Ny) #square lattice
NN = nb.NN_sqr(coor)
NNb = nb.Bound_Arr(coor)
lat_size = coor.shape[0]
print("Lattice Size: ", lat_size)

Lx = (max(coor[:, 0]) - min(coor[:, 0]) + 1)*ax #Unit cell size in x-direction
Ly = (max(coor[:, 1]) - min(coor[:, 1]) + 1)*ay #Unit cell size in y-direction
###################################################
#Defining Hamiltonian parameters
alpha = 100 #Spin-Orbit Coupling constant: [meV*A]
phi = np.pi #SC phase difference
delta = 1 #Superconducting Gap: [meV]
Vsc = 0 #SC potential: [meV]
Vj = 0 #Junction potential: [meV]
V = Vjj(coor, Wj = Wj, Vsc = Vsc, Vj = Vj, cutx = cutx, cuty = cuty)

mu = [1, 2, 3, 4, 5, 6, 7, 8, 9] #meV

gi = 0
gf = 1.0
res = 0.005
steps_gam = int((gf - gi)/(0.5*res)) + 1
gx = np.linspace(gi, gf, steps_gam)

q_steps = 51
qx = np.linspace(0, np.pi/Lx, q_steps) #kx in the first Brillouin zone
###################################################
k = 44
LE_Bands = np.zeros((qx.shape[0], len(mu), gx.shape[0]))
for i in range(q_steps):
    for j in range(len(mu)):
        if i == 0:
            Q = 1e-4*(np.pi/Lx)
        else:
            Q = qx[i]

        H0 = spop.HBDG(coor, ax, ay, NN, NNb=NNb, Wj=Wj, cutx=cutx, cuty=cuty, V=V, mu=mu[j], alpha=alpha, delta=delta, phi=phi, gammax=1e-4, qx=Q, periodicX=True) #gives low energy basis

        eigs_0, vecs_0 = spLA.eigsh(H0, k=k, sigma=0, which='LM')
        vecs_0_hc = np.conjugate(np.transpose(vecs_0)) #hermitian conjugate

        H_G0 = spop.HBDG(coor, ax, ay, NN, NNb=NNb, Wj=Wj, cutx=cutx, cuty=cuty, V=V, mu=mu[j], gammax=0, alpha=alpha, delta=delta, phi=phi, qx=qx[i], periodicX=True) #Matrix that consists of everything in the Hamiltonian except for the Zeeman energy in the x-direction
        H_G1 = spop.HBDG(coor, ax, ay, NN, NNb=NNb, Wj=Wj, cutx=cutx, cuty=cuty, V=V, mu=mu[j], gammax=1, alpha=alpha, delta=delta, phi=phi, qx=qx[i], periodicX=True) #Hamiltonian with ones on Zeeman energy along x-direction sites
        HG = H_G1 - H_G0 #the proporitonality matrix for gamma-x, it is ones along the sites that have a gamma value
        HG0_DB = np.dot(vecs_0_hc, H_G0.dot(vecs_0))
        HG_DB = np.dot(vecs_0_hc, HG.dot(vecs_0))
        for g in range(gx.shape[0]):
            print(qx.shape[0]-i, len(mu)-j, gx.shape[0]-g)
            H_DB = HG0_DB + gx[g]*HG_DB
            eigs_DB, U_DB = LA.eigh(H_DB)
            LE_Bands[i, j, g] = eigs_DB[int(k/2)]

gap = np.zeros((len(mu), gx.shape[0]))
q_minima = []
for i in range(LE_Bands.shape[1]):
    for j in range(LE_Bands.shape[2]):
        eig_min_idx = np.array(argrelextrema(LE_Bands[:, i, j], np.less)[0])
        q_minima.append(qx[eig_min_idx])
        gap[i, j] = min(LE_Bands[:, i, j])

q_minima = np.array(q_minima)
print(gap)
