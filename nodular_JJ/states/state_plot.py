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
Ny = 500 #Number of lattice sites along y-direction
ax = 50 #lattice spacing in x-direction: [A]
ay = 50 #lattice spacing in y-direction: [A]
Wj = 20 #Junction region
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
NN = nb.NN_Arr(coor) #neighbor array
NNb = nb.Bound_Arr(coor) #boundary array
lat_size = coor.shape[0]

Lx = (max(coor[:, 0]) - min(coor[:, 0]) + 1)*ax #Unit cell size in x-direction
Ly = (max(coor[:, 1]) - min(coor[:, 1]) + 1)*ay #Unit cell size in y-direction
plots.junction(coor, spop.Delta(coor, delta=1, Wj=Wj))
###################################################
#Hamiltonian Parameters
alpha = 100 #Spin-Orbit Coupling constant: [meV*A]
delta = 1 #meV
phi = 0
gx = 0.9 #parallel to junction: [meV]
gz = 0 #normal to plane of junction: [meV]
mu = 20 #Chemical Potential: [meV]
###################################################
k = 12
H = spop.HBDG(coor, ax, ay, NN, NNb=NNb, Wj=Wj, cutx=cutx, cuty=cuty, mu=mu, alpha=alpha, delta=delta, phi=phi, gamx=gx, qx=0, diff_g_factors = True)

eigs, vecs = spLA.eigsh(H, k=k, sigma=0, which='LM')
idx_sort = np.argsort(eigs)
eigs = eigs[idx_sort]
vecs = vecs[:, idx_sort]

n_es = 0 #nth excited state above zero energy
n = int(k/2) + n_es

N = coor.shape[0]
num_div = int(vecs.shape[0]/N)

idx_sort = np.argsort(eigs)
eigs = eigs[idx_sort]
vecs = vecs[:, idx_sort]

probdens = np.square(abs(vecs[:, n]))
map = np.zeros(N)
for i in range(num_div):
    map[:] = map[:] + probdens[i*N : (i+1)*N]

plots.state_cmap(coor, eigs, vecs, n = n, savenm = 'juncwidth = {} SCwidth = {} nodwidthx = {} nodwidthy = {} Delta = {} Alpha = {} phi = {} State_n={}.png'.format(Junc_width, SC_width, Nod_widthx, Nod_widthy, delta, alpha, phi, n_es))

oned_pd = []
for i in range(Ny):
    oned_pd.append(map[0 + Nx*i])

print(eigs[:])

plt.plot(oned_pd, c = 'r')
print("Energy of State", eigs[n])
y = np.linspace(263, 500, 1000)
#plt.plot(y, map[263*Nx]*np.exp(-2*(y-263)/63), c = 'b')
plt.show()
