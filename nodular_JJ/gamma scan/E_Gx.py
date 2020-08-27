import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sparse
import scipy.linalg as LA
import scipy.sparse.linalg as spLA

import majoranaJJ.operators.sparse.qmsops as spop #sparse operators
import majoranaJJ.lattice.nbrs as nb #neighbor arrays
import majoranaJJ.lattice.shapes as shps #lattice shapes
import majoranaJJ.modules.plots as plots #plotting functions
from majoranaJJ.modules.gamfinder import gamfinder as gf
from majoranaJJ.modules.gamfinder import gamfinder_lowE as gfLE
from majoranaJJ.operators.sparse.potentials import Vjj #potential JJ

#Defining System
Nx = 12 #Number of lattice sites along x-direction
Ny = 408 #Number of lattice sites along y-direction
ax = 50 #lattice spacing in x-direction: [A]
ay = 50 #lattice spacing in y-direction: [A]
Wj = 8 #Junction region
cutx = 3 #width of nodule
cuty = 3 #height of nodule

Junc_width = Wj*ay*.10 #nm
SC_width = ((Ny - Wj)*ay*.10)/2 #nm
Nod_widthx = cutx*ax*.1 #nm
Nod_widthy = cuty*ay*.1 #nm
print("Nodule Width in x-direction = ", Nod_widthx, "(nm)")
print("Nodule Width in y-direction = ", Nod_widthy, "(nm)")
print("Junction Width = ", Junc_width, "(nm)")
print("Supercondicting Lead Width = ", SC_width, "(nm)")

coor = shps.square(Nx, Ny) #square lattice
NN = nb.NN_sqr(coor)
NNb = nb.Bound_Arr(coor)
lat_size = coor.shape[0]
print("Lattice Size: ", lat_size)

Lx = (max(coor[:, 0]) - min(coor[:, 0]) + 1)*ax #Unit cell size in x-direction
Ly = (max(coor[:, 1]) - min(coor[:, 1]) + 1)*ay #Unit cell size in y-direction

###################################################

#Defining Hamiltonian parameters
steps = 100

alpha = 100 #Spin-Orbit Coupling constant: [meV*A]
gx = np.linspace(0, 1.2, 50)
phi = 0 #SC phase difference
delta = 1.0 #Superconducting Gap: [meV]
Vj = -50 #Amplitude of potential: [meV]
V = Vjj(coor, Wj = Wj, Vsc = 0, Vj = Vj, cutx = cutx, cuty = cuty)
MU = 55 #Chemical Potential: [meV]

###################################################

#Energy plot vs Zeeman energy in x-direction

k = 44 #number of perturbation energy eigs
Q = 1e-4*(np.pi/Lx)

H0 = spop.HBDG(coor, ax, ay, NN, NNb=NNb, Wj=Wj, cutx=cutx, cuty=cuty, V=V, mu=MU, alpha=alpha, delta=delta, phi=phi, qx=Q, periodicX=True)

eigs_0, vecs_0 = spLA.eigsh(H0, k=k, sigma = 0,  which='LM')
#idx_sort = np.argsort(eigs_0)
#print(eigs_0[idx_sort][int(k/2):])
vecs_0_hc = np.conjugate(np.transpose(vecs_0))

H_G0 =  spop.HBDG(coor, ax, ay, NN, NNb = NNb, Wj = Wj, cutx = cutx, cuty = cuty, V = V, mu = MU, gammax = 0, alpha = alpha, delta = delta, phi = phi, qx = 0, periodicX = True)
H_G1 = spop.HBDG(coor, ax, ay, NN, NNb = NNb, Wj = Wj, cutx = cutx, cuty = cuty, V = V, mu = MU, gammax = 1, alpha = alpha, delta = delta, phi = phi, qx = 0, periodicX = True)
HG = H_G1 - H_G0
HG0_DB = np.dot(vecs_0_hc, H_G0.dot(vecs_0))
HG_DB = np.dot(vecs_0_hc, HG.dot(vecs_0))

eig_arr = np.zeros((gx.shape[0], k))
eig_arr_NB = np.zeros((gx.shape[0], k))
for i in range(gx.shape[0]):
    print(gx.shape[0] - i, gx[i])
    H_DB = HG0_DB + gx[i]*HG_DB
    eigs_DB, U_DB = LA.eigh(H_DB)
    idx_sort = np.argsort(eigs_DB)
    eigs_DB = eigs_DB[idx_sort]
    eig_arr_NB[i, :] = eigs_DB

    energy = spop.EBDG(coor, ax, ay, NN, NNb=NNb, Wj=Wj, cutx=cutx, cuty=cuty, alpha=alpha, delta=delta, phi = phi, V=V, gammax=gx[i], mu=MU, qx=0, periodicX=True, k=k)
    eig_arr[i, :] = energy


for i in range(k):
    #if i % 2 == 0:
    plt.plot(gx, eig_arr[:, i], c = 'r')
    #else:
    plt.plot(gx, eig_arr_NB[:, i], c = 'b', ls = '--')

plt.xlabel(r'$E_z$ (meV)')
plt.ylabel("Energy (meV)")
plt.savefig("EvsGamx.png")
plt.show()
