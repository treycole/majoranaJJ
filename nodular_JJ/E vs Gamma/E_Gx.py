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
from majoranaJJ.operators.potentials.barrier_leads import V_BL

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
V0 = 50 #Amplitude of potential : [meV]
V = V_BL(coor, Wj = Wj, cutx=cutx, cuty=cuty, V0 = V0)
MU = 55 #Chemical Potential: [meV]

###################################################

#Energy plot vs Zeeman energy in x-direction

#seeing the junction layout
#D_test = spop.Delta(coor, Wj = Wj, delta = 1, cutx = cutx, cuty = cuty)
#plots.junction(coor, D_test)

k = 200 #number of perturbation energy eigs
Q = 0.002#1e-4*(np.pi/Lx)

H0 = spop.HBDG(coor, ax, ay, NN, NNb=NNb, Wj=Wj, cutx=cutx, cuty=cuty, V=V, mu=MU, alpha=alpha, delta=delta, phi=phi, qx=Q, periodicX=True)

eigs_0, vecs_0 = spLA.eigsh(H0, k=k, sigma = 0,  which='LM')
#idx_sort = np.argsort(eigs_0)
#print(eigs_0[idx_sort][int(k/2):])
vecs_0_hc = np.conjugate(np.transpose(vecs_0))

eig_arr = np.zeros((gx.shape[0], k))
eig_arr_NB = np.zeros((gx.shape[0], k))

H_G0 =  spop.HBDG(coor, ax, ay, NN, NNb = NNb, Wj = Wj, cutx = cutx, cuty = cuty, V = V, mu = MU, gammax = 0, alpha = alpha, delta = delta, phi = phi, qx = Q, periodicX = True)

H_G1 = spop.HBDG(coor, ax, ay, NN, NNb = NNb, Wj = Wj, cutx = cutx, cuty = cuty, V = V, mu = MU, gammax = 1, alpha = alpha, delta = delta, phi = phi, qx = Q, periodicX = True)

HG = H_G1 - H_G0
HG0_DB = np.dot(vecs_0_hc, H_G0.dot(vecs_0))
HG_DB = np.dot(vecs_0_hc, HG.dot(vecs_0))

for i in range(gx.shape[0]):
    print(gx.shape[0] - i, gx[i])
    H_DB = HG0_DB + gx[i]*HG_DB
    #H = H_G0 + gx[i]*HG
    #eigs, vecs = spLA.eigsh(H, k=k, sigma=0, which='LM')
    eigs_DB, U_DB = LA.eigh(H_DB)

    #idx_sort = np.argsort(eigs)
    #eigs = eigs[idx_sort]
    #eig_arr[i, :] = eigs

    idx_sort = np.argsort(eigs_DB)
    eigs_DB = eigs_DB[idx_sort]
    eig_arr_NB[i, :] = eigs_DB

for i in range(k):
    #if i % 2 == 0:
    #plt.plot(gx, eig_arr[:, i], c = 'r')
    #else:
    plt.plot(gx, eig_arr_NB[:, i], c = 'b', ls = '--')

plt.xlabel(r'$E_z$ (meV)')
plt.ylabel("Energy (meV)")
plt.savefig("EvsGamx.png")
plt.show()
