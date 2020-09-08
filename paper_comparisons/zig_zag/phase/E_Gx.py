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
import majoranaJJ.modules.gamfinder as gamfinder
from majoranaJJ.operators.sparse.potentials import Vjj #potential JJ

#Defining System
Nx = 3 #Number of lattice sites along x-direction
Ny = 360 #Number of lattice sites along y-direction
ax = 50 #lattice spacing in x-direction: [A]
ay = 50 #lattice spacing in y-direction: [A]
Wj = 40 #Junction region
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

coor = shps.square(Nx, Ny) #square lattice
NN = nb.NN_sqr(coor)
NNb = nb.Bound_Arr(coor)
lat_size = coor.shape[0]
print("Lattice Size: ", lat_size)

Lx = (max(coor[:, 0]) - min(coor[:, 0]) + 1)*ax #Unit cell size in x-direction
Ly = (max(coor[:, 1]) - min(coor[:, 1]) + 1)*ay #Unit cell size in y-direction
###################################################
#Defining Hamiltonian parameters
alpha = 200 #Spin-Orbit Coupling constant: [meV*A]
phi = np.pi #SC phase difference
delta = 1.0 #Superconducting Gap: [meV]
Vj = 0 #Amplitude of potential: [meV]
V = Vjj(coor, Wj = Wj, Vsc = 0, Vj = Vj, cutx = cutx, cuty = cuty)
MU = 0 #Chemical Potential: [meV]

tol = 0.02
delta_B = abs(5 - 0)
bsteps = int((delta_B/(0.5*tol))) + 1
Bx = np.linspace(0, 5, bsteps)
###################################################
#Energy plot vs Zeeman energy in x-direction

k = 100 #number of perturbation energy eigs
Q = 1e-4*(np.pi/Lx)

H0 = spop.HBDGb(coor, ax, ay, NN, NNb=NNb, Wj=Wj, cutx=cutx, cuty=cuty, V=V, mu=MU, Bx=1e-4, alpha=alpha, delta=delta, phi=phi, qx=Q, periodicX=True)

eigs_0, vecs_0 = spLA.eigsh(H0, k=k, sigma = 0,  which='LM')
vecs_0_hc = np.conjugate(np.transpose(vecs_0))

H_G0 =  spop.HBDGb(coor, ax, ay, NN, NNb = NNb, Wj = Wj, cutx = cutx, cuty = cuty, V = V, mu = MU, Bx = 0, alpha = alpha, delta = delta, phi = phi, qx = 0, periodicX = True)
H_G1 = spop.HBDGb(coor, ax, ay, NN, NNb = NNb, Wj = Wj, cutx = cutx, cuty = cuty, V = V, mu = MU, Bx = 1, alpha = alpha, delta = delta, phi = phi, qx = 0, periodicX = True)

HG = H_G1 - H_G0
HG0_DB = np.dot(vecs_0_hc, H_G0.dot(vecs_0))
HG_DB = np.dot(vecs_0_hc, HG.dot(vecs_0))

eig_arr = np.zeros((Bx.shape[0], k))
eig_arr_NB = np.zeros((Bx.shape[0], k))
for i in range(Bx.shape[0]):
    print(Bx.shape[0] - i, Bx[i])
    H_DB = HG0_DB + Bx[i]*HG_DB
    H = H_G0 + Bx[i]*HG

    eigs_DB, U_DB = LA.eigh(H_DB)
    eigs, vecs = spLA.eigsh(H, k=k, sigma=0, which='LM')
    idx_sort = np.argsort(eigs)

    eig_arr_NB[i, :] = eigs_DB
    eig_arr[i, :] = eigs[idx_sort]


for i in range(k):
    #if i % 2 == 0:
    plt.plot(Bx, eig_arr[:, i], c = 'r')
    #else:
    plt.plot(Bx, eig_arr_NB[:, i], c = 'b', ls = '--')

plt.xlabel(r'$B_x$ (T)')
plt.ylabel("Energy (meV)")
plt.savefig("EvsBx.png")
plt.show()
