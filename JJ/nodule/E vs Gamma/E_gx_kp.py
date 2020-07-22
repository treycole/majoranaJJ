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
import majoranaJJ.operators.sparse.k_dot_p as kp

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
phi = np.pi #SC phase difference
delta = 1.0 #Superconducting Gap: [meV]
V0 = 50 #Amplitude of potential : [meV]
V = V_BL(coor, Wj = Wj, cutx=cutx, cuty=cuty, V0 = V0)
mu = 88 #Chemical Potential: [meV]

###################################################

#Energy plot vs Zeeman energy in x-direction

#seeing the junction layout
#D_test = spop.Delta(coor, Wj = Wj, delta = 1, cutx = cutx, cuty = cuty)
#plots.junction(coor, D_test)

k = 100
H0, Hq, Hqq, DELTA, Hgam = kp.Hq(coor, ax, ay, NN, NNb = NNb, Wj = Wj, cutx = cutx, cuty = cuty, V = V, mu = mu, alpha = alpha, delta = delta, phi = phi, periodicX = True)

H = kp.H0(H0, Hq, Hqq, Hgam, q = 1e-6, gx = gx[0])
eigs_0, vecs_0 = spLA.eigsh(H, k=k, sigma=0, which='LM')
vecs_0_hc = np.conjugate(np.transpose(vecs_0)) #hermitian conjugate
vecs_0_c = np.conjugate(vecs_0)

H0_DB = np.dot(vecs_0_hc, H0.dot(vecs_0))
Hq_DB = np.dot(vecs_0_hc, Hq.dot(vecs_0))
Hqq_DB = np.dot(vecs_0_hc, Hqq.dot(vecs_0))
DELTA_DB = np.dot(vecs_0_hc, DELTA.dot(vecs_0_c))
Hgam_DB = np.dot(vecs_0_hc, Hgam.dot(vecs_0))
MU = np.eye(H0_DB.shape[0])

eig_arr_DB = np.zeros((gx.shape[0], 2*k))
eig_arr = np.zeros((gx.shape[0], 44))
for i in range(gx.shape[0]):
    print(gx.shape[0] - i, gx[i])
    #energy = spop.EBDG(coor, ax, ay, NN, NNb=NNb, Wj=Wj, cutx=cutx, cuty=cuty, alpha=alpha, delta=delta, phi = phi, V=V, gammax=gx[i], mu=mu, qx=0, periodicX=True, k=44)

    H = kp.HBDG_LE(H0_DB, Hq_DB, Hqq_DB, DELTA_DB, Hgam_DB, MU, 0, q = 0, gx = gx[i])

    eigs_DB, U_DB = LA.eigh(H)
    idx_sort = np.argsort(eigs_DB)
    eigs_DB = eigs_DB[idx_sort]

    eig_arr_DB[i, :] = eigs_DB
    #eig_arr[i, :] = energy

for i in range(44):
    #if i % 2 == 0:
    #plt.plot(gx, eig_arr[:, i], c = 'r')
    plt.plot(gx, eig_arr_DB[:, k+22- i], c = 'b', ls = '--')
    #else:

plt.xlabel(r'$E_z$ (meV)')
plt.ylabel("Energy (meV)")
plt.savefig("EvsGamx.png")
plt.show()
