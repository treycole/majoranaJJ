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

##############################################################

#Defining System
Nx = 3 #Number of lattice sites along x-direction
Ny = 360 #Number of lattice sites along y-direction
ax = 50 #lattice spacing in x-direction: [A]
ay = 50 #lattice spacing in y-direction: [A]
Wj = 40 #Junction region
cutx = 0 #(Nx - 2*Sx) #width of nodule
cuty = 0 #0 #height of nodule

Junc_width = Wj*ay*.10 #nm
SC_width = ((Ny - Wj)*ay*.10)/2 #nm
Nod_widthx = cutx*ax*.1 #nm
Nod_widthy = cuty*ay*.1 #nm
print("Nodule Width in x-direction = ", Nod_widthx, "(nm)")
print("Nodule Width in y-direction = ", Nod_widthy, "(nm)")
print("Junction Width = ", Junc_width, "(nm)")
print("Supercondicting Lead Width = ", SC_width, "(nm)")

##############################################################
coor = shps.square(Nx, Ny) #square lattice
NN = nb.NN_sqr(coor)
NNb = nb.Bound_Arr(coor)
lat_size = coor.shape[0]
print("Lattice Size: ", lat_size)

Lx = (max(coor[:, 0]) - min(coor[:, 0]) + 1)*ax #Unit cell size in x-direction
Ly = (max(coor[:, 1]) - min(coor[:, 1]) + 1)*ay #Unit cell size in y-direction

##############################################################
#Hamiltonian parameters
steps = 150

alpha = 200 #Spin-Orbit Coupling constant: [meV*A]
gx = 0.0 #Zeeman energy, field parallel to junction: [meV]
phi = np.pi #SC phase differences, only want pi in this case
delta = 1.0 #Superconducting Gap: [meV]
V0 = 0 #Amplitude of potential : [meV]

V = V_BL(coor, Wj = Wj, V0 = V0) #potential in normal region
mu = np.linspace(0, 20, steps) #Chemical Potential: [meV]

##############################################################

#energy plot vs mu

#seeing the system
#D_test = spop.Delta(coor, Wj = Wj, delta = 1, cutx = cutx, cuty = cuty)
#plots.junction(coor, D_test)

Gx = 0
k = 24 #energy eigs

eig_arr = np.zeros((steps, k))

for i in range(steps):
    print(steps - i)
    H0 = spop.HBDG(coor, ax, ay, NN, NNb=NNb, Wj=Wj, cutx=cutx, cuty=cuty, V=V, gammax = Gx, mu = mu[i], alpha=alpha, delta=delta, phi=phi, qx=0, periodicX=True)

    eigs, vecs = spLA.eigsh(H0, k=k, sigma = 0,  which='LM')
    idx_sort = np.argsort(eigs)
    eigs = eigs[idx_sort]

    eig_arr[i, :] = eigs


plt.plot(mu, eig_arr, c = 'b')

plt.xlabel(r'$\mu$ (meV)')
plt.ylabel("Energy (meV)")
plt.savefig("EvsMu.png")
plt.show()
