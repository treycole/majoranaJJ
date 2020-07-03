import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sparse
import scipy.linalg as LA
import scipy.sparse.linalg as spLA
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm

import majoranaJJ.operators.sparse.qmsops as spop #sparse operators
import majoranaJJ.lattice.nbrs as nb #neighbor arrays
import majoranaJJ.lattice.shapes as shps #lattice shapes
import majoranaJJ.modules.plots as plots #plotting functions

from majoranaJJ.operators.potentials.barrier_leads import V_BL
from majoranaJJ.modules.constants import xi

###################################################

#Defining System
Nx = 24 #Number of lattice sites along x-direction
Ny = 600 #Number of lattice sites along y-direction
ax = 12.5 #lattice spacing in x-direction: [A]
ay = 50 #lattice spacing in y-direction: [A]
Wj = 4 #Junction region
cutx = 12 #width of nodule
cuty = 1 #height of nodule

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
N = coor.shape[0]
print("Lattice Size: ", N)

Lx = (max(coor[:, 0]) - min(coor[:, 0]) + 1)*ax #Unit cell size in x-direction
Ly = (max(coor[:, 1]) - min(coor[:, 1]) + 1)*ay #Unit cell size in y-direction
print(Lx)
###################################################

#Hamiltonian Parameters
alpha = 100 #Spin-Orbit Coupling constant: [meV*A]
gx = 0 #parallel to junction: [meV]
gz = 0 #normal to plane of junction: [meV]
phi = 0*np.pi #SC phase difference
delta = 1.0 #Superconducting Gap: [meV]
V0 = 70 #Amplitude of potential: [meV]
V = V_BL(coor, Wj = Wj, cutx=cutx, cuty=cuty, V0 = V0)
mu = 0  #Chemical Potential: [meV]

###################################################
num_eigs = 250 #number of eigenvalues and eigenvectors
steps = 51 #Number of kx values that are evaluated

cut_sizes = np.array([2, 4, 6, 8, 10])
first_crossing = np.zeros((cut_sizes.shape[0]))

V = V_BL(coor, Wj = Wj, cutx = cutx, cuty = cuty, V0 = V0)
H = spop.H0(coor, ax, ay, NN, NNb=NNb, alpha=alpha, V=V, gammax=gx, gammaz=gz, mu=mu, qx=0, periodicX=True)
eigs, vecs = spLA.eigsh(H, k=num_eigs, sigma=0, which='LM')
idx_sort = np.argsort(eigs)
eigs = eigs[idx_sort]
vecs = vecs[:, idx_sort]

E_crossing = eigs[0] + (2*xi*np.pi**2)/(Lx**2)
for i in range(eigs.shape[0]):
    if abs(eigs[i] - E_crossing) < 0.01:
        idx_first_crossing = i

print(E_crossing, "meV")
print(eigs)

qx = np.linspace(0, np.pi/Lx, steps) #kx in the first Brillouin zone
wt = np.zeros(num_eigs)

num_div = int(vecs.shape[0]/N)

for j in range(vecs.shape[1]):
        probdens = np.square(abs(vecs[:, j]))
        PD_UP = probdens[0: int(probdens.shape[0]/2)]
        PD_DOWN = probdens[int(probdens.shape[0]/2):]
        for k in range(N):
            if V[k, k] != 0:
                wt[j] += PD_UP[k] + PD_DOWN[k]

    print(wt)
    first_crossing[i] = wt[idx_first_crossing]
