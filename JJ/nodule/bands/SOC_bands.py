import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sparse
import scipy.linalg as LA
import scipy.sparse.linalg as spLA

import majoranaJJ.operators.sparse.qmsops as spop #sparse operators
import majoranaJJ.lattice.nbrs as nb #neighbor arrays
import majoranaJJ.lattice.shapes as shps #lattice shapes
import majoranaJJ.modules.plots as plots #plotting functions

from majoranaJJ.operators.potentials.barrier_leads import V_BL

#params similar to Fornieri
#Defining System
Nx = 24 #Number of lattice sites along x-direction
Ny = 80 #Number of lattice sites along y-direction
ax = 25#50 #lattice spacing in x-direction: [A]
ay = 25#50 #lattice spacing in y-direction: [A]
Wj = 8 #16 #Junction region
cutx = 2 #(Nx - 2*Sx) #width of nodule
cuty = 2 #0 #height of nodule

Junc_width = Wj*ay*.10 #nm
SC_width = ((Ny - Wj)*ay*.10)/2 #nm
Nod_widthx = cutx*ax*.1 #nm
Nod_widthy = cuty*ay*.1 #nm
print("Nodule Width in x-direction = ", Nod_widthx, "(nm)")
print("Nodule Width in y-direction = ", Nod_widthy, "(nm)")
print("Junction Width = ", Junc_width, "(nm)")
print("Supercondicting Lead Width = ", SC_width, "(nm)")

coor = shps.square(Nx, Ny) #square lattice
NN = nb.NN_Arr(coor) #neighbor array
NNb = nb.Bound_Arr(coor) #boundary array
lat_size = coor.shape[0]
print("Lattice Size: ", lat_size)

Lx = (max(coor[:, 0]) - min(coor[:, 0]) + 1)*ax #Unit cell size in x-direction
Ly = (max(coor[:, 1]) - min(coor[:, 1]) + 1)*ay #Unit cell size in y-direction

#Hamiltonian Parameters
alpha = 100 #Spin-Orbit Coupling constant: [meV*A]
gx = 0 #parallel to junction: [meV]
gz = 0 #normal to plane of junction: [meV]
phi = np.pi #SC phase difference
delta = 0 #Superconducting Gap: [meV]
V0 = 60 #Amplitude of potential : [meV]
V = V_BL(coor, Wj = Wj, cutx=cutx, cuty=cuty, V0 = V0)
mu = 0 #Chemical Potential: [meV], 20

k = 24 #This is the number of eigenvalues and eigenvectors you want
steps = 101 #Number of kx and ky values that are evaluated
qx = np.linspace(-np.pi/Lx, np.pi/Lx, steps) #kx in the first Brillouin zone
qy = np.linspace(-np.pi/Ly, np.pi/Ly, steps) #ky in the first Brillouin zone

bands = np.zeros((steps, k))
for i in range(steps):
    print(steps - i)
    energy = spop.ESOC(coor, ax, ay, NN, NNb = NNb, V = V, mu = mu, gammax = gx, alpha = alpha, qx = qx[i], periodicX = True, k = k)

    bands[i, :] = energy

plots.bands(qx, bands, units = "meV", savenm = 'bands_ex.png')
