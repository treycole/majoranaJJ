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
###################################################

#Defining System
Nx = 12 #Number of lattice sites along x-direction
Ny = 20#408 #Number of lattice sites along y-direction
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
###################################################

coor = shps.square(Nx, Ny) #square lattice
NN = nb.NN_Arr(coor) #neighbor array
NNb = nb.Bound_Arr(coor) #boundary array
lat_size = coor.shape[0]
print("Lattice Size: ", lat_size)

Lx = (max(coor[:, 0]) - min(coor[:, 0]) + 1)*ax #Unit cell size in x-direction
Ly = (max(coor[:, 1]) - min(coor[:, 1]) + 1)*ay #Unit cell size in y-direction

###################################################

#Hamiltonian Parameters
alpha = 100 #Spin-Orbit Coupling constant: [meV*A]
gx = 0 #parallel to junction: [meV]
gz = 0 #normal to plane of junction: [meV]
V0 = 50 #Amplitude of potential: [meV]
V = V_BL(coor, Wj = Wj, cutx=cutx, cuty=cuty, V0 = V0)
mu = 70 #Chemical Potential: [meV]

###################################################

k = 100 #This is the number of eigenvalues and eigenvectors you want
steps = 101 #Number of kx and ky values that are evaluated

qx = np.linspace(0, np.pi/Lx, steps) #kx in the first Brillouin zone
bands = np.zeros((steps, k))
for i in range(steps):
    print(steps - i)
    energy = spop.ESOC(coor, ax, ay, NN, NNb = NNb, V = V, mu = mu, alpha = alpha, qx = qx[i], periodicX = True, k = k, sigma = 0)

    bands[i, :] = energy

title = r"$L_x = %d nm$ , $L_y = %d nm$ , $W_{SC} = %.1f nm$, $W_j = %.1f$ , $nodule_x = %.1f$, $nodule_y = %.1f$, $\alpha = %d$, $\mu = %d$" % (Lx*.1, Ly*.1, SC_width, Junc_width, Nod_widthx, Nod_widthy, alpha, mu)
plots.bands(qx, bands, units = "meV", savenm = 'nodx={}nm_nody={}nm_Wj={}nm_Wsc={}nm_mu={}_V0={}.png'.format(Nod_widthx, Nod_widthy, Junc_width, SC_width, mu, V0), title = title)
