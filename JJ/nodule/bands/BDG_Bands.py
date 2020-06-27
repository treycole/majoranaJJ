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

from majoranaJJ.operators.potentials.barrier_leads import V_BL
###################################################

#Defining System
Nx = 12 #Number of lattice sites along x-direction
Ny = 400 #Number of lattice sites along y-direction
ax = 50 #lattice spacing in x-direction: [A]
ay = 50 #lattice spacing in y-direction: [A]
Wj = 8 #Junction region
cutx = 0 #width of nodule
cuty = 3 #height of nodule

nod_bool = True
if cutx == 0 and cuty == 0:
    nod_bool = False

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
phi = 0*np.pi #SC phase difference
delta = 1.0 #Superconducting Gap: [meV]
V0 = 50 #Amplitude of potential: [meV]
V = V_BL(coor, Wj = Wj, cutx=cutx, cuty=cuty, V0 = V0)
mu = 56.6  #Chemical Potential: [meV]

#####################################

k = 12 #This is the number of eigenvalues and eigenvectors you want
steps = 101 #Number of kx values that are evaluated
qx = np.linspace(0, np.pi/Lx, steps) #kx in the first Brillouin zone

bands = np.zeros((steps, k))
for i in range(steps):
    print(steps - i)
    energy = spop.EBDG(coor, ax, ay, NN, NNb=NNb, Wj=Wj, cutx=cutx, cuty=cuty, alpha=alpha, delta=delta, phi = phi, V=V, gammax=gx, gammaz=gz, mu=mu, qx=qx[i], periodicX=True, k=k)

    bands[i, :] = energy

if nod_bool:
    x = "Present"
else:
    x = "Not Present"

plots.bands(qx, bands, title = r'Nodule {}: $\mu = {}$'.format(x, mu), units = "meV", savenm = 'bands_bdg.png'.format(mu, V))
sys.exit()

#####################################
"""
#VV = sparse.bmat([[None, V], [-V, None]], format='csc', dtype='complex')
#plots.junction(coor, VV, title = 'Potential Profile', savenm = 'potential_profile.jpg')

k = 26
H = spop.HBDG(coor, ax, ay, NN, NNb=NNb, Wj=Wj, alpha=alpha, delta=delta, phi = phi, V=V, gammax=gx, gammaz=gz, mu=mu, qx=0.00287, periodicX=True, periodicY=False)

eigs, vecs = spLA.eigsh(H, k=k, sigma=0, which='LM')
idx_sort = np.argsort(eigs)
eigs = eigs[idx_sort]
vecs = vecs[:, idx_sort]
print(eigs)

n = int(k/2)
plots.state_cmap(coor, eigs, vecs, n = int(k/2), title = r'$|\psi|^2$', savenm = 'State_k={}.png'.format(n))

sys.exit()

for i in range(int(k/2), k):
    plots.state_cmap(coor, eigs, vecs, n = i, title = r'$|\psi|^2$', savenm = 'State_k={}.png'.format(i))
"""
