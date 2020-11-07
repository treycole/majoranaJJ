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
import majoranaJJ.modules.golden_search as gs
import majoranaJJ.modules.checkers as check

import numpy as np
import matplotlib.pyplot as plt
###################################################
#Defining System
Nx = 3 #Number of lattice sites along x-direction
Ny = 50 #Number of lattice sites along y-direction
ax = 100 #lattice spacing in x-direction: [A]
ay = 100 #lattice spacing in y-direction: [A]
Wj = 5 #Junction region
cutx = 0 #width of nodule
cuty = 0 #height of nodule
Nx, Ny, cutx, cuty, Wj = check.junction_geometry_check(Nx, Ny, cutx, cuty, Wj)
print("Nx = {}, Ny = {}, cutx = {}, cuty = {}, Wj = {}".format(Nx, Ny, cutx, cuty, Wj))

Junc_width = Wj*ay*.10 #nm
SC_width = ((Ny - Wj)*ay*.10)/2 #nm
Nod_widthx = cutx*ax*.1 #nm
Nod_widthy = cuty*ay*.1 #nm
print("Nodule Width in x-direction = ", Nod_widthx, "(nm)")
print("Nodule Width in y-direction = ", Nod_widthy, "(nm)")
print("Junction Width = ", Junc_width, "(nm)")
print("Supercondicting Lead Width = ", SC_width, "(nm)")
###################################################coor = shps.square(Nx, Ny) #square lattice
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
gamx = 1 #parallel to junction: [meV]
phi = np.pi #SC phase difference
delta = 1.0 #Superconducting Gap: [meV]
mu = 10 #Chemical Potential: [meV]
#####################################
"""
k = 12 #This is the number of eigenvalues and eigenvectors you want
steps = 101 #Number of kx values that are evaluated
qx = np.linspace(0, np.pi/Lx, steps) #kx in the first Brillouin zone

bands = np.zeros((steps, k))
for i in range(steps):
    print(steps - i)
    H = spop.HBDG(coor, ax, ay, NN, NNb=NNb, Wj=Wj, mu=mu, alpha=alpha, delta=delta, phi=phi, gamx=gamx, qx=qx[i])
    eigs, vecs = spLA.eigsh(H, k=k, sigma=0, which='LM')
    idx_sort = np.argsort(eigs)
    eigs = eigs[idx_sort]
    bands[i, :] = eigs

for i in range(bands.shape[1]):
    plt.plot(qx, bands[:, i], c ='mediumblue', linestyle = 'solid')
    plt.plot(-qx, bands[:, i], c ='mediumblue', linestyle = 'solid')
    #plt.scatter(q, eigarr[:, i], c ='b')
plt.plot(qx, 0*qx, c = 'k', linestyle='solid', lw=1)
plt.plot(-qx, 0*qx, c = 'k', linestyle='solid', lw=1)
#plt.xticks(np.linspace(min(k), max(k), 3), ('-π/Lx', '0', 'π/Lx'))
plt.xlabel('kx (1/A)')
plt.ylabel('Energy (meV)')
plt.title('BDG Spectrum', wrap = True)
plt.savefig('juncwidth = {} SCwidth = {} nodwidthx = {} nodwidthy = {} Delta = {} Alpha = {} phi = {} mu = {}.png'.format(Junc_width, SC_width, Nod_widthx, Nod_widthy, delta, alpha, phi, mu))
plt.show()
"""
E_min, mu = gs.mu_scan_2(coor, ax, ay, NN, 0.0, 20.0, NNb=NNb, Wj=Wj, cutx=cutx, cuty=cuty, gamx=gamx, alpha=alpha, delta=delta, phi=phi, Vj=0)

plt.plot(mu, E_min)
plt.show()
