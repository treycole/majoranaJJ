import numpy as np
import matplotlib.pyplot as plt

import majoranaJJ.operators.sparse.qmsops as spop #sparse operators
import majoranaJJ.lattice.nbrs as nb #neighbor arrays
import majoranaJJ.lattice.shapes as shps #lattice shapes
import majoranaJJ.modules.plots as plots #plotting functions

#from majoranaJJ.etc.mufinder import mufinder

Nx = 4 #Number of lattice sites along x-direction
Ny = 80 #Number of lattice sites along y-direction
ax = 50 #lattice spacing in x-direction: [A]
ay = 50 #lattice spacing in y-direction: [A]
Wj = 16  #Junction region
print("Junction Width = ", Wj*ay*.10, "(nm)")
print("Supercondicting Lead Width = ", ((Ny - Wj)*ay*.10)/2, "(nm)")
print("L1 parallel to Junction Length = ", (Nx*ax)*.1, "(nm)")

coor = shps.square(Nx, Ny) #square lattice
NN = nb.NN_Arr(coor)
NNb = nb.Bound_Arr(coor)
lat_size = coor.shape[0]
print("Lattice Size: ", lat_size)

Lx = (max(coor[:, 0]) - min(coor[:, 0]) + 1)*ax #Unit cell size in x-direction
Ly = (max(coor[:, 1]) - min(coor[:, 1]) + 1)*ay #Unit cell size in y-direction

gz_steps = 100 #Number of gamma-Z values that are evaluated

alpha = 0#100 #Spin-Orbit Coupling constant: [meV*A]
phi = 0 #SC phase difference
gammax = np.linspace(0, 0.5, gz_steps) #Zeeman field energy contribution:[eV T]
delta = 0.15 #Superconducting Gap: [meV]
V0 = 0.0 #Amplitude of potential : [eV]
mu = 0.0 #79.1 #Chemical Potential: [meV]

neigs = 2 # This is the number of eigenvalues and eigenvectors you want
eig_arr = np.zeros((gz_steps, neigs))
for i in range(gz_steps):
    print(gz_steps - i)
    energy = spop.EBDG(coor, ax, ay, NN, Wj=Wj, NNb=NNb, mu=mu, alpha=alpha, delta=delta, phi=0, gammax=gammax[i], periodicX=True, periodicY =True, k=neigs)

    eig_arr[i, :] = energy

#k_steps = 501
#neigs = 24
#kx = np.linspace(-np.pi/Lx, np.pi/Lx, k_steps) #kx in the first Brillouin zone
#bands = np.zeros((k_steps, neigs))
#for i in range(k_steps):
#    print(k_steps - i)
#    energy = spop.ESOC(coor, ax, ay, NN, NNb=NNb, alpha=alpha, mu=0, qx=kx[i], periodicX=True, k=neigs)

#    bands[i, :] = energy

#plots.bands(kx, bands)
plots.phase(gammax, eig_arr, xlabel = r'$E_z$ (meV)', ylabel = 'Energy (meV)', title = r"Energy vs Gamma $\hat{x}$ at $\Delta = 0.15 (meV)$", savenm = 'crit_gx.png')
