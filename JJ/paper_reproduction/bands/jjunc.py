import numpy as np

import majoranaJJ.operators.sparsOP as spop #sparse operators
import majoranaJJ.lattice.neighbors as nb #neighbor arrays
import majoranaJJ.lattice.shapes as shps #lattice shapes
import majoranaJJ.etc.plots as plots #plotting functions

Nx = 4 #Number of lattice sites allong x-direction
Ny = 80 #Number of lattice sites along y-direction
ax = 50 #lattice spacing in x-direction: [A]
ay = 50 #lattice spacing in y-direction: [A]
Wj = 16  #Junction region
print("Junction Width = ", Wj*ay*.10, "(nm)")
print("Supercondicting Lead Width = ", ((Ny - Wj)*ay*.10)/2, "(nm)")

coor = shps.square(Nx, Ny) #square lattice
NN = nb.NN_Arr(coor) #neighbor array
NNb = nb.Bound_Arr(coor) #boundary array
lat_size = coor.shape[0]
print("Lattice Size: ", lat_size)

Lx = (max(coor[:, 0]) - min(coor[:, 0]) + 1)*ax #Unit cell size in x-direction
Ly = (max(coor[:, 1]) - min(coor[:, 1]) + 1)*ay #Unit cell size in y-direction

#Hamiltonian Parameters
alpha = 100 #Spin-Orbit Coupling constant: [meV*A]
gammaz = 0 #Zeeman energy normal to system: [meV]
gammax = 0 #Zeeman energy in plane of junction: [meV]
phi = np.pi #SC phase difference
delta = 0.15 #Superconducting Gap: [meV]
V0 = 0.0 #Amplitude of potential : [meV]
mu = 0.327 #Chemical Potential: [meV]

num = 24 # This is the number of eigenvalues and eigenvectors you want
steps = 201 #Number of kx and ky values that are evaluated
qx = np.linspace(-np.pi/Lx, np.pi/Lx, steps) #kx in the first Brillouin zone
qy = np.linspace(-np.pi/Ly, np.pi/Ly, steps) #ky in the first Brillouin zone

bands = np.zeros((steps, num))
for i in range(steps):
    #print(steps - i)
    energy = spop.EBDG(coor, ax, ay, NN, NNb=NNb, Wj=Wj, alpha=alpha, delta=delta, phi = phi, gammax=gammax, gammaz=gammaz, mu=mu, qx=qx[i], periodicX=True, periodicY=False, k=num)

    bands[i, :] = energy
plots.bands(bands, qx, units = "[meV]", ylim = [-5, 5])
