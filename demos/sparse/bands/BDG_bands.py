import numpy as np

import majoranaJJ.operators.sparsOP as spop #sparse operators
import majoranaJJ.lattice.neighbors as nb #neighbor arrays
import majoranaJJ.lattice.shapes as shps #lattice shapes
import majoranaJJ.etc.plots as plots #plotting functions

Nx = 20 #Number of lattice sites allong x-direction
Ny = 20 #Number of lattice sites along y-direction
ax = 50 #lattice spacing in x-direction: [A]
ay = 50 #lattice spacing in y-direction: [A]

coor = shps.square(Nx, Ny) #square lattice
NN = nb.NN_Arr(coor) #neighbor array
NNb = nb.Bound_Arr(coor) #boundary array

Wj = 5  #Junction region
Lx = (max(coor[:, 0]) - min(coor[:, 0]) + 1)*ax #Unit cell size in x-direction
Ly = (max(coor[:, 1]) - min(coor[:, 1]) + 1)*ay #Unit cell size in y-direction

#Hamiltonian for size test
H = spop.HBDG(coor, ax, ay, NN)
print("H shape: ", H.shape)

#Hamiltonian Parameters
alpha = 0   #Spin-Orbit Coupling constant: [eV*A]
gammax = 0  #Zeeman field in x-direction
gammay = 0 #Zeeman field in y-direction
gammaz = 0*1e-5 #Zeeman field in z-direction
delta = 3e-4 #Superconducting Gap: [eV]
V0 = 0 #Amplitude of potential : [eV]
mu = 0.0 #Chemical Potential: [eV]

num = 40 # This is the number of eigenvalues and eigenvectors you want
steps = 40 #Number of kx and ky values that are evaluated
qx = np.linspace(-np.pi/Lx, np.pi/Lx, steps) #kx in the first Brillouin zone
qy = np.linspace(-np.pi/Ly, np.pi/Ly, steps) #ky in the first Brillouin zone

bands = np.zeros((steps, num))
for i in range(steps):
    energy = spop.EBDG(coor, ax, ay, NN, NNb=NNb, Wj=Wj, alpha=alpha, delta=delta, gammaz=gammaz, mu=0, qx=qx[i], periodicX=True, periodicY=True, neigs=num)

    bands[i, :] = energy
plots.bands(bands, qx, units = "[meV]")
