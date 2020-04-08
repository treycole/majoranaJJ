import numpy as np

import majoranaJJ.operators.sparsOP as spop #sparse operators
import majoranaJJ.lattice.neighbors as nb #neighbor arrays
import majoranaJJ.lattice.shapes as shps #lattice shapes
import majoranaJJ.etc.plots as plots #plotting functions

Nx = 15 #Number of lattice sites allong x-direction
Ny = 15 #Number of lattice sites along y-direction
ax = 2 #lattice spacing in x-direction: [A]
ay = 2 #lattice spacing in y-direction: [A]

coor = shps.square(Nx, Ny) #square lattice
NN = nb.NN_Arr(coor) #neighbor array
NNb = nb.Bound_Arr(coor) #boundary array

Wsc = Ny - 5 #Superconducting region
Wj = 5  #Junction region
Lx = (max(coor[:, 0]) - min(coor[:, 0]) + 1)*ax #Unit cell size in x-direction
Ly = (max(coor[:, 1]) - min(coor[:, 1]) + 1)*ay #Unit cell size in y-direction

#Hamiltonian for size test
H = spop.HBDG(coor, ax, ay, NN, Wsc, Wj)
print("H shape: ", H.shape)

#Hamiltonian Parameters
alpha = 0*30e-3   #Spin-Orbit Coupling constant: [eV*A]
gammax = 0  #Zeeman field in x-direction
gammay = 0 #Zeeman field in y-direction
gammaz = 0.00001  #Zeeman field in z-direction
delta = 3e-2 #Superconducting Gap: [eV]
V0 = 0 #Amplitude of potential : [eV]
mu = 0.0 #Chemical Potential: [eV]

num = 24 # This is the number of eigenvalues and eigenvectors you want
steps = 30 #Number of kx and ky values that are evaluated
qx = np.linspace(-np.pi/Lx, np.pi/Lx, steps) #kx in the first Brillouin zone
qy = np.linspace(-np.pi/Ly, np.pi/Ly, steps) #ky in the first Brillouin zone
eig_arr = np.zeros((steps, num))

for i in range(steps):
    Energy = spop.EBDG(coor, ax, ay, NN, Wsc, Wj, NNb=NNb, mu=mu, V=V0, alpha=alpha, delta=delta, gammax=gammax, gammay=gammay, gammaz=gammaz, qx=qx[i], qy=0, periodicX=True, periodicY=True, num=num)

    eig_arr[i,:] = np.sort(Energy)


plots.bands(eig_arr, qx, Lx, Ly, title="Superconducting Gap Present: Periodic in X and Y")
