'''
Goal here is to see an energy band splitting the size of the superconducting
gap
'''
import numpy as np
import majoranaJJ.operators.sparsOP as spop #sparse operators
import majoranaJJ.lattice.neighbors as nb #neighbor arrays
import majoranaJJ.lattice.shapes as shps #lattice shapes
import majoranaJJ.etc.plots as plots #plotting functions

Nx = 10 #Number of lattice sites allong x-direction
Ny = 10 #Number of lattice sites along y-direction
ax = 50 #lattice spacing in x-direction: [A]
ay = 50 #lattice spacing in y-direction: [A]

coor = shps.square(Nx, Ny) #square lattice
NN = nb.NN_Arr(coor) #neighbor array
NNb = nb.Bound_Arr(coor) #boundary array
lat_size = coor.shape[0]
print("Unit Cell = {} lattice sites".format(lat_size))

Lx = (max(coor[:, 0]) - min(coor[:, 0]) + 1)*ax #Unit cell size in x-direction
Ly = (max(coor[:, 1]) - min(coor[:, 1]) + 1)*ay #Unit cell size in y-direction

mu = 0 #[meV]
gamma = [0, 0, 5e-3] #[meV]
alpha = 0 #[meV * A]
delta = 0.15 #Superconducting Gap: [meV]
print("Delta = {} (meV)".format(0.15))

neigs = 24 # This is the number of eigenvalues and eigenvectors you want
k_steps = 201 #Number of kx and ky values that are evaluated
kx = np.linspace(-np.pi/Lx, np.pi/Lx, k_steps) #kx in the first Brillouin zone

bands = np.zeros((k_steps, neigs))
for i in range(k_steps):
    #print(steps - i)
    energy = spop.EBDG(coor, ax, ay, NN, NNb=NNb, mu=mu, gammax=gamma[0], gammay=gamma[1], gammaz=gamma[2], alpha = alpha, delta=delta,  periodicX=True, periodicY=True, qx=kx[i], qy=0, k=neigs)

    bands[i, :] = energy

plots.bands(kx, bands, title = "delta = {} (nm)".format(delta), savenm = 'gap_tst.png', ylim = [0.5, -0.5])
