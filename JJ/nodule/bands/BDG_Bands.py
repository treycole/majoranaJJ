import numpy as np

import majoranaJJ.operators.sparse.qmsops as spop #sparse operators
import majoranaJJ.lattice.nbrs as nb #neighbor arrays
import majoranaJJ.lattice.shapes as shps #lattice shapes
import majoranaJJ.modules.plots as plots #plotting functions

#Defining System
Nx = 12 #Number of lattice sites along x-direction
Ny = 80 #Number of lattice sites along y-direction
ax = 50 #lattice spacing in x-direction: [A]
ay = 50 #lattice spacing in y-direction: [A]
Wj = 16 #Junction region
cutx = 0 #(Nx - 2*Sx) #width of nodule
cuty = 0 #0 #height of nodule

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
gz = 0.1
phi = np.pi #SC phase difference
delta = 0 #Superconducting Gap: [meV]
V0 = 0 #Amplitude of potential : [meV]
#V = V_BL(coor, Wj = Wj, cutx=cutx, cuty=cuty, V0 = V0)
mu = 0 #Chemical Potential: [meV], 20

num = 80 # This is the number of eigenvalues and eigenvectors you want
steps = 81 #Number of kx and ky values that are evaluated
qx = np.linspace(-np.pi/Lx, np.pi/Lx, steps) #kx in the first Brillouin zone
qy = np.linspace(-np.pi/Ly, np.pi/Ly, steps) #ky in the first Brillouin zone

bands = np.zeros((steps, num))
for i in range(steps):
    print(steps - i)
    energy = spop.EBDG(coor, ax, ay, NN, NNb=NNb, Wj=Wj, alpha=alpha, delta=delta, phi = phi, gammax=gx, gammaz=gz, mu=mu, qx=qx[i], periodicX=True, periodicY=False, k=num)

    bands[i, :] = energy
plots.bands(qx, bands, units = "meV", savenm = 'bands_ex.png')