import numpy as np
import matplotlib.pyplot as plt

import majoranaJJ.operators.sparse.qmsops as spop #sparse operators
import majoranaJJ.lattice.nbrs as nb #neighbor arrays
import majoranaJJ.lattice.shapes as shps #lattice shapes
import majoranaJJ.modules.plots as plots #plotting functions
from majoranaJJ.modules.gamfinder import gamfinder as gf
from majoranaJJ.operators.potentials.barrier_leads import V_BL

Nx = 4 #Number of lattice sites along x-direction
Ny = 80 #Number of lattice sites along y-direction
ax = 100 #lattice spacing in x-direction: [A]
ay = 100 #lattice spacing in y-direction: [A]
Wj = 4 #Junction region
print("Junction Width = ", Wj*ay*.10, "(nm)")
print("Supercondicting Lead Width = ", ((Ny - Wj)*ay*.10)/2, "(nm)")

coor = shps.square(Nx, Ny) #square lattice
NN = nb.NN_sqr(coor)
NNb = nb.Bound_Arr(coor)
lat_size = coor.shape[0]
print("Lattice Size: ", lat_size)

Lx = (max(coor[:, 0]) - min(coor[:, 0]) + 1)*ax #Unit cell size in x-direction
Ly = (max(coor[:, 1]) - min(coor[:, 1]) + 1)*ay #Unit cell size in y-direction

steps = 200

V = V_BL(coor, Wj = Wj, V0 = 10)
alpha = 200 #Spin-Orbit Coupling constant: [meV*A]
gammax_i = 0.0001 #parallel to junction: [meV]
gammaz = 0.0
phi = [0, np.pi] #SC phase difference
delta = 1.0 #Superconducting Gap: [meV]
V0 = 0.0 #Amplitude of potential : [meV]
mu = np.linspace(0, 20, steps) #Chemical Potential: [meV]

gamx_pi = []
mu_arr = []

gamx_0 = []
gx = np.linspace(gammax_i, 6, 50)
eig_arr = np.zeros((gx.shape[0], 20))

for i in range(steps):
    print(steps-i)

    gammaxpi = gf(
        coor, ax, ay, NN, mu[i], NNb = NNb, Wj = Wj, alpha = alpha, delta = delta, phi = np.pi, gammax = gammax_i, V = V, periodicX = True, k=10
        )

    mu_arr.append(mu[i])
    gamx_pi.append(gammaxpi)

mu_arr, gamx_pi = np.array(mu_arr), np.array(gamx_pi)
print(gamx_pi.shape, mu_arr.shape)

plt.plot(gamx_pi, mu_arr, c='r', ls='solid')

plt.xlabel(r'$E_z$ (meV)')
plt.ylabel(r'$\mu$ (meV)')

plt.legend()
plt.savefig('mu_gx.png')
plt.show()
