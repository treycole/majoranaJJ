import numpy as np
import matplotlib.pyplot as plt

import majoranaJJ.operators.sparse.qmsops as spop #sparse operators
import majoranaJJ.lattice.nbrs as nb #neighbor arrays
import majoranaJJ.lattice.shapes as shps #lattice shapes
import majoranaJJ.modules.plots as plots #plotting functions
from majoranaJJ.modules.gamfinder import gamfinder

Nx = 4 #Number of lattice sites along x-direction
Ny = 80 #Number of lattice sites along y-direction
ax = 50 #lattice spacing in x-direction: [A]
ay = 50 #lattice spacing in y-direction: [A]
Wj = 16  #Junction region
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

alpha = 100 #Spin-Orbit Coupling constant: [meV*A]
gammax_i = 0.01 #parallel to junction: [meV]
gammaz = 0.0
phi = [0, np.pi] #SC phase difference
delta = 0.15 #Superconducting Gap: [meV]
V0 = 0.0 #Amplitude of potential : [meV]
mu = np.linspace(75, 81, steps) #Chemical Potential: [meV]

gamx_pi = []
mu_arr = []
gamx_0 = []

for i in range(steps):
    print(steps-i)
    gammax0 = gamfinder(
        coor, ax, ay, NN, mu[i], NNb = NNb, Wj = Wj, alpha = alpha, delta = delta, phi = 0, gammax = gammax_i, periodicX = True
        )
    gammaxpi = gamfinder(
        coor, ax, ay, NN, mu[i], NNb = NNb, Wj = Wj, alpha = alpha, delta = delta, phi = np.pi, gammax = gammax_i, periodicX = True
        )

    mu_arr.append(mu[i])
    gamx_0.append(gammax0)
    gamx_pi.append(gammaxpi)

mu_arr, gamx_0, gamx_pi = np.array(mu_arr), np.array(gamx_0), np.array(gamx_pi)

plt.plot(gamx_0, mu_arr, c='k', ls='solid', label = r'$\phi = 0$')
plt.plot(gamx_pi, mu_arr, c='r', ls='solid', label = r'$\phi = \pi$')

plt.xlabel(r'$E_z$ (meV)')
plt.ylabel(r'$\mu$ (meV)')

plt.xlim(0, 0.35)
plt.ylim(78, 80.1)

plt.legend()
plt.savefig('mu_gx.png')
plt.show()
