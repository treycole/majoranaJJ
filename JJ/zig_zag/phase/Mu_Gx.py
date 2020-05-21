import sys
import numpy as np
import matplotlib.pyplot as plt

import majoranaJJ.operators.sparse.qmsops as spop #sparse operators
import majoranaJJ.lattice.nbrs as nb #neighbor arrays
import majoranaJJ.lattice.shapes as shps #lattice shapes
import majoranaJJ.modules.plots as plots #plotting functions
from majoranaJJ.modules.gamfinder import gamfinder as gf
from majoranaJJ.operators.potentials.barrier_leads import V_BL

"""
This script is replicating a figure in a paper published under the name of:
Laeven T. et.al, "Enhanced proximity effect in zigzag-shaped Majorana Josephson junctions" (2019)

The figure being replicated is figure 4(c), where their system is a Josephson Junction with a straight interface between the superconducting leads and the normal region.

The parameters used are:
m_ef = 0.02*m_e ... effective mass of electron
W = 200 (nm) ... Width of the normal region
L_SC = 800 (nm) ... Width of each superconducting lead
phi = pi ... phase difference across the junction
a = 5 (nm) ... lattice spacing constant
Delta = 1 (meV) ... superconducting gap
alpha = 20 (meV*nm) ... spin-orbit coupling constant

"""
Nx = 3 #Number of lattice sites along x-direction, periodic in this direction
Ny = 360 #Number of lattice sites along y-direction, transverse normal region
ax = 50 #lattice spacing in x-direction: [A]
ay = 50 #lattice spacing in y-direction: [A]
Wj = 40 #Junction region, number of sites
print("Junction Width = ", Wj*ay*.10, "(nm)")
print("Supercondicting Lead Width = ", ((Ny - Wj)*ay*.10)/2, "(nm)")

coor = shps.square(Nx, Ny) #square lattice, stores x and y coordinate
NN = nb.NN_sqr(coor) #neighbor array, which sites are nearest neighbors
NNb = nb.Bound_Arr(coor) #boundary array, which sites are periodic neighbors
lat_size = coor.shape[0]
print("Lattice Size: ", lat_size)

Lx = (max(coor[:, 0]) - min(coor[:, 0]) + 1)*ax #Unit cell size in x-direction
Ly = (max(coor[:, 1]) - min(coor[:, 1]) + 1)*ay #Unit cell size in y-direction

steps = 150

alpha = 200 #Spin-Orbit Coupling constant: [meV*A]
gammax_i = 0.0 #parallel to junction: [meV], actually zero but avoiding degeneracy
gammaz = 0.0
phi = [0, np.pi] #SC phase differences, only want pi in this case
delta = 1.0 #Superconducting Gap: [meV]
V0 = 0 #Amplitude of potential : [meV]
mu = np.linspace(0, 20, steps) #Chemical Potential: [meV]
V = V_BL(coor, Wj = Wj, V0 = V0) #potential in normal region

gamx_pi = []
mu_arr = []

gx = np.linspace(gammax_i, 5, steps)
eig_arr = np.zeros((gx.shape[0], 20))
MU = 7.5 #meV
for i in range(steps):
    print(steps-i)

    gammaxpi = gf(
        coor, ax, ay, NN, mu[i], NNb = NNb, Wj = Wj, alpha = alpha, delta = delta, phi = phi[1], gammax = gammax_i, V= V, periodicX=True, k=12
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
