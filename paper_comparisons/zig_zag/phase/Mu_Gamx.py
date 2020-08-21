import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sparse
import scipy.linalg as LA
import scipy.sparse.linalg as spLA

import majoranaJJ.operators.sparse.qmsops as spop #sparse operators
import majoranaJJ.lattice.nbrs as nb #neighbor arrays
import majoranaJJ.lattice.shapes as shps #lattice shapes
import majoranaJJ.modules.plots as plots #plotting functions
from majoranaJJ.modules.gamfinder import gamfinder as gf
from majoranaJJ.modules.gamfinder import gamfinder_lowE as gfLE

from majoranaJJ.operators.potentials.barrier_leads import V_BL
##############################################################
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
##############################################################

#Defining System
Nx = 3 #Number of lattice sites along x-direction
Ny = 360 #Number of lattice sites along y-direction
ax = 50 #lattice spacing in x-direction: [A]
ay = 50 #lattice spacing in y-direction: [A]
Wj = 40 #Junction region
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
NN = nb.NN_sqr(coor)
NNb = nb.Bound_Arr(coor)
lat_size = coor.shape[0]
print("Lattice Size: ", lat_size)

Lx = (max(coor[:, 0]) - min(coor[:, 0]) + 1)*ax #Unit cell size in x-direction
Ly = (max(coor[:, 1]) - min(coor[:, 1]) + 1)*ay #Unit cell size in y-direction

##############################################################

steps = 50

alpha = 200 #Spin-Orbit Coupling constant: [meV*A]
gz_i = 0.0 #parallel to junction: [meV], actually zero but avoiding degeneracy
gx = 0.0
phi = np.pi #SC phase differences, only want pi in this case
delta = 1.0 #Superconducting Gap: [meV]
V0 = 0 #Amplitude of potential : [meV]
mu = np.linspace(0, 7, steps) #Chemical Potential: [meV]
V = V_BL(coor, Wj = Wj, V0 = V0) #potential in normal region

##############################################################
#low energy projection Hamiltonian method
num_bound = 6
gamx_pi = np.zeros((steps, num_bound))

for i in range(steps):
    print(steps-i)

    gxpi = gfLE(coor, ax, ay, NN, NNb = NNb, Wj = Wj,
    V = V, mu = mu[i], gi = -0.001, gf = 1.5, alpha = alpha, delta = delta, phi = np.pi)
    print(gxpi)

    for j in range(num_bound):
        if j >= gxpi.size:
            gamx_pi[i, j] = None
        else:
            gamx_pi[i, j] = gxpi[j]

gamx_pi = np.array(gamx_pi)
print(gamx_pi)
print(gamx_pi.shape, mu.shape)

for i in range(num_bound):
    plt.plot(gamx_pi[:, i], mu, c='r', ls='solid', label = r'$\phi=\pi$')

plt.title("Zig-Zag Paper: Straight Junction")
plt.xlabel(r'$E_z$ (meV)')
plt.ylabel(r'$\mu$ (meV)')

plt.savefig('mu_gx_LE.png')
plt.show()

##############################################################
#original linear extrapolation method
"""
gamx_pi = []
mu_arr = []

for i in range(steps):
    print(steps-i)

    gammaxpi = gf(
        coor, ax, ay, NN, mu[i], NNb = NNb, Wj = Wj, alpha = alpha, delta = delta, phi = phi, gammaz = gz_i, V= V, k=12
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
"""
