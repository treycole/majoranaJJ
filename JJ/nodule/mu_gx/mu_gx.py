import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sparse
import scipy.sparse.linalg as spLA

import majoranaJJ.operators.sparse.qmsops as spop #sparse operators
import majoranaJJ.lattice.nbrs as nb #neighbor arrays
import majoranaJJ.lattice.shapes as shps #lattice shapes
import majoranaJJ.modules.plots as plots #plotting functions
from majoranaJJ.modules.gamfinder import gamfinder as gf
from majoranaJJ.operators.potentials.barrier_leads import V_BL

Nx = 20 #Number of lattice sites along x-direction
Ny = 350 #Number of lattice sites along y-direction
ax = 50 #lattice spacing in x-direction: [A]
ay = 50 #lattice spacing in y-direction: [A]
Wj = 4 #Junction region
Sx = 8 #3 #length of either side of nodule
cutx = (Nx - 2*Sx) #width of nodule
cuty = 1 #4 #height of nodule

Junc_width = Wj*ay*.10 #nm
SC_width = ((Ny - Wj)*ay*.10)/2 #nm
#Nod_widthx = cutx*ax*.1 #nm
#Nod_widthy = cuty*ay*.1 #nm
#print("Nodule Width in x-direction = ", Nod_widthx, "(nm)")
#print("Nodule Width in y-direction = ", Nod_widthy, "(nm)")
print("Junction Width = ", Junc_width, "(nm)")
print("Supercondicting Lead Width = ", SC_width, "(nm)")

coor = shps.square(Nx, Ny) #square lattice
NN = nb.NN_sqr(coor)
NNb = nb.Bound_Arr(coor)
lat_size = coor.shape[0]
print("Lattice Size: ", lat_size)

Lx = (max(coor[:, 0]) - min(coor[:, 0]) + 1)*ax #Unit cell size in x-direction
Ly = (max(coor[:, 1]) - min(coor[:, 1]) + 1)*ay #Unit cell size in y-direction

steps = 50

alpha = 200 #Spin-Orbit Coupling constant: [meV*A]
gammax_i = 0 #parallel to junction: [meV]
gammaz = 0.0
phi = np.pi #SC phase difference
delta = 1.0 #Superconducting Gap: [meV]
V0 = -5 #50 #Amplitude of potential : [meV]
V = V_BL(coor, Wj = Wj, Sx = Sx, cutx=cutx, cuty=cuty, V0 = V0)
mu = np.linspace(0, 40, steps) #Chemical Potential: [meV], 20

"""
#state plot
MU = 0
GX = 0.3

H = spop.HBDG(
    coor, ax, ay, NN, NNb = NNb, Wj = Wj,
    V = V, mu = MU,
    gammax = GX,
    alpha = alpha, delta = delta, phi = np.pi,
    qx = 0,
    periodicX = True, periodicY = False
    )

eigs, states = spLA.eigsh(H, k=8, sigma=0, which='LM')
idx_sort = np.argsort(eigs)
print(eigs[idx_sort])
plots.state_cmap(coor, eigs, states, n=4, savenm='prob_density_nodule_n=4.png')
plots.state_cmap(coor, eigs, states, n=5, savenm='prob_density_nodule_n=5.png')
plots.state_cmap(coor, eigs, states, n=6, savenm='prob_density_nodule_n=6.png')
"""

"""
#phase diagram mu vs gamx
gamx_0 = []
gamx_pi = []
mu_arr = []
for i in range(steps):
    print(steps-i)

    gammaxpi = gf(
        coor, ax, ay, NN, mu[i], NNb = NNb, Wj = Wj, Sx = Sx, cutx = cutx, cuty=cuty, alpha = alpha, delta = delta, phi = np.pi, gammax = gammax_i, V = V, periodicX = True, k=12
        )

    mu_arr.append(mu[i])
    gamx_pi.append(gammaxpi)

mu_arr, gamx_pi = np.array(mu_arr), np.array(gamx_pi)
print(gamx_pi.shape, mu_arr.shape)

plt.plot(gamx_pi, mu_arr, c='r', ls='solid')

plt.xlabel(r'$E_z$ (meV)')
plt.ylabel(r'$\mu$ (meV)')

plt.legend()
plt.savefig('juncwidth = {} SCwith = {} V0 = {} nodwidthx = {} nodwidthy = {} .png'.format(Junc_width, SC_width, V0, 0, 0 ))
plt.show()
"""
