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
from majoranaJJ.operators.potentials.barrier_leads import V_BL

Nx = 10 #Number of lattice sites along x-direction
Ny = 350 #Number of lattice sites along y-direction
ax = 50 #lattice spacing in x-direction: [A]
ay = 50 #lattice spacing in y-direction: [A]
Wj = 10 #Junction region
Sx = 3 #length of either side of nodule
cutx = (Nx - 2*Sx) #width of nodule
cuty = 4 #height of nodule
Junc_width = Wj*ay*.10 #nm
SC_width = ((Ny - Wj)*ay*.10)/2 #nm
nodwidthx = cutx*ax*.1 #nm
nodwidthy = cuty*ay*.1 #nm

coor = shps.square(Nx, Ny) #square lattice
NN = nb.NN_sqr(coor)
NNb = nb.Bound_Arr(coor)
lat_size = coor.shape[0]
print("Lattice Size: ", lat_size)

Lx = (max(coor[:, 0]) - min(coor[:, 0]) + 1)*ax #Unit cell size in x-direction
Ly = (max(coor[:, 1]) - min(coor[:, 1]) + 1)*ay #Unit cell size in y-direction

steps = 20

alpha = 200 #Spin-Orbit Coupling constant: [meV*A]
gammax_i = 0 #parallel to junction: [meV]
gammaz = 0.0
phi = np.pi #SC phase difference
delta = 1.0 #Superconducting Gap: [meV]
V0 = -10 #50 #Amplitude of potential : [meV]
V = V_BL(coor, Wj = Wj, Sx = Sx, cutx=cutx, cuty=cuty, V0 = V0)
mu = np.linspace(0, 40, steps) #Chemical Potential: [meV], 20

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
plt.savefig('juncwidth = {} SCwith = {} V0 = {} nodwidthx = {} nodwidthy = {} .png'.format(Junc_width, SC_width, V0, nodwidthx, nodwidthy ))
plt.show()

"""
#energy plot vs gamx
k = 20
k0 = 64

H0 = spop.HBDG(coor, ax, ay, NN, NNb = NNb, Wj = Wj, Sx = Sx, cutx = cutx, cuty = cuty, V = V, mu = mu, gammax = 0.0001, alpha = alpha, delta = delta, phi = np.pi, qx = 0, periodicX = True)

eigs_0, vecs_0 = spLA.eigsh(H0, k=k0, sigma=0, which='LM')
idx_sort = np.argsort(eigs_0)
print(eigs_0[idx_sort])

vecs_0_hc = np.conjugate(np.transpose(vecs_0))

gx = np.linspace(0.0001, 1, steps)
eig_arr = np.zeros((gx.shape[0], k))
eig_arr_NB = np.zeros((gx.shape[0], k0))

for i in range(gx.shape[0]):
    print(gx.shape[0] - i, gx[i])
    H = spop.HBDG(coor, ax, ay, NN, NNb = NNb, Wj = Wj, Sx = Sx, cutx = cutx, cuty = cuty, V = V, mu = mu, gammax = gx[i], alpha = alpha, delta = delta, phi = np.pi, qx = 0, periodicX = True)

    eigs, vecs = spLA.eigsh(H, k=k, sigma=0, which='LM')
    idx_sort = np.argsort(eigs)
    eigs = eigs[idx_sort]

    eig_arr[i, :] = eigs

    H_dif_basis = np.dot(vecs_0_hc, H.dot(vecs_0)) # H' = U^dagger H U
    eigs_dif_basis, U_dif_basis = LA.eigh(H_dif_basis)
    eig_arr_NB[i, :] = eigs_dif_basis

for i in range(k):
   plt.plot(gx, eig_arr[:, i], c = 'b')

for i in range(k0):
    plt.plot(gx, eig_arr_NB[:, i], c = 'r', ls = 'dashed')

plt.xlabel(r'$E_z$ (meV)')
plt.ylabel("Energy (meV)")
plt.savefig("EvsGamx.png")
plt.show()
"""
