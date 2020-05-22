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

#Defining System
Nx = 4 #Number of lattice sites along x-direction
Ny = 150 #Number of lattice sites along y-direction
ax = 50 #lattice spacing in x-direction: [A]
ay = 50 #lattice spacing in y-direction: [A]
Wj = 10 #Junction region
Sx = int(Nx/2) #length of either side of nodule
cutx = (Nx - 2*Sx) #width of nodule
cuty = 0 #height of nodule

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
###############################################################################

#Defining Hamiltonian parameters
steps = 200

alpha = 200 #Spin-Orbit Coupling constant: [meV*A]
gammax_i = 0.0001 #parallel to junction: [meV]
phi = np.pi #SC phase difference
delta = 1.0 #Superconducting Gap: [meV]
V0 = 0 #Amplitude of potential : [meV]
V = V_BL(coor, Wj = Wj, Sx = Sx, cutx=cutx, cuty=cuty, V0 = V0)
mu = np.linspace(0, 20, steps) #Chemical Potential: [meV], 20

###############################################################################

#phase diagram mu vs gamx

gamx_0 = []
gamx_pi = []
mu_arr = []
for i in range(steps):
    print(steps-i)

    gxpi = gfLE(coor, ax, ay, NN, NNb = NNb, Wj = Wj,
    V = V, mu = mu[i], alpha = alpha, delta = delta, phi = np.pi, qx=0, periodicX = True)

    mu_arr.append(mu[i])
    gamx_pi.append(gxpi)

mu_arr, gamx_pi = np.array(mu_arr), np.array(gamx_pi)
print(gamx_pi.shape, mu_arr.shape)

plt.plot(gamx_pi, mu_arr, c='r', ls='solid')

plt.xlabel(r'$E_z$ (meV)')
plt.ylabel(r'$\mu$ (meV)')

plt.legend()
plt.savefig('juncwidth = {} SCwith = {} V0 = {} nodwidthx = {} nodwidthy = {} .png'.format(Junc_width, SC_width, V0, Nod_widthx, Nod_widthy ))
plt.show()

################################################################################
"""
#energy plot vs gamx
MU = 29 #fixed mu value
k = 4 #number of energy eigenvalues/vectors for nonlinear
k0 = 20#64 #perturbation energy eigs

H0 = spop.HBDG(coor, ax, ay, NN, NNb=NNb, Wj=Wj, Sx=Sx, cutx=cutx, cuty=cuty, V=V, mu=MU, gammax=0.0001, alpha=alpha, delta=delta, phi=np.pi, qx=0, periodicX=True)

eigs_0, vecs_0 = spLA.eigsh(H0, k=k0, sigma=0, which='LM')
idx_sort = np.argsort(eigs_0)
print(eigs_0[idx_sort])

vecs_0_hc = np.conjugate(np.transpose(vecs_0))

gx = np.linspace(0.0001, 3, steps)
eig_arr = np.zeros((gx.shape[0], k))
eig_arr_NB = np.zeros((gx.shape[0], k0))


H_G0 =  spop.HBDG(coor, ax, ay, NN, NNb = NNb, Wj = Wj, Sx = Sx, cutx = cutx, cuty = cuty, V = V, mu = MU, gammax = 0, alpha = alpha, delta = delta, phi = np.pi, qx = 0, periodicX = True)

H_G1 = spop.HBDG(coor, ax, ay, NN, NNb = NNb, Wj = Wj, Sx = Sx, cutx = cutx, cuty = cuty, V = V, mu = MU, gammax = 1, alpha = alpha, delta = delta, phi = np.pi, qx = 0, periodicX = True)

HG = H_G1 - H_G0

for i in range(gx.shape[0]):
    print(gx.shape[0] - i, gx[i])
    #H = spop.HBDG(coor, ax, ay, NN, NNb = NNb, Wj = Wj, Sx = Sx, cutx = cutx, cuty = cuty, V = V, mu = MU, gammax = gx[i], alpha = alpha, delta = delta, phi = np.pi, qx = 0, periodicX = True)

    H = H_G0 + gx[i]*HG
    #eigs, vecs = spLA.eigsh(H, k=k, sigma=0, which='LM')
    #idx_sort = np.argsort(eigs)
    #eigs = eigs[idx_sort]

    #eig_arr[i, :] = eigs

    H_dif_basis = np.dot(vecs_0_hc, H.dot(vecs_0)) # H' = U^dagger H U
    eigs_dif_basis, U_dif_basis = LA.eigh(H_dif_basis)
    eig_arr_NB[i, :] = eigs_dif_basis

#for i in range(k):
#   plt.plot(gx, eig_arr[:, i], c = 'b')

for i in range(k0):
    plt.plot(gx, eig_arr_NB[:, i], c = 'r', ls = 'dashed')

plt.xlabel(r'$E_z$ (meV)')
plt.ylabel("Energy (meV)")
plt.savefig("EvsGamx.png")
plt.show()
"""
