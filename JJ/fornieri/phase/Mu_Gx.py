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

Nx = 3 #Number of lattice sites along x-direction
Ny = 80 #Number of lattice sites along y-direction
ax = 50 #lattice spacing in x-direction: [A]
ay = 50 #lattice spacing in y-direction: [A]
Wj = 16  #Junction region
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
##############################################################

coor = shps.square(Nx, Ny) #square lattice
NN = nb.NN_sqr(coor)
NNb = nb.Bound_Arr(coor)
lat_size = coor.shape[0]
print("Lattice Size: ", lat_size)

Lx = (max(coor[:, 0]) - min(coor[:, 0]) + 1)*ax #Unit cell size in x-direction
Ly = (max(coor[:, 1]) - min(coor[:, 1]) + 1)*ay #Unit cell size in y-direction

##############################################################
steps = 150

alpha = 100 #Spin-Orbit Coupling constant: [meV*A]
gx = 0 #parallel to junction: [meV]
gz = 1e-6
phi = np.pi #SC phase difference
delta = 0.15 #Superconducting Gap: [meV]
V0 = 0 #Amplitude of potential : [meV]
V = V_BL(coor, Wj = Wj, cutx=cutx, cuty=cuty, V0 = V0)
mu = np.linspace(78, 80, steps) #Chemical Potential: [meV]

##############################################################
"""
gamx_pi = []
gamx_0 = []
mu_arr = []

for i in range(steps):
    print(steps-i)
    gammax0 = gf(
        coor, ax, ay, NN, mu[i], NNb = NNb, Wj = Wj, alpha = alpha, delta = delta, phi = 0, gammaz = gz, k=2
        )[0]

    gammaxpi = gf(
        coor, ax, ay, NN, mu[i], NNb = NNb, Wj = Wj, alpha = alpha, delta = delta, phi = np.pi, gammaz = gz, k=2
        )[0]

    mu_arr.append(mu[i])
    gamx_0.append(gammax0)
    gamx_pi.append(gammaxpi)

mu_arr, gamx_0, gamx_pi = np.array(mu_arr), np.array(gamx_0), np.array(gamx_pi)
print(gamx_0.shape, mu_arr.shape)

plt.plot(gamx_0, mu, c='k', ls='solid', label = r'$\phi = 0$')
plt.plot(gamx_pi, mu, c='r', ls='solid', label = r'$\phi=\pi$')

plt.xlabel(r'$E_z$ (meV)')
plt.ylabel(r'$\mu$ (meV)')

plt.xlim(0, 0.35)
plt.ylim(78, 80.1)

plt.legend()
plt.savefig('mu_gx.png')
plt.show()
"""
##############################################################

num_bound = 1
gamx_pi = np.zeros((steps, num_bound))#[]
gamx_0 = np.zeros((steps, num_bound))#[]
for i in range(steps):
    print(steps-i)

    gxpi = gfLE(coor, ax, ay, NN, NNb = NNb, Wj = Wj,
    V = V, mu = mu[i], gi = -0.0001, gf = 0.36, alpha = alpha, delta = delta, phi = np.pi, steps = 200)
    
    gx0 = gfLE(coor, ax, ay, NN, NNb = NNb, Wj = Wj,
    V = V, mu = mu[i], gi = -0.0001, gf = 0.36, alpha = alpha, delta = delta, phi = 0, steps = 200)

    #mu_arr[0].append(mu[i])
    for j in range(num_bound):
        if j >= gxpi.size:
            gamx_pi[i, j] = None
        else:
            gamx_pi[i, j]= gxpi[j]
        if j >= gx0.size:
            gamx_0[i, j] = None
        else:
            gamx_0[i, j]= gx0[j]

gamx_0, gamx_pi = np.array(gamx_0), np.array(gamx_pi)
print(gamx_0.shape, mu.shape)

plt.plot(gamx_0, mu, c='k', ls='solid', label = r'$\phi = 0$')
plt.plot(gamx_pi, mu, c='r', ls='solid', label = r'$\phi=\pi$')

plt.xlabel(r'$E_z$ (meV)')
plt.ylabel(r'$\mu$ (meV)')

plt.xlim(0, 0.35)
plt.ylim(78, 80.1)

plt.legend()
plt.savefig('mu_gx_LE.png')
plt.show()

##############################################################
"""
MU = 78.96 #fixed mu value
k = 48 #spectra energy eigs
k0 = k #number of basis eigenvectors
Lx = (max(coor[:, 0]) - min(coor[:, 0]) + 1)*ax #Unit cell size in x-direction

H0 = spop.HBDG(coor, ax, ay, NN, NNb=NNb, Wj=Wj, cutx=cutx, cuty=cuty, V=V, mu=MU, gammaz = 0.01, alpha=alpha, delta=delta, phi=phi, qx=0.0001*(np.pi/Lx), periodicX=True)

eigs_0, vecs_0 = spLA.eigsh(H0, k=k0, sigma = 0,  which='LM')
idx_sort = np.argsort(eigs_0)
#vecs_0 = vecs_0[:, idx_sort]
print(eigs_0[idx_sort][int(k/2):])
#plots.state_cmap(coor, eigs_0, vecs_0, n=12, savenm='prob_density_nodule_n=12.png')
#plots.state_cmap(coor, eigs_0, vecs_0, n=13, savenm='prob_density_nodule_n=12.png')

vecs_0_hc = np.conjugate(np.transpose(vecs_0))

gx = np.linspace(0, 0.5, steps)
eig_arr = np.zeros((gx.shape[0], k))
eig_arr_NB = np.zeros((gx.shape[0], k0))

H_G0 =  spop.HBDG(coor, ax, ay, NN, NNb = NNb, Wj = Wj, cutx = cutx, cuty = cuty, V = V, mu = MU, gammax = 0, alpha = alpha, delta = delta, phi = phi, qx = 0, periodicX = True)

H_G1 = spop.HBDG(coor, ax, ay, NN, NNb = NNb, Wj = Wj, cutx = cutx, cuty = cuty, V = V, mu = MU, gammax = 1, alpha = alpha, delta = delta, phi = phi, qx = 0, periodicX = True)

HG = H_G1 - H_G0

for i in range(gx.shape[0]):
    print(gx.shape[0] - i, gx[i])

    H = H_G0 + gx[i]*HG
    eigs, vecs = spLA.eigsh(H, k=k, sigma=0, which='LM')

    H_DB = np.dot(vecs_0_hc, H.dot(vecs_0)) # H' = U^dagger H U
    eigs_DB, U_DB = LA.eigh(H_DB)

    idx_sort = np.argsort(eigs)
    eigs = eigs[idx_sort]
    eig_arr[i, :] = eigs

    idx_sort = np.argsort(eigs_DB)
    eigs_DB = eigs_DB[idx_sort]
    eig_arr_NB[i, :] = eigs_DB

for i in range(k):
   plt.plot(gx, eig_arr[:, i], c = 'b')
   plt.plot(gx, eig_arr_NB[:, i], c = 'r', ls = 'dashed')

plt.title("mu = {} (meV)".format(MU))
plt.xlabel(r'$E_z$ (meV)')
plt.ylabel("Energy (meV)")
plt.savefig("EvsGamx.png")
plt.show()
"""
