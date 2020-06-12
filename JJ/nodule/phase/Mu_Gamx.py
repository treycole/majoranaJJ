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

#Defining Hamiltonian parameters
steps = 100

alpha = 200 #Spin-Orbit Coupling constant: [meV*A]
gammax_i = 0.0001 #parallel to junction: [meV]
phi = np.pi #SC phase difference
delta = 1.0 #Superconducting Gap: [meV]
V0 = 0 #Amplitude of potential : [meV]
V = V_BL(coor, Wj = Wj, cutx=cutx, cuty=cuty, V0 = V0)
mu = np.linspace(0, 20, steps) #Chemical Potential: [meV], 20

##############################################################

#phase diagram mu vs gamx
num_bound = 3

gamx_pi = np.zeros((steps, num_bound))#[[]]*num_bound
gamx_0 = []
mu_arr = [[],[]] #an array for each phase diff value

for i in range(steps):
    print(steps-i)

    gxpi = gfLE(coor, ax, ay, NN, NNb = NNb, Wj = Wj,
    V = V, mu = mu[i], gi = -0.0001, gf = 1.2, alpha = alpha, delta = delta, phi = np.pi, tol = 0.01, steps = 200)
    print(gxpi)
    #gx0 = gfLE(coor, ax, ay, NN, NNb = NNb, Wj = Wj,
    #V = V, mu = mu[i], gi = -0.0001, gf = 0.36, alpha = alpha, delta = delta, phi = 0, tol = 0.01, steps = 200)

    #mu_arr[0].append(mu[i])
    for j in range(num_bound):
        if j >= gxpi.size:
            gamx_pi[i, j] = None
        else:
            gamx_pi[i, j]= gxpi[j]
    #gamx_pi.append(gxpi)
    #if gx0.size != 0:
    #    mu_arr[1].append(mu[i])
    #    gamx_0.append(gx0[0])

mu_arr, gamx_pi, gamx_0 = np.array(mu_arr), np.array(gamx_pi), np.array(gamx_0)
print(gamx_pi, gamx_pi.shape, mu.shape)

plt.plot(gamx_pi, mu, c='r')

#plt.plot(gamx_0, mu_arr[1], c='b', label = r'$\phi = 0$')

plt.xlabel(r'$E_z$ (meV)')
plt.ylabel(r'$\mu$ (meV)')

#plt.xlim(0, 0.35)
#plt.ylim(78, 80.1)

#plt.legend()
plt.title('Low-Energy Basis Calculated')
plt.savefig('juncwidth = {} SCwidth = {} V0 = {} nodwidthx = {} nodwidthy = {} #.png'.format(Junc_width, SC_width, V0, Nod_widthx, Nod_widthy ))
plt.show()

sys.exit()

##############################################################

#state plot
MU = 2
GX = 0.75

H = spop.HBDG(coor, ax, ay, NN, NNb = NNb, Wj = Wj, V = V, mu = MU, gammax = GX, alpha = alpha, delta = delta, phi = np.pi, qx= 0,periodicX = True, periodicY = False)

eigs, states = spLA.eigsh(H, k=8, sigma=0, which='LM')
idx_sort = np.argsort(eigs)
print(eigs[idx_sort])
plots.state_cmap(coor, eigs, states, n=4, savenm='prob_density_nodule_n=4.png')
plots.state_cmap(coor, eigs, states, n=5, savenm='prob_density_nodule_n=5.png')
plots.state_cmap(coor, eigs, states, n=6, savenm='prob_density_nodule_n=6.png')
