import sys
import time
import os
dir = os.getcwd()
import numpy as np
import gc
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.sparse as sparse
from scipy.signal import argrelextrema
import scipy.linalg as LA
import scipy.sparse.linalg as spLA

import majoranaJJ.operators.sparse.qmsops as spop #sparse operators
import majoranaJJ.lattice.nbrs as nb #neighbor arrays
import majoranaJJ.lattice.shapes as shps #lattice shapes
import majoranaJJ.modules.plots as plots #plotting functions
from majoranaJJ.modules.gamfinder import gamfinder as gf
from majoranaJJ.modules.gamfinder import gamfinder_lowE as gfLE
from majoranaJJ.modules.top_checker import boundary_check as bc
from majoranaJJ.operators.potentials.barrier_leads import V_BL
###################################################
#Defining System
Nx = 12 #Number of lattice sites along x-direction
Ny = 408 #Number of lattice sites along y-direction
ax = 50 #lattice spacing in x-direction: [A]
ay = 50 #lattice spacing in y-direction: [A]
Wj = 8 #Junction region
cutx = 3 #width of nodule
cuty = 3 #height of nodule

Junc_width = Wj*ay*.10 #nm
SC_width = ((Ny - Wj)*ay*.10)/2 #nm
Nod_widthx = cutx*ax*.1 #nm
Nod_widthy = cuty*ay*.1 #nm
print("Nodule Width in x-direction = ", Nod_widthx, "(nm)")
print("Nodule Width in y-direction = ", Nod_widthy, "(nm)")
print("Junction Width = ", Junc_width, "(nm)")
print("Supercondicting Lead Width = ", SC_width, "(nm)")
###################################################
coor = shps.square(Nx, Ny) #square lattice
NN = nb.NN_sqr(coor)
NNb = nb.Bound_Arr(coor)
lat_size = coor.shape[0]
print("Lattice Size: ", lat_size)
Lx = (max(coor[:, 0]) - min(coor[:, 0]) + 1)*ax #Unit cell size in x-direction
Ly = (max(coor[:, 1]) - min(coor[:, 1]) + 1)*ay #Unit cell size in y-direction
###################################################
#Defining Hamiltonian parameters
alpha = 100 #Spin-Orbit Coupling constant: [meV*A]
phi = 0*np.pi #SC phase difference
delta = 1 #Superconducting Gap: [meV]
V0 = 50 #Amplitude of potential : [meV]
V = V_BL(coor, Wj = Wj, cutx=cutx, cuty=cuty, V0 = V0)

mu_i = 50
mu_f = 100
res = 0.1
delta_mu = mu_f - mu_i
steps = int(delta_mu/res)
mu = np.linspace(mu_i, mu_f, steps) #Chemical Potential: [meV]

gi = 0
gf = 1.3
tol = 0.01
n_steps = int((gf - gi)/(0.5*tol)) + 1
gx = np.linspace(gi, gf, n_steps)

q_steps = 51
qx = np.linspace(0, np.pi/Lx, q_steps) #kx in the first Brillouin zone

k = 100
###################################################
#phase diagram mu vs gamx
dirS = 'gap_data'
if not os.path.exists(dirS):
    os.makedirs(dirS)
try:
    PLOT = str(sys.argv[1])
except:
    PLOT = 'F'
    
if PLOT != 'P':
    LE_Bands = np.zeros((mu.shape[0], gx.shape[0]))
    gap = np.zeros((mu.shape[0], gx.shape[0]))
    for i in range(mu.shape[0]):
        H0 = spop.HBDG(coor, ax, ay, NN, NNb=NNb, Wj=Wj, cutx=cutx, cuty=cuty, V=V, mu=mu[i], alpha=alpha, delta=delta, phi=phi, gammax=1e-4, qx=0, periodicX=True) #gives low energy basis
        eigs_0, vecs_0 = spLA.eigsh(H0, k=k, sigma=0, which='LM')
        vecs_0_hc = np.conjugate(np.transpose(vecs_0)) #hermitian conjugate
        H_G0 = spop.HBDG(coor, ax, ay, NN, NNb=NNb, Wj=Wj, cutx=cutx, cuty=cuty, V=V, mu=mu[i], gammax=0, alpha=alpha, delta=delta, phi=phi, qx=0, periodicX=True) #Matrix that consists of everything in the Hamiltonian except for the Zeeman energy in the x-direction
        H_G1 = spop.HBDG(coor, ax, ay, NN, NNb=NNb, Wj=Wj, cutx=cutx, cuty=cuty, V=V, mu=mu[i], gammax=1, alpha=alpha, delta=delta, phi=phi, qx=0, periodicX=True) #Hamiltonian with ones on Zeeman energy along x-direction sites
        HG = H_G1 - H_G0    #the proporitonality matrix for gamma-x, it is ones along the sites that have a gamma value
        HG0_DB = np.dot(vecs_0_hc, H_G0.dot(vecs_0))
        HG_DB = np.dot(vecs_0_hc, HG.dot(vecs_0))
        start = time.perf_counter()
        for j in range(gx.shape[0]):
            print(mu.shape[0]-i, gx.shape[0]-j)
            H_DB = HG0_DB + gx[j]*HG_DB
            eigs_DB, U_DB = LA.eigh(H_DB)
            gap[i, j] = eigs_DB[int(k/2)]
        end = time.perf_counter()
        print("Time: ", end-start)

    np.save("%s/gap_data Lx = %.1f Ly = %.1f Wsc = %.1f Wj = %.1f nodx = %.1f nody = %.1f alpha = %.1f delta = %.2f V_sc = %.1f phi = %.3f mu_i = %.1f mu_f = %.1f.npy" % (dirS, Lx*.1, Ly*.1, SC_width, Junc_width, Nod_widthx,  Nod_widthy, alpha, delta, V0, phi, mu[0], mu[-1]), gap)
    np.save("%s/gx Lx = %.1f Ly = %.1f Wsc = %.1f Wj = %.1f nodx = %.1f nody = %.1f alpha = %.1f delta = %.2f V_sc = %.1f phi = %.3f mu_i = %.1f mu_f = %.1f.npy" % (dirS, Lx*.1, Ly*.1, SC_width, Junc_width, Nod_widthx,  Nod_widthy, alpha, delta, V0, phi, mu[0], mu[-1]), gx)
    np.save("%s/mu Lx = %.1f Ly = %.1f Wsc = %.1f Wj = %.1f nodx = %.1f nody = %.1f alpha = %.1f delta = %.2f V_sc = %.1f phi = %.3f mu_i = %.1f mu_f = %.1f.npy" % (dirS, Lx*.1, Ly*.1, SC_width, Junc_width, Nod_widthx,  Nod_widthy, alpha, delta, V0, phi, mu[0], mu[-1]), mu)
    gc.collect()

    sys.exit()
else:
    gap = np.load("%s/gap_data Lx = %.1f Ly = %.1f Wsc = %.1f Wj = %.1f nodx = %.1f nody = %.1f alpha = %.1f delta = %.2f V_sc = %.1f phi = %.3f mu_i = %.1f mu_f = %.1f.npy" % (dirS, Lx*.1, Ly*.1, SC_width, Junc_width, Nod_widthx,  Nod_widthy, alpha, delta, V0, phi, mu[0], mu[-1]))
    gx = np.load("%s/gx Lx = %.1f Ly = %.1f Wsc = %.1f Wj = %.1f nodx = %.1f nody = %.1f alpha = %.1f delta = %.2f V_sc = %.1f phi = %.3f mu_i = %.1f mu_f = %.1f.npy" % (dirS, Lx*.1, Ly*.1, SC_width, Junc_width, Nod_widthx,  Nod_widthy, alpha, delta, V0, phi, mu[0], mu[-1]))
    mu =  np.load("%s/mu Lx = %.1f Ly = %.1f Wsc = %.1f Wj = %.1f nodx = %.1f nody = %.1f alpha = %.1f delta = %.2f V_sc = %.1f phi = %.3f mu_i = %.1f mu_f = %.1f.npy" % (dirS, Lx*.1, Ly*.1, SC_width, Junc_width, Nod_widthx,  Nod_widthy, alpha, delta, V0, phi, mu[0], mu[-1]))

    gap = gap/delta

    plt.contourf(gx, mu, gap, 100, vmin = 0, vmax = max(gap.flatten()), cmap = 'magma')
    cbar = plt.colorbar()
    cbar.set_label(r'$E_{gap}/\Delta$')

    plt.xlabel(r'$\Gamma_x$ (meV)')
    plt.ylabel(r'$\mu$ (meV)')

    plt.xlim(gi, gf)

    title = r"$L_x =$ {} nm, $L_y =$ {} nm, SC width = {} nm, $W_j =$ {} nm, $nodule_x = ${} nm, $nodule_y = ${} nm, $\alpha = $ {} meV*A, $\phi =$ {} ".format(Lx*.1, Ly*.1, SC_width, Junc_width, Nod_widthx, Nod_widthy, alpha, phi)
    plt.title(title, loc = 'center', wrap = True)
    plt.savefig('juncwidth = {} SCwidth = {} V0 = {} nodwidthx = {} nodwidthy = {} Delta = {} Alpha = {} phi = {} mu_i = {} mu_f = {}.png'.format(Junc_width, SC_width, V0, Nod_widthx, Nod_widthy, delta, alpha, phi, mu_i, mu_f))
    plt.show()

    sys.exit()

##############################################################
