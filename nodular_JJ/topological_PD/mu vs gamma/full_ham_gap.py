import sys
import time
import os
import gc

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.signal import argrelextrema
import scipy.linalg as LA
import scipy.sparse.linalg as spLA

import majoranaJJ.operators.sparse_operators as spop #sparse operators
from majoranaJJ.operators.potentials import Vjj #potential JJ
import majoranaJJ.lattice.nbrs as nb #neighbor arrays
import majoranaJJ.lattice.shapes as shps #lattice shapes
import majoranaJJ.modules.plots as plots #plotting functions
import majoranaJJ.modules.gamfinder as gamfinder
from majoranaJJ.modules.checkers import boundary_check as bc
import majoranaJJ.modules.checkers as check
###################################################
#Defining System
Nx = 3 #Number of lattice sites along x-direction
Ny = 360 #Number of lattice sites along y-direction
ax = 50 #lattice spacing in x-direction: [A]
ay = 50 #lattice spacing in y-direction: [A]
Wj = 10 #Junction region
cutx = 0 #width of nodule
cuty = 0 #height of nodule
Nx, Ny, cutx, cuty, Wj = check.junction_geometry_check(Nx, Ny, cutx, cuty, Wj)
print("Nx = {}, Ny = {}, cutx = {}, cuty = {}, Wj = {}".format(Nx, Ny, cutx, cuty, Wj))

Junc_width = Wj*ay*.10 #nm
SC_width = ((Ny - Wj)*ay*.10)/2 #nm
Nod_widthx = cutx*ax*.1 #nm
Nod_widthy = cuty*ay*.1 #nm
print("Nodule Width in x-direction = ", Nod_widthx, "(nm)")
print("Nodule Width in y-direction = ", Nod_widthy, "(nm)")
print("Junction Width = ", Junc_width, "(nm)")
print("Supercondicting Lead Width = ", SC_width, "(nm)")
###################################################coor = shps.square(Nx, Ny) #square lattice
coor = shps.square(Nx, Ny) #square lattice
NN = nb.NN_sqr(coor)
NNb = nb.Bound_Arr(coor)
lat_size = coor.shape[0]

Lx = (max(coor[:, 0]) - min(coor[:, 0]) + 1)*ax #Unit cell size in x-direction
Ly = (max(coor[:, 1]) - min(coor[:, 1]) + 1)*ay #Unit cell size in y-direction
###################################################
#Defining Hamiltonian parameters
alpha = 200 #Spin-Orbit Coupling constant: [meV*A]
phi = np.pi #SC phase difference
delta = 1 #Superconducting Gap: [meV]
Vsc = 0 #Amplitude of potential in SC region: [meV]
Vj = 0 #Amplitude of potential in junction region: [meV]
V = Vjj(coor, Wj = Wj, Vsc = Vsc, Vj = Vj, cutx = cutx, cuty = cuty)

mu_i = 0
mu_f = 20.0
res = 0.25
delta_mu = mu_f - mu_i
steps = int(delta_mu/(res)) + 1
mu = np.linspace(mu_i, mu_f, steps) #Chemical Potential: [meV]
dmu = 0

gi = 0
gf = 10
res = 0.25
steps = int((gf - gi)/(res)) + 1
gamx = np.linspace(gi, gf, steps)

q_steps = 501
qx = np.linspace(0, np.pi/Lx, q_steps) #kx in the first Brillouin zone

k = 4
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
    LE_Bands = np.zeros((qx.shape[0], mu.shape[0], gamx.shape[0]))
    top_array = np.zeros((mu.shape), dtype='int')

    for m in range(mu.shape[0]):
        for g in range(gamx.shape[0]):
            start  = time.perf_counter()
            for q in range(qx.shape[0]):
                print(mu.shape[0]-m, gamx.shape[0]-g, qx.shape[0]-q)
                H = spop.HBDG(coor, ax, ay, NN, NNb=NNb, Wj=Wj, cutx=cutx, cuty=cuty, V=V, mu=mu[m], alpha=alpha, delta=delta, phi=phi, gamx=gamx[g], qx=qx[q])
                eigs, vecs = spLA.eigsh(H, k=k, sigma=0, which='LM')
                idx_sort = np.argsort(eigs)
                eigs = eigs[idx_sort]
                LE_Bands[q, m, g] = eigs[int(k/2)]
            #if q == 0:
            #    top_array[i] = bc(LE_Bands[0, i, :], gx, max_gam = 1.0)
            #    print(top_array[i])
            end = time.perf_counter()
            print("Time: {} seconds".format(end-start))

    gap = np.zeros((mu.shape[0], gamx.shape[0]))
    #q_minima = np.zeros((mu.shape[0], gx.shape[0]))
    for i in range(LE_Bands.shape[1]):
        for j in range(LE_Bands.shape[2]):
            #eig_min_idx = np.array(argrelextrema(LE_Bands[:, i, j], np.less)[0])
            #q_minima[i,j] = qx[eig_min_idx]
            gap[i, j] = min(LE_Bands[:, i, j])

    #q_minima = np.array(q_minima)

    np.save("%s/gap_data Lx = %.1f Ly = %.1f Wsc = %.1f Wj = %.1f nodx = %.1f nody = %.1f alpha = %.1f delta = %.2f V_sc = %.1f phi = %.3f mu_i = %.1f mu_f = %.1f.npy" % (dirS, Lx*.1, Ly*.1, SC_width, Junc_width, Nod_widthx,  Nod_widthy, alpha, delta, Vsc, phi, mu[0], mu[-1]), gap)
    #np.save("%s/LE_Bands Lx = %.1f Ly = %.1f Wsc = %.1f Wj = %.1f nodx = %.1f nody = %.1f alpha = %.1f delta = %.2f V_sc = %.1f phi = %.3f mu_i = %.1f mu_f = %.1f.npy" % (dirS, Lx*.1, Ly*.1, SC_width, Junc_width, Nod_widthx,  Nod_widthy, alpha, delta, Vsc, phi, mu[0], mu[-1]), LE_Bands)
    #np.save("%s/Top_array Lx = %.1f Ly = %.1f Wsc = %.1f Wj = %.1f nodx = %.1f nody = %.1f alpha = %.1f delta = %.2f V_sc = %.1f phi = %.3f mu_i = %.1f mu_f = %.1f.npy" % (dirS, Lx*.1, Ly*.1, SC_width, Junc_width, Nod_widthx,  Nod_widthy, alpha, delta, Vsc, phi, mu[0], mu[-1]), top_array)
    #np.save("%s/q_minima Lx = %.1f Ly = %.1f Wsc = %.1f Wj = %.1f nodx = %.1f nody = %.1f alpha = %.1f delta = %.2f V_sc = %.1f phi = %.3f mu_i = %.1f mu_f = %.1f.npy" % (dirS, Lx*.1, Ly*.1, SC_width, Junc_width, Nod_widthx,  Nod_widthy, alpha, delta, Vsc, phi, mu[0], mu[-1]), q_minima)
    gc.collect()

    sys.exit()
else:
    gap = np.load("%s/gap_data Lx = %.1f Ly = %.1f Wsc = %.1f Wj = %.1f nodx = %.1f nody = %.1f alpha = %.1f delta = %.2f V_sc = %.1f phi = %.3f mu_i = %.1f mu_f = %.1f.npy" % (dirS, Lx*.1, Ly*.1, SC_width, Junc_width, Nod_widthx,  Nod_widthy, alpha, delta, Vsc, phi, mu[0], mu[-1]))
    #np.load("%s/q_minima Lx = %.1f Ly = %.1f Wsc = %.1f Wj = %.1f nodx = %.1f nody = %.1f alpha = %.1f delta = %.2f V_sc = %.1f phi = %.3f mu_i = %.1f mu_f = %.1f.npy" % (dirS, Lx*.1, Ly*.1, SC_width, Junc_width, Nod_widthx,  Nod_widthy, alpha, delta, V0, phi, mu[0], mu[-1]))
    print(gap)
    gap = gap/delta

    plt.contourf(gamx, mu, gap, 500, vmin = 0, vmax = max(gap.flatten()), cmap = 'hot')
    cbar = plt.colorbar()
    cbar.set_label(r'$E_{gap}/\Delta$')

    plt.xlabel(r'$\Gamma_x$ (meV)')
    plt.ylabel(r'$\mu$ (meV)')

    plt.xlim(gi, gf)

    title = r"$L_x$ = %.1f nm, $L_y$ = %.1f nm, $W_{SC}$ = %.1f nm, $W_j$ = %.1f nm, $nodule_x$ = %.1f nm, $nodule_y$ = %.1f nm, $\alpha$ = %.1f meV*A, $\phi$ = %.2f" % (Lx*.1, Ly*.1, SC_width, Junc_width, Nod_widthx, Nod_widthy, alpha, phi)
    plt.title(title, loc = 'center', wrap = True)
    plt.subplots_adjust(top=0.85)
    plt.savefig('juncwidth = {} SCwidth = {} V0 = {} nodwidthx = {} nodwidthy = {} Delta = {} Alpha = {} phi = {} mu_i = {} mu_f = {}.png'.format(Junc_width, SC_width, Vsc, Nod_widthx, Nod_widthy, delta, alpha, phi, mu_i, mu_f))
    plt.show()

    sys.exit()

##############################################################
