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
Wj = 40 #Junction region
cutx = 0 #width of nodule
cuty = 0 #height of nodule
Nx, Ny, cutx, cuty, Wj = check.junction_geometry_check(Ny, Nx, Wj, cutx, cuty)
print(Nx, Ny, cutx, cuty, Wj)
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
alpha = 200 #Spin-Orbit Coupling constant: [meV*A]
phi = np.pi #SC phase difference
delta = 1 #Superconducting Gap: [meV]
Vsc = 0 #Amplitude of potential in SC region: [meV]
Vj = 0 #Amplitude of potential in junction region: [meV]
V = Vjj(coor, Wj = Wj, Vsc = Vsc, Vj = Vj, cutx = cutx, cuty = cuty)

mu_i = 0
mu_f = 20
res = 0.1
delta_mu = mu_f - mu_i
steps = int(delta_mu/(0.5*res)) + 1
mu = np.linspace(mu_i, mu_f, steps) #Chemical Potential: [meV]

gi = 0
gf = 1.0
res = 0.05
steps = int((gf - gi)/(0.5*res)) + 1
gx = np.linspace(gi, gf, steps)

q_steps = 51
qx = np.linspace(0, np.pi/Lx, q_steps) #kx in the first Brillouin zone

k = 50
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
    LE_Bands = np.zeros((qx.shape[0], mu.shape[0], gx.shape[0]))
    top_array = np.zeros((mu.shape), dtype='int')
    for q in range(qx.shape[0]):
        print(qx.shape[0]-q)

        if q == 0:
            Q = 1e-4*(np.pi/Lx)
        else:
            Q = qx[q]

        for i in range(mu.shape[0]):
            start = time.perf_counter()
            if q == 0 or top_array[i] == 1:
                H0 = spop.HBDG(coor, ax, ay, NN, NNb=NNb, Wj=Wj, cutx=cutx, cuty=cuty, V=V, mu=mu[i], alpha=alpha, delta=delta, phi=phi, gammax=1e-4, qx=Q, Tesla = False, Zeeman_in_SC = False, SOC_in_SC = True) #gives low energy basis
                eigs_0, vecs_0 = spLA.eigsh(H0, k=k, sigma=0, which='LM')
                vecs_0_hc = np.conjugate(np.transpose(vecs_0)) #hermitian conjugate
                H_G0 = spop.HBDG(coor, ax, ay, NN, NNb=NNb, Wj=Wj, cutx=cutx, cuty=cuty, V=V, mu=mu[i], gammax=0, alpha=alpha, delta=delta, phi=phi, qx=qx[q], Tesla = False, Zeeman_in_SC = False, SOC_in_SC = False) #Matrix that consists of everything in the Hamiltonian except for the Zeeman energy in the x-direction
                H_G1 = spop.HBDG(coor, ax, ay, NN, NNb=NNb, Wj=Wj, cutx=cutx, cuty=cuty, V=V, mu=mu[i], gammax=1, alpha=alpha, delta=delta, phi=phi, qx=qx[q], Tesla = False, Zeeman_in_SC = False, SOC_in_SC = False) #Hamiltonian with ones on Zeeman energy along x-direction sites
                HG = H_G1 - H_G0 #the proporitonality matrix for gamma-x, it is ones along the sites that have a gamma value
                HG0_DB = np.dot(vecs_0_hc, H_G0.dot(vecs_0))
                HG_DB = np.dot(vecs_0_hc, HG.dot(vecs_0))

                for j in range(gx.shape[0]):
                    print(qx.shape[0]-q, mu.shape[0]-i, gx.shape[0]-j)
                    H_DB = HG0_DB + gx[j]*HG_DB
                    eigs_DB, U_DB = LA.eigh(H_DB)
                    LE_Bands[q, i, j] = eigs_DB[int(k/2)]
                #plt.plot(gx, LE_Bands[q, i, :], c='b')
                #plt.show()

                if q == 0:
                    top_array[i] = bc(LE_Bands[0, i, :], gx, max_gam = 1.0)
                    print(top_array[i])

                print("Mu = ", mu[i])
                print("Minimum energy w.r.t Gammax = ", min(LE_Bands[0, i, :]))
                end = time.perf_counter()
                print("Time: {} seconds".format(end-start))

                if (min(LE_Bands[0, i, :])) < -1:
                    print("Negative eigenvalue minima found. Something is wrong")
                    sys.exit()

    gap = np.zeros((mu.shape[0], gx.shape[0]))
    q_minima = []
    for i in range(LE_Bands.shape[1]):
        for j in range(LE_Bands.shape[2]):
            eig_min_idx = np.array(argrelextrema(LE_Bands[:, i, j], np.less)[0])
            q_minima.append(qx[eig_min_idx])
            gap[i, j] = abs(min(LE_Bands[:, i, j]))
            #index of minima
    q_minima = np.array(q_minima)

    np.save("%s/gap_data Lx = %.1f Ly = %.1f Wsc = %.1f Wj = %.1f nodx = %.1f nody = %.1f alpha = %.1f delta = %.2f V_sc = %.1f phi = %.3f mu_i = %.1f mu_f = %.1f.npy" % (dirS, Lx*.1, Ly*.1, SC_width, Junc_width, Nod_widthx,  Nod_widthy, alpha, delta, Vsc, phi, mu[0], mu[-1]), gap)
    np.save("%s/LE_Bands Lx = %.1f Ly = %.1f Wsc = %.1f Wj = %.1f nodx = %.1f nody = %.1f alpha = %.1f delta = %.2f V_sc = %.1f phi = %.3f mu_i = %.1f mu_f = %.1f.npy" % (dirS, Lx*.1, Ly*.1, SC_width, Junc_width, Nod_widthx,  Nod_widthy, alpha, delta, Vsc, phi, mu[0], mu[-1]), LE_Bands)
    np.save("%s/Top_array Lx = %.1f Ly = %.1f Wsc = %.1f Wj = %.1f nodx = %.1f nody = %.1f alpha = %.1f delta = %.2f V_sc = %.1f phi = %.3f mu_i = %.1f mu_f = %.1f.npy" % (dirS, Lx*.1, Ly*.1, SC_width, Junc_width, Nod_widthx,  Nod_widthy, alpha, delta, Vsc, phi, mu[0], mu[-1]), top_array)
    np.save("%s/gx Lx = %.1f Ly = %.1f Wsc = %.1f Wj = %.1f nodx = %.1f nody = %.1f alpha = %.1f delta = %.2f V_sc = %.1f phi = %.3f mu_i = %.1f mu_f = %.1f.npy" % (dirS, Lx*.1, Ly*.1, SC_width, Junc_width, Nod_widthx,  Nod_widthy, alpha, delta, Vsc, phi, mu[0], mu[-1]), gx)
    np.save("%s/mu Lx = %.1f Ly = %.1f Wsc = %.1f Wj = %.1f nodx = %.1f nody = %.1f alpha = %.1f delta = %.2f V_sc = %.1f phi = %.3f mu_i = %.1f mu_f = %.1f.npy" % (dirS, Lx*.1, Ly*.1, SC_width, Junc_width, Nod_widthx,  Nod_widthy, alpha, delta, Vsc, phi, mu[0], mu[-1]), mu)
    np.save("%s/q_minima Lx = %.1f Ly = %.1f Wsc = %.1f Wj = %.1f nodx = %.1f nody = %.1f alpha = %.1f delta = %.2f V_sc = %.1f phi = %.3f mu_i = %.1f mu_f = %.1f.npy" % (dirS, Lx*.1, Ly*.1, SC_width, Junc_width, Nod_widthx,  Nod_widthy, alpha, delta, Vsc, phi, mu[0], mu[-1]), q_minima)
    gc.collect()

    sys.exit()
else:
    gap = np.load("%s/gap_data Lx = %.1f Ly = %.1f Wsc = %.1f Wj = %.1f nodx = %.1f nody = %.1f alpha = %.1f delta = %.2f V_sc = %.1f phi = %.3f mu_i = %.1f mu_f = %.1f.npy" % (dirS, Lx*.1, Ly*.1, SC_width, Junc_width, Nod_widthx,  Nod_widthy, alpha, delta, Vsc, phi, mu[0], mu[-1]))
    gx = np.load("%s/gx Lx = %.1f Ly = %.1f Wsc = %.1f Wj = %.1f nodx = %.1f nody = %.1f alpha = %.1f delta = %.2f V_sc = %.1f phi = %.3f mu_i = %.1f mu_f = %.1f.npy" % (dirS, Lx*.1, Ly*.1, SC_width, Junc_width, Nod_widthx,  Nod_widthy, alpha, delta, Vsc, phi, mu[0], mu[-1]))
    mu =  np.load("%s/mu Lx = %.1f Ly = %.1f Wsc = %.1f Wj = %.1f nodx = %.1f nody = %.1f alpha = %.1f delta = %.2f V_sc = %.1f phi = %.3f mu_i = %.1f mu_f = %.1f.npy" % (dirS, Lx*.1, Ly*.1, SC_width, Junc_width, Nod_widthx,  Nod_widthy, alpha, delta, Vsc, phi, mu[0], mu[-1]))
    #np.load("%s/q_minima Lx = %.1f Ly = %.1f Wsc = %.1f Wj = %.1f nodx = %.1f nody = %.1f alpha = %.1f delta = %.2f V_sc = %.1f phi = %.3f mu_i = %.1f mu_f = %.1f.npy" % (dirS, Lx*.1, Ly*.1, SC_width, Junc_width, Nod_widthx,  Nod_widthy, alpha, delta, V0, phi, mu[0], mu[-1]))

    gap = gap/delta

    plt.contourf(gx, mu, gap, 100, vmin = 0, vmax = max(gap.flatten()), cmap = 'magma')
    cbar = plt.colorbar()
    cbar.set_label(r'$E_{gap}/\Delta$')

    plt.xlabel(r'$\Gamma_x$ (meV)')
    plt.ylabel(r'$\mu$ (meV)')

    plt.xlim(gi, gf)

    title = r"$L_x$ = %.1f nm, $L_y$ = %.1f nm, $W_{SC}$ = %.1f nm, $W_j$ = %.1f nm, $nodule_x$ = %.1f nm, $nodule_y$ = %.1f nm, $\alpha$ = %.1f meV*A, $\phi$ = %.2f " % (Lx*.1, Ly*.1, SC_width, Junc_width, Nod_widthx, Nod_widthy, alpha, phi)
    plt.title(title, loc = 'center', wrap = True)
    plt.subplots_adjust(top=0.85)
    plt.savefig('juncwidth = {} SCwidth = {} V0 = {} nodwidthx = {} nodwidthy = {} Delta = {} Alpha = {} phi = {} mu_i = {} mu_f = {}.png'.format(Junc_width, SC_width, Vsc, Nod_widthx, Nod_widthy, delta, alpha, phi, mu_i, mu_f))
    plt.show()

    sys.exit()

##############################################################
