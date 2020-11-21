import sys
import os
import time
dir = os.getcwd()
import numpy as np
import gc
import matplotlib.pyplot as plt
import matplotlib.cm as cm
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
import majoranaJJ.operators.sparse.k_dot_p as kp

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

res = 0.1
mu_i = 55
mu_f = 55.2
delta_mu = mu_f - mu_i
steps = int(delta_mu/res)
mu = np.linspace(mu_i, mu_f, steps) #Chemical Potential: [meV]

gi = 0.2
gf = 0.3
tol = 0.05
n_steps = int((gf - gi)/tol)
gx = np.linspace(gi, gf, n_steps)

steps_k = 101
qx = np.linspace(0, np.pi/Lx, steps_k)

k = 500
###################################################
dirS = 'gap_data'
if not os.path.exists(dirS):
    os.makedirs(dirS)
try:
    PLOT = str(sys.argv[1])
except:
    PLOT = 'F'
if PLOT != 'P':
    #phase diagram mu vs gamx
    gap = np.zeros((mu.shape[0], gx.shape[0]))
    #gap_k0 = np.zeros((mu.shape[0], gx.shape[0]))
    LE_bands = np.zeros((mu.shape[0], gx.shape[0], steps_k))

    H = spop.H0(coor, ax, ay, NN, NNb=NNb,alpha=alpha, V=V, gammax=1e-4, mu=mu[0], qx=0, periodicX=True)#kp.H0(H0, Hq, Hqq, Hgam, q = 0, gx = 1e-4)
    eigs_0, vecs_0 = spLA.eigsh(H, k=k, sigma=0, which='LM')
    vecs_0_hc = np.conjugate(np.transpose(vecs_0)) #hermitian conjugate
    vecs_0_c = np.conjugate(vecs_0)

    H0, Hq, Hqq, DELTA, Hgam = kp.Hq(coor, ax, ay, NN, NNb = NNb, Wj = Wj, cutx = cutx, cuty = cuty, V = V, mu = mu[0], alpha = alpha, delta = delta, phi = phi, periodicX = True)

    H0_DB = np.dot(vecs_0_hc, H0.dot(vecs_0))
    Hq_DB = np.dot(vecs_0_hc, Hq.dot(vecs_0))
    Hqq_DB = np.dot(vecs_0_hc, Hqq.dot(vecs_0))
    DELTA_DB = np.dot(vecs_0_hc, DELTA.dot(vecs_0_c))
    Hgam_DB = np.dot(vecs_0_hc, Hgam.dot(vecs_0))
    MU = np.eye(H0_DB.shape[0])

    mu_cp = mu[0]
    for i in range(mu.shape[0]):
        if (mu[i] - mu_cp > 1):
            print("Recalculating Low Energy Basis ...")
            mu_cp = mu[i]
            H0, Hq, Hqq, DELTA, Hgam = kp.Hq(coor, ax, ay, NN, NNb = NNb, Wj = Wj, cutx = cutx, cuty = cuty, V = V, mu = mu_cp, alpha = alpha, delta = delta, phi = phi, periodicX = True)

            H = kp.H0(H0, Hq, Hqq, Hgam, q = 1e-6, gx = 0)
            eigs_0, vecs_0 = spLA.eigsh(H, k=k, sigma=0, which='LM')
            vecs_0_hc = np.conjugate(np.transpose(vecs_0)) #hermitian conjugate
            vecs_0_c = np.conjugate(vecs_0)

            H0_DB = np.dot(vecs_0_hc, H0.dot(vecs_0))
            Hq_DB = np.dot(vecs_0_hc, Hq.dot(vecs_0))
            Hqq_DB = np.dot(vecs_0_hc, Hqq.dot(vecs_0))
            DELTA_DB = np.dot(vecs_0_hc, DELTA.dot(vecs_0_c))
            Hgam_DB = np.dot(vecs_0_hc, Hgam.dot(vecs_0))
            MU = np.eye(H0_DB.shape[0])
        for j in range(gx.shape[0]):
            start = time.perf_counter()
            for m in range(qx.shape[0]):
                print(mu.shape[0] - i, gx.shape[0] - j, qx.shape[0] - m)
                H = kp.HBDG_LE(H0_DB, Hq_DB, Hqq_DB, DELTA_DB, Hgam_DB, MU, q = qx[m], d_mu = mu[i] - mu_cp, gx = gx[j])

                eigs_DB, U_DB = LA.eigh(H)
                if m == 0:
                    bands = np.zeros((steps_k, eigs_DB.shape[0]))
                bands[m, :] = eigs_DB
                #print(eigs_DB, eigs_DB[int(k)])
                LE_bands[i, j, m] = eigs_DB[int(k)]
            end = time.perf_counter()
            print("Time: ", end-start)
            print(min(LE_bands[i, j, :]))
            for n in range(12): #bands.shape[1]):
                plt.plot(qx, bands[:, int(k) - 6 +n], c ='mediumblue', linestyle = 'solid')
                plt.plot(-qx, bands[:, int(k) - 6 +n], c ='mediumblue', linestyle = 'solid')
            plt.plot(qx, 0*qx, c = 'k', linestyle='solid', lw=1)
            plt.plot(-qx, 0*qx, c = 'k', linestyle='solid', lw=1)
            plt.title('gx = {}, mu = {}'.format(gx[j], mu[i]))
            #plt.ylim(-2*delta, 2*delta)
            plt.show()
            sys.exit()

    for i in range(LE_bands.shape[0]):
        for j in range(LE_bands.shape[1]):
            gap[i, j] = min(LE_bands[i, j, :])
            #index of minima

    np.save("%s/gap_data Lx = %.1f Ly = %.1f Wsc = %.1f Wj = %.1f nodx = %.1f nody = %.1f alpha = %.1f delta = %.2f V_sc = %.1f phi = %.3f mu_i = %.1f mu_f = %.1f.npy" % (dirS, Lx*.1, Ly*.1, SC_width, Junc_width, Nod_widthx,  Nod_widthy, alpha, delta, V0, phi, mu[0], mu[-1]), gap)
    np.save("%s/gx Lx = %.1f Ly = %.1f Wsc = %.1f Wj = %.1f nodx = %.1f nody = %.1f alpha = %.1f delta = %.2f V_sc = %.1f phi = %.3f mu_i = %.1f mu_f = %.1f.npy" % (dirS, Lx*.1, Ly*.1, SC_width, Junc_width, Nod_widthx,  Nod_widthy, alpha, delta, V0, phi, mu[0], mu[-1]), gx)
    np.save("%s/mu Lx = %.1f Ly = %.1f Wsc = %.1f Wj = %.1f nodx = %.1f nody = %.1f alpha = %.1f delta = %.2f V_sc = %.1f phi = %.3f mu_i = %.1f mu_f = %.1f.npy" % (dirS, Lx*.1, Ly*.1, SC_width, Junc_width, Nod_widthx,  Nod_widthy, alpha, delta, V0, phi, mu[0], mu[-1]), mu)
    gc.collect()

    sys.exit()
else:
    gap = np.load("%s/gap_data Lx = %.1f Ly = %.1f Wsc = %.1f Wj = %.1f nodx = %.1f nody = %.1f alpha = %.1f delta = %.2f V_sc = %.1f phi = %.3f mu_i = %.1f mu_f = %.1f.npy" % (dirS, Lx*.1, Ly*.1, SC_width, Junc_width, Nod_widthx,  Nod_widthy, alpha, delta, V0, phi, mu[0], mu[-1]))
    #gap_k0 = np.load("%s/gap_data_k0 Lx = %.1f Ly = %.1f Wsc = %.1f Wj = %.1f nodx = %.1f nody = %.1f alpha = %.1f delta = %.2f V_sc = %.1f phi = %.3f mu_i = %.1f mu_f = %.1f.npy" % (dirS, Lx*.1, Ly*.1, SC_width, Junc_width, Nod_widthx,  Nod_widthy, alpha, delta, V0, phi, mu[0], mu[-1]))
    gx = np.load("%s/gx Lx = %.1f Ly = %.1f Wsc = %.1f Wj = %.1f nodx = %.1f nody = %.1f alpha = %.1f delta = %.2f V_sc = %.1f phi = %.3f mu_i = %.1f mu_f = %.1f.npy" % (dirS, Lx*.1, Ly*.1, SC_width, Junc_width, Nod_widthx,  Nod_widthy, alpha, delta, V0, phi, mu[0], mu[-1]))
    mu =  np.load("%s/mu Lx = %.1f Ly = %.1f Wsc = %.1f Wj = %.1f nodx = %.1f nody = %.1f alpha = %.1f delta = %.2f V_sc = %.1f phi = %.3f mu_i = %.1f mu_f = %.1f.npy" % (dirS, Lx*.1, Ly*.1, SC_width, Junc_width, Nod_widthx,  Nod_widthy, alpha, delta, V0, phi, mu[0], mu[-1]))

    gap = gap/delta
    print(gap.shape, mu.shape, gx.shape)
    plt.contourf(gx, mu, gap, 100, vmin = min(gap.flatten()), vmax = max(gap.flatten()), cmap = 'magma')
    cbar = plt.colorbar()
    cbar.set_label(r'$E_{gap}/\Delta$')

    plt.xlabel(r'$\Gamma_x$ (meV)')
    plt.ylabel(r'$\mu$ (meV)')

    plt.xlim(gi, gf)

    title = r"$L_x =$ {} nm, $L_y =$ {} nm, SC width = {} nm, $W_j =$ {} nm, $nodule_x = ${} nm, $nodule_y = ${} nm, $\alpha = $ {} meV*A, $\phi =$ {} ".format(Lx*.1, Ly*.1, SC_width, Junc_width, Nod_widthx, Nod_widthy, alpha, phi)
    plt.title(title, loc = 'center', wrap = True, fontsize = 8)
    plt.savefig('juncwidth = {} SCwidth = {} V0 = {} nodwidthx = {} nodwidthy = {} Delta = {} Alpha = {} phi = {}.png'.format(Junc_width, SC_width, V0, Nod_widthx, Nod_widthy, delta, alpha, phi))
    plt.show()

    sys.exit()

##############################################################
