import sys
import time
import os
dir = os.getcwd()
import numpy as np
import gc
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as patches
import scipy.sparse as sparse
import scipy.linalg as LA
import scipy.sparse.linalg as spLA

import majoranaJJ.operators.sparse.qmsops as spop #sparse operators
import majoranaJJ.lattice.nbrs as nb #neighbor arrays
import majoranaJJ.lattice.shapes as shps #lattice shapes
import majoranaJJ.modules.plots as plots #plotting functions

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

nod_bool = True
if cutx == 0 and cuty == 0:
    nod_bool = False

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
NN = nb.NN_Arr(coor) #neighbor array
NNb = nb.Bound_Arr(coor) #boundary array
lat_size = coor.shape[0]

Lx = (max(coor[:, 0]) - min(coor[:, 0]) + 1)*ax #Unit cell size in x-direction
Ly = (max(coor[:, 1]) - min(coor[:, 1]) + 1)*ay #Unit cell size in y-direction
print("Lattice size in x-direction", Lx*.1, "(nm)")
print("Lattice size in y-direction", Ly*.1, "(nm)")
###################################################
#Hamiltonian Parameters
alpha = 100 #Spin-Orbit Coupling constant: [meV*A]
gx = 0 #parallel to junction: [meV]
gz = 0 #normal to plane of junction: [meV]
phi_steps = 0
phi = np.pi#np.linspace(0, np.pi, phi_steps) #SC phase difference
delta = 1.0 #Superconducting Gap: [meV]
V0 = 50 #Amplitude of potential: [meV]
V = V_BL(coor, Wj = Wj, cutx=cutx, cuty=cuty, V0 = V0)

k = 164 #This is the number of eigenvalues and eigenvectors you want
mu_steps = 1000 #Number of mu values that are evaluated
mu_i = 50
mu_f = 100
mu = np.linspace(mu_i, mu_f, mu_steps)  #Chemical Potential: [meV]
#####################################
dirS = 'e_mu_data'
if not os.path.exists(dirS):
    os.makedirs(dirS)
try:
    PLOT = str(sys.argv[1])
except:
    PLOT = 'F'
if PLOT != 'P':
    start = time.perf_counter()
    bands = np.zeros((mu_steps, 2*k))
    H0, Hq, Hqq, DELTA, Hgam = kp.Hq(coor, ax, ay, NN, NNb = NNb, Wj = Wj, cutx = cutx, cuty = cuty, V = V, mu = mu[0], alpha = alpha, delta = delta, phi = phi, periodicX = True)

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

    mu_cp = mu[0]
    for i in range(mu_steps):
        print(mu_steps - i)
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

        H = kp.HBDG_LE(H0_DB, Hq_DB, Hqq_DB, DELTA_DB, Hgam_DB, MU, mu[i] - mu_cp, q = 0, gx = 0)
        eigs_DB, U_DB = LA.eigh(H)
        idx_sort = np.argsort(eigs_DB)
        eigs_DB = eigs_DB[idx_sort]

        bands[i, :] = eigs_DB

    np.save("%s/bands Lx = %.1f Ly = %.1f Wsc = %.1f Wj = %.1f nodx = %.1f nody = %.1f alpha = %.1f delta = %.2f V_sc = %.1f  mu_i = %.1f mu_f = %.1f.npy" % (dirS, Lx*.1, Ly*.1, SC_width, Junc_width, Nod_widthx,  Nod_widthy, alpha, delta, V0, mu_i, mu_f), bands)
    np.save("%s/mu Lx = %.1f Ly = %.1f Wsc = %.1f Wj = %.1f nodx = %.1f nody = %.1f alpha = %.1f delta = %.2f V_sc = %.1f  mu_i = %.1f mu_f = %.1f.npy" % (dirS, Lx*.1, Ly*.1, SC_width, Junc_width, Nod_widthx,  Nod_widthy, alpha, delta, V0, mu_i, mu_f), mu)
    end = time.perf_counter()
    print("time for execution: ", end - start)
else:
    bands = np.load("%s/bands Lx = %.1f Ly = %.1f Wsc = %.1f Wj = %.1f nodx = %.1f nody = %.1f alpha = %.1f delta = %.2f V_sc = %.1f  mu_i = %.1f mu_f = %.1f.npy" % (dirS, Lx*.1, Ly*.1, SC_width, Junc_width, Nod_widthx,  Nod_widthy, alpha, delta, V0, mu_i, mu_f))
    mu = np.load("%s/mu Lx = %.1f Ly = %.1f Wsc = %.1f Wj = %.1f nodx = %.1f nody = %.1f alpha = %.1f delta = %.2f V_sc = %.1f  mu_i = %.1f mu_f = %.1f.npy" % (dirS, Lx*.1, Ly*.1, SC_width, Junc_width, Nod_widthx,  Nod_widthy, alpha, delta, V0, mu_i, mu_f))

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.patch.set_facecolor('black')
    print(mu.shape, bands.shape)
    for i in range(bands.shape[1]):
        #print(bands[:, i])
        ax.plot(mu, bands[:, i])

    ax.set_xlabel(r"$\mu$ (meV)")
    ax.set_ylabel("E (meV)")
    ax.set_title(r"Lx = {} nm, Ly = {} nm, $\Delta$ = {} meV, $\alpha$ = {} meV A, $W_sc$ = {} nm, $W_J$ = {} nm, $Nodule_x$ = {} nm, $Nodule_y$ = {} nm".format(Lx*.1, Ly*.1, delta, alpha, SC_width, Junc_width, Nod_widthx, Nod_widthy), loc = 'center', wrap=True)
    ax.set_ylim(-1.5, 1.5)
    plt.savefig("e_mu_phase.png", bbox_inches="tight")
    plt.show()
