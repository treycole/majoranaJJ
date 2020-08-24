import sys
import os

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

from majoranaJJ.operators.sparse.potentials import Vjj #potential JJ
dir = os.getcwd()
###################################################
#Defining System
Nx = 11 #Number of lattice sites along x-direction
Ny = 408 #Number of lattice sites along y-direction
ax = 50 #lattice spacing in x-direction: [A]
ay = 50 #lattice spacing in y-direction: [A]
Wj = 11 #Junction region
cutx = 3 #width of nodule
cuty = 3 #height of nodule

Junc_width = Wj*ay*.1 #nm
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
phi_steps = 5
phi = np.linspace(0, np.pi, phi_steps) #SC phase difference
delta = 1.0 #Superconducting Gap: [meV]
Vj = -50 #Amplitude of potential: [meV]
V = Vjj(coor, Wj = Wj, Vsc = 0, Vj = Vj, cutx = cutx, cuty = cuty)
#####################################
k = 44 #This is the number of eigenvalues and eigenvectors you want
mu_steps = 500 #Number of kx values that are evaluated
mu_i = 50
mu_f = 100
mu = np.linspace(mu_i, mu_f, mu_steps)  #Chemical Potential: [meV]
bands = np.zeros((phi_steps, mu_steps, k))
cmap = cm.get_cmap('Oranges')
lin = np.linspace(0, 1, phi_steps)

def get_LE_basis(coor, ax, ay, NN, NNb, Wj, cutx, cuty, V, mu, gz, alpha, delta, phi):
    H0 = spop.HBDG(coor, ax, ay, NN, NNb=NNb, Wj=Wj, cutx=cutx, cuty=cuty, V=V, mu=mu, gammaz=gz, alpha=alpha, delta=delta, phi=phi, qx=1e-4*(np.pi/Lx), periodicX=True) #gives low energy basis

    eigs_0, vecs_0 = spLA.eigsh(H0, k=k, sigma=0, which='LM')
    vecs_0_hc = np.conjugate(np.transpose(vecs_0)) #hermitian conjugate

    H_M0 =  spop.HBDG(coor, ax, ay, NN, NNb = NNb, Wj = Wj, cutx = cutx, cuty = cuty, V = V, mu = 0, alpha = alpha, delta = delta, phi = phi, qx = 0, periodicX = True)

    H_M1 = spop.HBDG(coor, ax, ay, NN, NNb = NNb, Wj = Wj, cutx = cutx, cuty = cuty, V = V, mu = 1, alpha = alpha, delta = delta, phi = phi, qx = 0, periodicX =True)

    HM = H_M1 - H_M0

    HM0_DB = np.dot(vecs_0_hc, H_M0.dot(vecs_0))
    HM_DB = np.dot(vecs_0_hc, HM.dot(vecs_0))
    return HM0_DB, HM_DB

dirS = 'e_mu_data'
if not os.path.exists(dirS):
    os.makedirs(dirS)
try:
    PLOT = str(sys.argv[1])
except:
    PLOT = 'F'
if PLOT != 'P':
    for i in range(phi_steps):
        print(phi_steps - i)
        phi_num = phi_steps - i
        #HM0_DB, HM_DB = get_LE_basis(coor, ax, ay, NN, NNb, Wj, cutx, cuty, V, mu_DB, 0, alpha, delta, phi[i])
        for j in range(mu_steps):
            print(phi_num, mu_steps - j)
            #if (mu[j] - mu_DB) >= 1:
            #    mu_DB = mu[j]
            #    HM0_DB, HM_DB = get_LE_basis(coor, ax, ay, NN, NNb, Wj, cutx, cuty, V, mu_DB, 0, alpha, delta, phi[i])
            #H = HM0_DB + mu[j]*HM_DB
            #eigs, U = LA.eigh(H)
            H = spop.HBDG(coor, ax, ay, NN, NNb=NNb, Wj=Wj, cutx=cutx, cuty=cuty, V=V, mu=mu[j], alpha=alpha, delta=delta, phi=phi[i], qx=0, periodicX=True)
            eigs, vecs = spLA.eigsh(H, k=k, sigma=0, which='LM')
            idx_sort = np.argsort(eigs)
            eigs = eigs[idx_sort]

            bands[i, j, :] = eigs

    np.save("%s/bands Lx = %.1f Ly = %.1f Wsc = %.1f Wj = %.1f nodx = %.1f nody = %.1f alpha = %.1f delta = %.2f Vj = %.1f  mu_i = %.1f mu_f = %.1f.npy" % (dirS, Lx*.1, Ly*.1, SC_width, Junc_width, Nod_widthx,  Nod_widthy, alpha, delta, Vj, mu_i, mu_f), bands)
    np.save("%s/mu Lx = %.1f Ly = %.1f Wsc = %.1f Wj = %.1f nodx = %.1f nody = %.1f alpha = %.1f delta = %.2f Vj = %.1f  mu_i = %.1f mu_f = %.1f.npy" % (dirS, Lx*.1, Ly*.1, SC_width, Junc_width, Nod_widthx,  Nod_widthy, alpha, delta, Vj, mu_i, mu_f), mu)
else:
    bands = np.load("%s/bands Lx = %.1f Ly = %.1f Wsc = %.1f Wj = %.1f nodx = %.1f nody = %.1f alpha = %.1f delta = %.2f Vj = %.1f  mu_i = %.1f mu_f = %.1f.npy" % (dirS, Lx*.1, Ly*.1, SC_width, Junc_width, Nod_widthx,  Nod_widthy, alpha, delta, Vj, mu_i, mu_f))
    mu = np.load("%s/mu Lx = %.1f Ly = %.1f Wsc = %.1f Wj = %.1f nodx = %.1f nody = %.1f alpha = %.1f delta = %.2f Vj = %.1f  mu_i = %.1f mu_f = %.1f.npy" % (dirS, Lx*.1, Ly*.1, SC_width, Junc_width, Nod_widthx,  Nod_widthy, alpha, delta, Vj, mu_i, mu_f))

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.patch.set_facecolor('black')
    for i in range(bands.shape[0]):
        for j in range(bands.shape[2]):
            ax.plot(mu, bands[i, :, j], c = cmap(lin[i]), zorder = -i)

    ax.set_xlabel(r"$\mu$ (meV)")
    ax.set_ylabel("E (meV)")
    ax.set_title(r"Lx = %.1f nm, Ly = %.1f nm, $\Delta$ = %.2f meV, $\alpha$ = %.2f meV A, $W_{sc}$ = %.1f nm, $W_J$ = %.1f nm, $Nodule_x$ = %.1f nm, $Nodule_y$ = %.1f nm" % (Lx*.1, Ly*.1, delta, alpha, SC_width, Junc_width, Nod_widthx, Nod_widthy), loc = 'center', wrap=True)
    ax.set_ylim(-1.5, 1.5)
    plt.subplots_adjust(top=0.85)
    plt.savefig("e_mu_phase.png", bbox_inches="tight")
    plt.show()
