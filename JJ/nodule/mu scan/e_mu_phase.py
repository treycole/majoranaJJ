import sys
import numpy as np
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
phi_steps = 6
phi = np.linspace(0, np.pi, phi_steps) #SC phase difference
delta = 1.0 #Superconducting Gap: [meV]
V0 = 50 #Amplitude of potential: [meV]
V = V_BL(coor, Wj = Wj, cutx=cutx, cuty=cuty, V0 = V0)

#####################################

k = 64 #This is the number of eigenvalues and eigenvectors you want
mu_steps = 500 #Number of kx values that are evaluated
mu = np.linspace(80, 90, mu_steps)  #Chemical Potential: [meV]
bands = np.zeros((phi_steps, mu_steps, k))
cmap = cm.get_cmap('Oranges')
lin = np.linspace(0, 1, phi_steps)

def get_LE_basis(coor, ax, ay, NN, NNb, Wj, cutx, cuty, V, mu, gz, alpha, delta, phi):
    H0 = spop.HBDG(coor, ax, ay, NN, NNb=NNb, Wj=Wj, cutx=cutx, cuty=cuty, V=V, mu=mu, gammaz=gz, alpha=alpha, delta=delta, phi=phi, qx=1e-5*(np.pi/Lx), periodicX=True) #gives low energy basis

    eigs_0, vecs_0 = spLA.eigsh(H0, k=k, sigma=0, which='LM')
    vecs_0_hc = np.conjugate(np.transpose(vecs_0)) #hermitian conjugate

    H_M0 =  spop.HBDG(coor, ax, ay, NN, NNb = NNb, Wj = Wj, cutx = cutx, cuty = cuty, V = V, mu = 0, alpha = alpha, delta = delta, phi = phi, qx = 0, periodicX = True)

    H_M1 = spop.HBDG(coor, ax, ay, NN, NNb = NNb, Wj = Wj, cutx = cutx, cuty = cuty, V = V, mu = 1, alpha = alpha, delta = delta, phi = phi, qx = 0, periodicX =True)

    HM = H_M1 - H_M0

    HM0_DB = np.dot(vecs_0_hc, H_M0.dot(vecs_0))
    HM_DB = np.dot(vecs_0_hc, HM.dot(vecs_0))
    return HM0_DB, HM_DB

for i in range(phi_steps):
    print(phi_steps - i)
    phi_num = phi_steps - i
    mu_DB = mu[0]
    HM0_DB, HM_DB = get_LE_basis(coor, ax, ay, NN, NNb, Wj, cutx, cuty, V, mu_DB, 0, alpha, delta, phi[i])
    for j in range(mu_steps):
        print(phi_num, (j/mu_steps)*100, "%")
        if (mu[j] - mu_DB) >= 0.5:
            mu_DB = mu[j]
            HM0_DB, HM_DB = get_LE_basis(coor, ax, ay, NN, NNb, Wj, cutx, cuty, V, mu_DB, 1e-5, alpha, delta, phi[i])
        H = HM0_DB + mu[j]*HM_DB
        eigs, U = LA.eigh(H)
        #eigs, vecs = spLA.eigsh(H, k=k, sigma=0, which='LM')
        #idx_sort = np.argsort(eigs)
        #eigs = eigs[idx_sort]

        bands[i, j, :] = eigs

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.patch.set_facecolor('black')
for i in range(bands.shape[0]):
    for j in range(bands.shape[2]):
        ax.plot(mu, bands[i, :, j], c = cmap(lin[i]), zorder = -i)

ax.set_xlabel(r"$\mu$ (meV)")
ax.set_ylabel("E (meV)")
ax.set_title(r"Lx = {} nm, Ly = {} nm, $\Delta$ = {} meV, $\alpha$ = {} meV A, $W_sc$ = {} nm, $W_J$ = {} nm, $Nodule_x$ = {} nm, $Nodule_y$ = {} nm".format(Lx*.1, Ly*.1, delta, alpha, SC_width, Junc_width, Nod_widthx, Nod_widthy), wrap=True)
ax.set_ylim(-1.5, 1.5)
plt.savefig("e_mu_phase.png")
plt.show()
