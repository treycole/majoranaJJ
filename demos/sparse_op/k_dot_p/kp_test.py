import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sparse
import scipy.linalg as LA
import scipy.sparse.linalg as spLA

import majoranaJJ.operators.sparse.qmsops as spop #sparse operators
import majoranaJJ.lattice.nbrs as nb #neighbor arrays
import majoranaJJ.lattice.shapes as shps #lattice shapes
import majoranaJJ.modules.plots as plots #plotting functions
import majoranaJJ.operators.sparse.k_dot_p as kp
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
print("Lattice Size: ", lat_size)

Lx = (max(coor[:, 0]) - min(coor[:, 0]) + 1)*ax #Unit cell size in x-direction
Ly = (max(coor[:, 1]) - min(coor[:, 1]) + 1)*ay #Unit cell size in y-direction

###################################################
#Hamiltonian Parameters
alpha = 100 #Spin-Orbit Coupling constant: [meV*A]
gx = 0.2 #parallel to junction: [meV]
gz = 0 #normal to plane of junction: [meV]
phi = 0*np.pi #SC phase difference
delta = 1.0 #Superconducting Gap: [meV]
V0 = 50 #Amplitude of potential: [meV]
V = V_BL(coor, Wj = Wj, cutx=cutx, cuty=cuty, V0 = V0)
mu = 55 #Chemical Potential: [meV]

#####################################

k = 500 #This is the number of eigenvalues and eigenvectors you want
k_full = 12
steps = 101  #Number of kx values that are evaluated
qx = np.linspace(0, np.pi/Lx, steps) #kx in the first Brillouin zone

H = spop.H0(coor, ax, ay, NN, NNb=NNb,alpha=alpha, V=V, gammax=1e-4, mu=mu, qx=0, periodicX=True)

eigs_0, vecs_0 = spLA.eigsh(H, k=k, sigma=0, which='LM')
vecs_0_hc = np.conjugate(np.transpose(vecs_0)) #hermitian conjugate
vecs_0_c = np.conjugate(vecs_0)

H0, Hq, Hqq, DELTA, Hgam, MU = kp.Hq(coor, ax, ay, NN, NNb = NNb, Wj = Wj, cutx = cutx, cuty = cuty, V = V, mu = mu, alpha = alpha, delta = delta, phi = phi, periodicX = True)
H0_DB = np.dot(vecs_0_hc, H0.dot(vecs_0))
Hq_DB = np.dot(vecs_0_hc, Hq.dot(vecs_0))
Hqq_DB = np.dot(vecs_0_hc, Hqq.dot(vecs_0))
DELTA_DB = np.dot(vecs_0_hc, DELTA.dot(vecs_0))
Hgam_DB = np.dot(vecs_0_hc, Hgam.dot(vecs_0))
MU_DB = np.dot(vecs_0_hc, MU.dot(vecs_0))

start = time.perf_counter()
for i in range(steps):
    print(steps - i)
    #eigs = spop.EBDG(coor, ax, ay, NN, NNb=NNb, Wj=Wj, cutx=cutx, cuty=cuty, alpha=alpha, delta=delta, phi = phi, V=V, gammax=gx, mu=mu, qx=qx[i], periodicX=True, k=k_full)

    H = kp.HBDG_LE(H0_DB, Hq_DB, Hqq_DB, DELTA_DB, Hgam_DB, MU_DB, q = qx[i], gx = gx)

    eigs_kp, U_kp = LA.eigh(H)
    if i == 0:
        bands_kp = np.zeros((steps, eigs_kp.shape[0]))
        #bands = np.zeros((steps, eigs.shape[0]))

    #bands[i, :] = eigs
    bands_kp[i, :] = eigs_kp
end = time.perf_counter()
print("Time: ", end-start)

if nod_bool:
    x = "Present"
else:
    x = "Not Present"

for i in range(bands_kp.shape[1]):
    plt.plot(qx, bands_kp[:, i], c ='red', linestyle = '--')
    plt.plot(-qx, bands_kp[:, i], c ='red', linestyle = '--')
#for i in range(bands.shape[1]):
    #plt.plot(qx, bands[:, i], c ='mediumblue', linestyle = 'solid')
    #plt.plot(-qx, bands[:, i], c ='mediumblue', linestyle = 'solid')
plt.plot(qx, 0*qx, c = 'k', linestyle='solid', lw=1)
plt.plot(-qx, 0*qx, c = 'k', linestyle='solid', lw=1)
#plt.xticks(np.linspace(min(k), max(k), 3), ('-π/Lx', '0', 'π/Lx'))
plt.xlabel('kx (1/A)')
plt.ylabel('Energy (meV)')
plt.ylim(-2*delta, 2*delta)
plt.savefig('k_dot_p_bands.png')
plt.show()
#####################################
