import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sparse
import scipy.linalg as LA
import scipy.sparse.linalg as spLA
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm

import majoranaJJ.operators.sparse.qmsops as spop #sparse operators
import majoranaJJ.lattice.nbrs as nb #neighbor arrays
import majoranaJJ.lattice.shapes as shps #lattice shapes
import majoranaJJ.modules.plots as plots #plotting functions

from majoranaJJ.operators.potentials.barrier_leads import V_BL
###################################################

#Defining System
Nx = 12 #Number of lattice sites along x-direction
Ny = 20 #Number of lattice sites along y-direction
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
N = coor.shape[0]
print("Lattice Size: ", N)

Lx = (max(coor[:, 0]) - min(coor[:, 0]) + 1)*ax #Unit cell size in x-direction
Ly = (max(coor[:, 1]) - min(coor[:, 1]) + 1)*ay #Unit cell size in y-direction

###################################################
#Hamiltonian Parameters
alpha = 100 #Spin-Orbit Coupling constant: [meV*A]
gx = 0 #parallel to junction: [meV]
gz = 0 #normal to plane of junction: [meV]
phi = 0*np.pi #SC phase difference
delta = 1.0 #Superconducting Gap: [meV]
V0 = 50 #Amplitude of potential: [meV]
V = V_BL(coor, Wj = Wj, cutx=cutx, cuty=cuty, V0 = V0)
mu = 0  #Chemical Potential: [meV]

#####################################

num_eigs = 100 #number of eigenvalues and eigenvectors
steps = 101 #Number of kx values that are evaluated

qx = np.linspace(0, np.pi/Lx, steps) #kx in the first Brillouin zone
bands = np.zeros((steps, num_eigs))
wt = np.zeros((steps, num_eigs))

for i in range(steps):
    print(steps - i)

    H = spop.H0(coor, ax, ay, NN, NNb=NNb, alpha=alpha, V=V, gammax=gx, gammaz=gz, mu=mu, qx=qx[i], periodicX=True)

    eigs, vecs = spLA.eigsh(H, k=num_eigs, sigma=0, which='LM')
    idx_sort = np.argsort(eigs)
    eigs = eigs[idx_sort]
    vecs = vecs[:, idx_sort] #idx sort size = k

    bands[i, :] = eigs
    num_div = int(vecs.shape[0]/N)

    for j in range(vecs.shape[1]):
        probdens = np.square(abs(vecs[:, j]))
        PD_UP = probdens[0: int(probdens.shape[0]/2)]
        PD_DOWN = probdens[int(probdens.shape[0]/2):]
        for k in range(N):
            if V[k, k] != 0:
                wt[i, j] += PD_UP[k] + PD_DOWN[k]

fig = plt.figure()
norm = plt.Normalize(0, 1)
ax = fig.add_subplot(1,1,1)
for i in range(0, bands.shape[1]):
    ax.plot(qx, bands[:, i], linewidth=1.25, c='w', zorder = -10)
for i in range(0, bands.shape[1]):
    print(bands.shape[1] - i)
    points = np.array([qx, bands[:, i]]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap='seismic', norm=norm)
    lc.set_array(wt[:, i])
    lc.set_linewidth(1.0)
    line = ax.add_collection(lc)

ax.patch.set_facecolor('black')
axcb = fig.colorbar(lc)
ax.set_xlim(qx[0], qx[-1])
ax.set_xlabel('kx (1/A)')
ax.set_ylabel('Energy (meV)')
plt.savefig('weighted_bands.png')
plt.show()
sys.exit()

#####################################

VV = sparse.bmat([[None, V], [-V, None]], format='csc', dtype='complex')
plots.junction(coor, VV, title = 'Potential Profile', savenm = 'potential_profile.jpg')

k = 48
H = spop.HBDG(coor, ax, ay, NN, NNb=NNb, Wj=Wj, cutx=cutx, cuty=cuty, alpha=alpha, delta=delta, phi = phi, V=V, gammax=gx, gammaz=gz, mu=mu, qx=0.00212, periodicX=True, periodicY=False)

eigs, vecs = spLA.eigsh(H, k=k, sigma=0, which='LM')
idx_sort = np.argsort(eigs)
eigs = eigs[idx_sort]
vecs = vecs[:, idx_sort]
print(eigs)

n = int(k/2)
plots.state_cmap(coor, eigs, vecs, n = int(k/2), title = r'$|\psi|^2$', savenm = 'State_k={}.png'.format(n))

sys.exit()

for i in range(int(k/2), k):
    plots.state_cmap(coor, eigs, vecs, n = i, title = r'$|\psi|^2$', savenm = 'State_k={}.png'.format(i))
