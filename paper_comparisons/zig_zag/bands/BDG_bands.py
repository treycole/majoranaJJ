import sys
import os
import gc

import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sparse
import scipy.sparse.linalg as spLA

import majoranaJJ.operators.sparse_operators as spop #sparse operators
import majoranaJJ.lattice.nbrs as nb #neighbor arrays
import majoranaJJ.lattice.shapes as shps #lattice shapes
import majoranaJJ.modules.plots as plots #plotting functions
import majoranaJJ.modules.gamfinder as gamfinder
from majoranaJJ.operators.potentials import Vjj #potential JJ
import majoranaJJ.modules.checkers as check

###################################################
#Defining System
Nx = 130 #Number of lattice sites along x-direction
Ny = 80 #Number of lattice sites along y-direction
ax = 100 #lattice spacing in x-direction: [A]
ay = 100 #lattice spacing in y-direction: [A]
Wj = 20 #Junction region
cutx = 0 #width of nodule
cuty = 0 #height of nodule
Nx, Ny, cutx, cuty, Wj = check.junction_geometry_check(Ny, Nx, Wj, cutx, cuty)

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
phi = [0, np.pi] #SC phase difference
delta = 1 #Superconducting Gap: [meV]
Vsc = 0 #SC potential: [meV]
Vj = 0 #Junction potential: [meV]
V = Vjj(coor, Wj = Wj, Vsc = Vsc, Vj = Vj, cutx = cutx, cuty = cuty)
mu = 10 #meV
B = [0, 1] #T
###################################################
#phase diagram mu vs gam
dirS = 'bands_data'
if not os.path.exists(dirS):
    os.makedirs(dirS)
try:
    PLOT = str(sys.argv[1])
except:
    PLOT = 'F'
if PLOT != 'P':
    k = 22 #This is the number of eigenvalues and eigenvectors you want
    steps = 201 #Number of kx values that are evaluated
    qx = np.linspace(0, np.pi/Lx, steps) #kx in the first Brillouin zone

    bands0 = np.zeros((steps, k))
    bands1 = np.zeros((steps, k))
    for i in range(steps):
        print(steps - i)
        H0 = spop.HBDG(coor, ax, ay, NN, NNb = NNb, Wj = Wj, cutx = cutx, cuty = cuty, V = V, mu = mu, gammax = B[0], alpha = alpha, delta = delta, phi = phi[0], qx = qx[i], Tesla = True, Zeeman_in_SC = True, SOC_in_SC = False)

        H1 = spop.HBDG(coor, ax, ay, NN, NNb = NNb, Wj = Wj, cutx = cutx, cuty = cuty, V = V, mu = mu, gammax = B[1], alpha = alpha, delta = delta, phi = phi[1], qx = qx[i], Tesla = True, Zeeman_in_SC = True, SOC_in_SC = False)

        eigs0, vecs0 = spLA.eigsh(H0, k=k, sigma=0, which='LM')
        eigs1, vecs1 = spLA.eigsh(H1, k=k, sigma=0, which='LM')
        idx_sort0 = np.argsort(eigs0)
        idx_sort1 = np.argsort(eigs1)
        eigs0 = eigs0[idx_sort0]
        eigs1 = eigs1[idx_sort1]

        bands0[i, :] = eigs0
        bands1[i, :] = eigs1

    np.save("%s/bands0 SOC and Zeman in SC Lx = %.1f Ly = %.1f Wsc = %.1f Wj = %.1f nodx = %.1f nody = %.1f Vj = %.1f Vsc = %.1f alpha = %.1f delta = %.2f phi = %.3f.npy" % (dirS, Lx*.1, Ly*.1, SC_width, Junc_width, Nod_widthx,  Nod_widthy, Vj, Vsc, alpha, delta, phi[0]), bands0)
    np.save("%s/bands1 SOC and Zeman in SC Lx = %.1f Ly = %.1f Wsc = %.1f Wj = %.1f nodx = %.1f nody = %.1f Vj = %.1f Vsc = %.1f alpha = %.1f delta = %.2f phi = %.3f.npy" % (dirS, Lx*.1, Ly*.1, SC_width, Junc_width, Nod_widthx,  Nod_widthy, Vj, Vsc, alpha, delta, phi[1]), bands1)
    np.save("%s/qx SOC and Zeman in SC Lx = %.1f Ly = %.1f Wsc = %.1f Wj = %.1f nodx = %.1f nody = %.1f Vj = %.1f Vsc = %.1f alpha = %.1f delta = %.2f.npy" % (dirS, Lx*.1, Ly*.1, SC_width, Junc_width, Nod_widthx,  Nod_widthy, Vj, Vsc, alpha, delta), qx)
    gc.collect()

    sys.exit()

else:
    bands0 = np.load("%s/bands0 SOC and Zeman in SC Lx = %.1f Ly = %.1f Wsc = %.1f Wj = %.1f nodx = %.1f nody = %.1f Vj = %.1f Vsc = %.1f alpha = %.1f delta = %.2f phi = %.3f.npy" % (dirS, Lx*.1, Ly*.1, SC_width, Junc_width, Nod_widthx,  Nod_widthy, Vj, Vsc, alpha, delta, phi[0]))
    bands1 = np.load("%s/bands1 SOC and Zeman in SC Lx = %.1f Ly = %.1f Wsc = %.1f Wj = %.1f nodx = %.1f nody = %.1f Vj = %.1f Vsc = %.1f alpha = %.1f delta = %.2f phi = %.3f.npy" % (dirS, Lx*.1, Ly*.1, SC_width, Junc_width, Nod_widthx,  Nod_widthy, Vj, Vsc, alpha, delta, phi[1]))
    qx = np.load("%s/qx SOC and Zeman in SC Lx = %.1f Ly = %.1f Wsc = %.1f Wj = %.1f nodx = %.1f nody = %.1f Vj = %.1f Vsc = %.1f alpha = %.1f delta = %.2f.npy" % (dirS, Lx*.1, Ly*.1, SC_width, Junc_width, Nod_widthx,  Nod_widthy, Vj, Vsc, alpha, delta))

    for i in range(bands0.shape[1]):
        plt.plot(qx, bands0[:, i], c ='mediumblue', linestyle = 'solid')
        plt.plot(-qx, bands0[:, i], c ='mediumblue', linestyle = 'solid')
        plt.plot(qx, bands1[:, i], c ='orange', linestyle = 'solid')
        plt.plot(-qx, bands1[:, i], c ='orange', linestyle = 'solid')

        plt.scatter(qx, bands0[:, i], c ='mediumblue', s = 10)
        plt.scatter(-qx, bands0[:, i], c ='mediumblue', s = 10)
        plt.scatter(qx, bands1[:, i], c ='orange', s = 10)
        plt.scatter(-qx, bands1[:, i], c ='orange', s = 10)

    plt.plot(qx, 0*qx, c = 'k', linestyle='solid', lw=1)
    plt.plot(-qx, 0*qx, c = 'k', linestyle='solid', lw=1)
    plt.xticks(np.linspace(-max(qx), max(qx), 5), (r'$-\pi/Lx$', r'$-\pi/2Lx$', '0', r'$\pi/2Lx$', r'$\pi/Lx$'))
    plt.xlabel(r'$k_x$ (1/A)')
    plt.ylabel('Energy (meV)')
    plt.ylim(-0.325, 0.325)
    plt.title('Band Structures', wrap = True)

    plt.show()
