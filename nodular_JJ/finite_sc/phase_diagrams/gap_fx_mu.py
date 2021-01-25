import sys
import time
import os
import gc

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import majoranaJJ.lattice.nbrs as nb #neighbor arrays
import majoranaJJ.lattice.shapes as shps #lattice shapes
import majoranaJJ.modules.finders as fndrs
import majoranaJJ.modules.checkers as check
import majoranaJJ.modules.plots as plots
###################################################
#Defining System
Nx = 10 #Number of lattice sites along x-direction
Ny = 300 #Number of lattice sites along y-direction
ax = 50 #lattice spacing in x-direction: [A]
ay = 50 #lattice spacing in y-direction: [A]
Wj = 40 #Junction region
cutx = 4 #width of nodule
cuty = 10 #height of nodule
Nx, Ny, cutx, cuty, Wj = check.junction_geometry_check(Nx, Ny, cutx, cuty, Wj)
print("Nx = {}, Ny = {}, cutx = {}, cuty = {}, Wj = {}".format(Nx, Ny, cutx, cuty, Wj))

Nod_widthx = cutx*ax*.1 #nm
Nod_widthy = cuty*ay*.1 #nm
Junc_width = Wj*ay*.10 #nm
SC_width = ((Ny - Wj)*ay*.10)/2 #nm
print("Nodule Width in x-direction = ", Nod_widthx, "(nm)")
print("Nodule Width in y-direction = ", Nod_widthy, "(nm)")
print("Junction Width = ", Junc_width, "(nm)")
print("Supercondicting Lead Width = ", SC_width, "(nm)")
###################################################coor = shps.square(Nx, Ny) #square lattice
coor = shps.square(Nx, Ny) #square lattice
NN = nb.NN_sqr(coor)
NNb = nb.Bound_Arr(coor)
Lx = (max(coor[:, 0]) - min(coor[:, 0]) + 1)*ax #Unit cell size in x-direction
Ly = (max(coor[:, 1]) - min(coor[:, 1]) + 1)*ay #Unit cell size in y-direction
###################################################
#Defining Hamiltonian parameters
alpha = 200 #Spin-Orbit Coupling constant: [meV*A]
gx = 1.0 #meV
phi = np.pi #SC phase difference
delta = 1 #Superconducting Gap: [meV]
Vsc = 0 #Amplitude of potential in SC region: [meV]
Vj = -5 #Amplitude of potential in junction region: [meV]

mu_i = 2.6
mu_f = 2.8
res = 0.05
delta_mu = mu_f - mu_i
steps = int(delta_mu/(res))
mu = np.linspace(mu_i, mu_f, steps) #Chemical Potential: [meV]
dmu = 0*-.010224
###################################################
dirS = 'gap_data'
if not os.path.exists(dirS):
    os.makedirs(dirS)
try:
    PLOT = str(sys.argv[1])
except:
    PLOT = 'F'
if PLOT != 'P':
    GAP = []
    QMIN = []
    for i in range(mu.shape[0]):
        print(mu.shape[0]-i)
        gap, q_minimum = fndrs.gap_finder(coor, NN, NNb, ax, ay, mu[i], gx, Wj=Wj, cutx=cutx, cuty=cuty, Vj=Vj, alpha=alpha, delta=delta, phi=phi, steps_targ=50000)
        print("gap", gap)
        GAP.append(gap)
        QMIN.append(q_minimum)

    np.save("%s/gap_datafxmu Lx = %.1f Ly = %.1f Wsc = %.1f Wj = %.1f nodx = %.1f nody = %.1f alpha = %.1f delta = %.2f V_sc = %.1f phi = %.3f mu_i = %.1f mu_f = %.1f gx = %.1f.npy" % (dirS, Lx*.1, Ly*.1, SC_width, Junc_width, Nod_widthx,  Nod_widthy, alpha, delta, Vsc, phi, mu[0], mu[-1], gx), GAP)
    np.save("%s/qminfxmu Lx = %.1f Ly = %.1f Wsc = %.1f Wj = %.1f nodx = %.1f nody = %.1f alpha = %.1f delta = %.2f V_sc = %.1f phi = %.3f mu_i = %.1f mu_f = %.1f gx = %.1f.npy" % (dirS, Lx*.1, Ly*.1, SC_width, Junc_width, Nod_widthx,  Nod_widthy, alpha, delta, Vsc, phi, mu[0], mu[-1], gx), QMIN)
    gc.collect()
    sys.exit()
else:
    GAP = np.load("%s/gap_datafxmu Lx = %.1f Ly = %.1f Wsc = %.1f Wj = %.1f nodx = %.1f nody = %.1f alpha = %.1f delta = %.2f V_sc = %.1f phi = %.3f mu_i = %.1f mu_f = %.1f gx = %.1f.npy" % (dirS, Lx*.1, Ly*.1, SC_width, Junc_width, Nod_widthx,  Nod_widthy, alpha, delta, Vsc, phi, mu[0], mu[-1], gx))
    q_minima = np.load("%s/qminfxmu Lx = %.1f Ly = %.1f Wsc = %.1f Wj = %.1f nodx = %.1f nody = %.1f alpha = %.1f delta = %.2f V_sc = %.1f phi = %.3f mu_i = %.1f mu_f = %.1f gx = %.1f.npy" % (dirS, Lx*.1, Ly*.1, SC_width, Junc_width, Nod_widthx,  Nod_widthy, alpha, delta, Vsc, phi, mu[0], mu[-1], gx))
    mu = np.linspace(mu_i, mu_f, GAP.shape[0])
    GAP = GAP/delta

    plt.plot(mu, GAP)
    plt.grid()
    #plt.ylim(0.06, 0.20)
    #plt.xlim(7.9, 10)
    plt.xlabel(r'$\mu$')
    plt.ylabel(r'gap/$\Delta$')
    title = r"$E_Z$ = %.2f meV $W_j$ = %.1f nm, $nodule_x$ = %.1f nm, $nodule_y$ = %.1f nm, $V_j$ = %.1f meV, $\phi$ = %.2f " % (gx, Junc_width, Nod_widthx, Nod_widthy, Vj, phi)
    plt.title(title, wrap=True)
    plt.show()
    sys.exit()
