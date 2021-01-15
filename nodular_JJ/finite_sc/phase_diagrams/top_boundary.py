import sys
import os
import gc

import numpy as np
import matplotlib.pyplot as plt

import majoranaJJ.lattice.nbrs as nb #neighbor arrays
import majoranaJJ.lattice.shapes as shps #lattice shapes
import majoranaJJ.modules.plots as plots #plotting functions
import majoranaJJ.modules.finders as fndrs
import majoranaJJ.operators.sparse_operators as spop #sparse operators
import majoranaJJ.modules.checkers as check
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

Lx = (max(coor[:, 0]) - min(coor[:, 0]) + 1)*ax #Unit cell size in x-direction
Ly = (max(coor[:, 1]) - min(coor[:, 1]) + 1)*ay #Unit cell size in y-direction
###################################################
#Defining Hamiltonian parameters
alpha = 200 #Spin-Orbit Coupling constant: [meV*A]
phi = np.pi #SC phase difference
delta = 1 #Superconducting Gap: [meV]
Vsc = 0 #SC potential: [meV]
Vj = -5 #Junction potential: [meV]

mu_i = -5
mu_f = 0
res = 0.05
delta_mu = mu_f - mu_i
mu_steps = int(delta_mu/res)
mu = np.linspace(mu_i, mu_f, mu_steps) #Chemical Potential: [meV]
#dmu = -0.010347

gi = 0
gf = 5.0
num_bound = 6
boundary = np.zeros((mu_steps, num_bound))
###################################################
#phase diagram mu vs gam
dirS = 'boundary_data'
if not os.path.exists(dirS):
    os.makedirs(dirS)
try:
    PLOT = str(sys.argv[1])
except:
    PLOT = 'F'
if PLOT != 'P':
    for i in range(mu_steps):
        print(mu_steps-i, "| mu =", mu[i])
        gx = fndrs.local_min_gam_finder(coor, NN, NNb, ax, ay, mu[i], gi, gf, Wj=Wj, cutx=cutx, cuty=cuty, Vj=Vj, alpha=alpha, delta=delta, phi=phi, k=200)
        for j in range(num_bound):
            if j >= gx.size:
                boundary[i, j] = None
            else:
                boundary[i, j] = gx[j]
    boundary = np.array(boundary)

    np.save("%s/boundary Lx = %.1f Ly = %.1f Wsc = %.1f Wj = %.1f nodx = %.1f nody = %.1f Vj = %.1f Vsc = %.1f alpha = %.1f delta = %.2f phi = %.3f mu_i = %.1f mu_f=%.1f.npy" % (dirS, Lx*.1, Ly*.1, SC_width, Junc_width, Nod_widthx,  Nod_widthy, Vj, Vsc, alpha, delta, phi, mu_i, mu_f), boundary)
    np.save("%s/mu Lx = %.1f Ly = %.1f Wsc = %.1f Wj = %.1f nodx = %.1f nody = %.1f Vj = %.1f Vsc = %.1f alpha = %.1f delta = %.2f phi = %.3f mu_i = %.1f mu_f=%.1f.npy" % (dirS, Lx*.1, Ly*.1, SC_width, Junc_width, Nod_widthx,  Nod_widthy, Vj, Vsc, alpha, delta, phi, mu_i, mu_f), mu)
    gc.collect()

    sys.exit()
else:
    boundary = np.load("%s/boundary Lx = %.1f Ly = %.1f Wsc = %.1f Wj = %.1f nodx = %.1f nody = %.1f Vj = %.1f Vsc = %.1f alpha = %.1f delta = %.2f phi = %.3f mu_i = %.1f mu_f=%.1f.npy" % (dirS, Lx*.1, Ly*.1, SC_width, Junc_width, Nod_widthx,  Nod_widthy, Vj, Vsc, alpha, delta, phi, mu_i, mu_f))
    mu = np.load("%s/mu Lx = %.1f Ly = %.1f Wsc = %.1f Wj = %.1f nodx = %.1f nody = %.1f Vj = %.1f Vsc = %.1f alpha = %.1f delta = %.2f phi = %.3f mu_i = %.1f mu_f=%.1f.npy" % (dirS, Lx*.1, Ly*.1, SC_width, Junc_width, Nod_widthx,  Nod_widthy, Vj, Vsc, alpha, delta, phi, mu_i, mu_f))
    print(boundary[:, 0])

    for i in range(boundary.shape[1]):
        plt.scatter(boundary[:, i], mu, c='r', s = 2)

    plt.xlabel(r'$E_z$ (meV)')
    plt.ylabel(r'$\mu$ (meV)')
    plt.xlim(gi, gf)
    title = r"$L_x$ = %.1f nm, $L_y$ = %.1f nm, $W_{sc}$ = %.1f nm, $W_j$ = %.1f nm, $nodule_x$ = %.1f nm, $nodule_y$ = %.1f nm, $V_j$ = %.1f meV, $V_{SC}$ = %.1f meV, $\phi$ = %.2f " % (Lx*.1, Ly*.1, SC_width, Junc_width, Nod_widthx, Nod_widthy, Vj, Vsc, phi)
    plt.grid()
    plt.title(title, loc = 'center', wrap = True)
    plt.subplots_adjust(top=0.85)
    plt.savefig('juncwidth = {} SCwidth = {} nodwidthx = {} nodwidthy = {} phi = {} Vj = {} Vsc = {}.png'.format(Junc_width, SC_width, Nod_widthx, Nod_widthy, delta, alpha, phi, Vj, Vsc))
    plt.show()

    sys.exit()

##############################################################

#state plot
MU = 2
GX = 0.75

H = spop.HBDG(coor, ax, ay, NN, NNb = NNb, Wj = Wj, V = V, mu = MU, gammax = GX, alpha = alpha, delta = delta, phi = np.pi, qx= 0, periodicX = True, periodicY = False)

eigs, states = spLA.eigsh(H, k=8, sigma=0, which='LM')
idx_sort = np.argsort(eigs)
print(eigs[idx_sort])
plots.state_cmap(coor, eigs, states, n=4, savenm='prob_density_nodule_n=4.png')
plots.state_cmap(coor, eigs, states, n=5, savenm='prob_density_nodule_n=5.png')
plots.state_cmap(coor, eigs, states, n=6, savenm='prob_density_nodule_n=6.png')
