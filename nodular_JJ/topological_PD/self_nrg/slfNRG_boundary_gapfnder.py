import sys
import os
import gc

import numpy as np
import matplotlib.pyplot as plt

import majoranaJJ.modules.plots as plots #plotting functions
import majoranaJJ.modules.gamfinder as gamfinder
import majoranaJJ.modules.self_energy_nodule as slfNRG
###################################################
#Defining System
ax = 50 #lattice spacing in x-direction: [A]
ay = 50 #lattice spacing in y-direction: [A]
Nx = 10 #Number of lattice sites along x-direction
Wj = 2000 #Junction region [A]
cutx = 3 #width of nodule
cuty = 3 #height of nodule
Lx = Nx*ax

Junc_width = Wj*.10 #nm
Nod_widthx = cutx*ax*.1 #nm
Nod_widthy = cuty*ay*.1 #nm
print("Nodule Width in x-direction = ", Nod_widthx, "(nm)")
print("Nodule Width in y-direction = ", Nod_widthy, "(nm)")
print("Junction Width = ", Junc_width, "(nm)")
###################################################
#Defining Hamiltonian parameters
alpha = 200 #Spin-Orbit Coupling constant: [meV*A]
phi = np.pi #SC phase difference
delta = 1 #Superconducting Gap: [meV]
Vj = -5 #junction potential: [meV]

mu_i = -5
mu_f = 15
res = 0.02
delta_mu = mu_f - mu_i
mu_steps = int(delta_mu/res)
mu = np.linspace(mu_i, mu_f, mu_steps) #Chemical Potential: [meV]
#dmu = -0.010347

gi = 0
gf = 2.0
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
        gx = gamfinder.slfNRG(ax, ay, mu[i], gi, gf, Wj=Wj, Lx=Lx, cutx=cutx, cuty=cuty, Vj=Vj, m_eff=0.026, alpha=alpha, delta=delta, phi=phi)
        for j in range(num_bound):
            if j >= gx.size:
                boundary[i, j] = None
            else:
                boundary[i, j] = gx[j]
    boundary = np.array(boundary)

    np.save("%s/boundary Lx = %.1f Wj = %.1f nodx = %.1f nody = %.1f Vj = %.1f alpha = %.1f delta = %.2f phi = %.3f.npy" % (dirS, Lx*.1, Junc_width, Nod_widthx, Nod_widthy, Vj, alpha, delta, phi), boundary)
    np.save("%s/mu Lx = %.1f Wj = %.1f nodx = %.1f nody = %.1f Vj = %.1f alpha = %.1f delta = %.2f phi = %.3f.npy" % (dirS, Lx*.1, Junc_width, Nod_widthx, Nod_widthy, Vj, alpha, delta, phi), mu)
    gc.collect()

    sys.exit()
else:
    boundary = np.load("%s/boundary Lx = %.1f Wj = %.1f nodx = %.1f nody = %.1f Vj = %.1f alpha = %.1f delta = %.2f phi = %.3f.npy" % (dirS, Lx*.1, Junc_width, Nod_widthx, Nod_widthy, Vj, alpha, delta, phi))
    mu = np.load("%s/mu Lx = %.1f Wj = %.1f nodx = %.1f nody = %.1f Vj = %.1f alpha = %.1f delta = %.2f phi = %.3f.npy" % (dirS, Lx*.1, Junc_width, Nod_widthx, Nod_widthy, Vj, alpha, delta, phi))

    print(boundary[:, 0])

    for i in range(boundary.shape[1]):
        plt.scatter(boundary[:, i], mu, c='r', s = 2)

    plt.grid()
    plt.xlabel(r'$E_z$ (meV)')
    plt.ylabel(r'$\mu$ (meV)')
    plt.xlim(gi, gf)
    #plt.ylim(-2,2)
    title = r"$L_x$ = %.1f nm, $W_j$ = %.1f nm, $nodule_x$ = %.1f nm, $nodule_y$ = %.1f nm, $V_j$ = %.1f meV, $\phi$ = %.2f " % (Lx*.1, Junc_width, Nod_widthx, Nod_widthy, Vj, phi)
    plt.title(title, loc = 'center', wrap = True)
    plt.subplots_adjust(top=0.85)
    plt.savefig('juncwidth = {} nodwidthx = {} nodwidthy = {} phi = {} Vj = {}.png'.format(Junc_width, Nod_widthx, Nod_widthy, delta, alpha, phi, Vj))
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
