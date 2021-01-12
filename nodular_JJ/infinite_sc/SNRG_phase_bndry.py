import sys
import os
import gc

import numpy as np
import matplotlib.pyplot as plt

import majoranaJJ.modules.plots as plots #plotting functions
import majoranaJJ.modules.finders as fndrs
import majoranaJJ.modules.SNRG as SNRG
###################################################
#Defining System
ax = 50 #lattice spacing in x-direction: [A]
ay = 50 #lattice spacing in y-direction: [A]
Nx = 10 #Number of lattice sites along x-direction
Wj = 2000 #Junction region [A]
cutx = 4 #width of nodule
cuty = 10 #height of nodule
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
mu_f = 10
res = 0.1
delta_mu = mu_f - mu_i
mu_steps = int(delta_mu/res)
mu = np.linspace(mu_i, mu_f, mu_steps) #Chemical Potential: [meV]
#dmu = -0.010347

print("alpha = ", alpha)
print("Mu_i = ", mu_i)
print("Mu_f = ", mu_f)
print("Vj = ", Vj)

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
        gx = fndrs.SNRG_gam_finder(ax, ay, mu[i], gi, gf, Wj=Wj, Lx=Lx, cutx=cutx, cuty=cuty, Vj=Vj, alpha=alpha, delta=delta, phi=phi)
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


"""
for i in range(gx.shape[0]):
    for j in range(mu.shape[0]):
        print(gx.shape[0]-i, mu.shape[0]-j)
        H = SNRG.Junc_eff_Ham_gen(omega=0, Wj=Wj, nodx=0, nody=0, ax=ax, ay_targ=ay, kx=qx, m_eff=0.026, alp_l=alpha, alp_t=alpha, mu=mu[j], Vj=Vj, Gam=gx[i], delta=delta, phi=phi, iter=50,eta=0)

        eigs, vecs = spLA.eigsh(H, k=k, sigma=0, which='LM')
        idx_sort = np.argsort(eigs)
        eigs = eigs[idx_sort]
        gap_k0[i,j] = eigs[int(k/2)]
"""
