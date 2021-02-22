import sys
import os
import gc

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
import scipy.interpolate as interp

import majoranaJJ.modules.plots as plots #plotting functions
import majoranaJJ.modules.finders as fndrs
import majoranaJJ.modules.SNRG as SNRG
###################################################
#Defining System
ax = 50 #lattice spacing in x-direction: [A]
ay = 50 #lattice spacing in y-direction: [A]
Nx = 10 #Number of lattice sites along x-direction
Wj = 600 #Junction region [A]
cutx = 2 #width of nodule
cuty = 5 #height of nodule
Lx = Nx*ax #Angstrom
Junc_width = Wj*.1 #nm
Nod_widthx = cutx*ax*.1 #nm
Nod_widthy = cuty*ay*.1 #nm

print("Lx = ", Lx*.1, "(nm)" )
print("Nodule Width in x-direction = ", Nod_widthx, "(nm)")
print("Nodule Width in y-direction = ", Nod_widthy, "(nm)")
print("Junction Width = ", Junc_width, "(nm)")
###################################################
#Defining Hamiltonian parameters
alpha = 200 #Spin-Orbit Coupling constant: [meV*A]
phi = 0*np.pi #SC phase difference
delta = 0.3 #Superconducting Gap: [meV]
Vj = -50 #junction potential: [meV]

mu_i = -5
mu_f = 15
res = 0.01
delta_mu = mu_f - mu_i
mu_steps = int(delta_mu/res)
mu = np.linspace(mu_i, mu_f, mu_steps) #Chemical Potential: [meV]
print("alpha = ", alpha)
print("Mu_i = ", mu_i)
print("Mu_f = ", mu_f)
print("Vj = ", Vj)

gi = 0
gf = 5.0
num_bound = 10
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
        gx = fndrs.SNRG_gam_finder(ax, ay, mu[i], gi, gf, Wj=Wj, Lx=Lx, cutx=cutx, cuty=cuty, Vj=Vj, alpha=alpha, delta=delta, phi=phi, k=20, tol = 3e-3, PLOT=False)
        for j in range(num_bound):
            if j >= gx.size:
                boundary[i, j] = None
            else:
                boundary[i, j] = gx[j]

        np.save("%s/boundary Lx = %.1f Wj = %.1f nodx = %.1f nody = %.1f Vj = %.1f alpha = %.1f delta = %.2f phi = %.3f mu_i = %.1f mu_f = %.1f.npy" % (dirS, Lx*.1, Junc_width, Nod_widthx,  Nod_widthy, Vj, alpha, delta, phi, mu_i, mu_f), np.array(boundary))
        gc.collect()

    sys.exit()
else:
    boundary = np.load("%s/boundary Lx = %.1f Wj = %.1f nodx = %.1f nody = %.1f Vj = %.1f alpha = %.1f delta = %.2f phi = %.3f mu_i = %.1f mu_f = %.1f.npy" % (dirS, Lx*.1, Junc_width, Nod_widthx,  Nod_widthy, Vj, alpha, delta, phi, mu_i, mu_f))
    mu = np.linspace(mu_i, mu_f, boundary.shape[0])

    for i in range(num_bound):
        #spl = interp.splrep(mu, boundary[:, i])
        #mu_new = np.linspace(mu_i, mu_f, mu.shape[0]*10)
        #bnd_new = interp.splev(mu_new, spl)
        #plt.plot(bnd_new, mu_new, c='k', linewidth=1.5, zorder=1)
        #plt.plot(boundary[:, i], mu, c='k', linewidth=1.5)
        plt.scatter(boundary[:, i], mu, s=1.5, c='r', zorder=2)

    #plt.fill_betweenx(mu, boundary[:, 0], boundary[:, 1], visible = True, alpha=0.5, color='steelblue')

    """
    Ez_arr = np.linspace(gi, gf, 50000)
    top_arr = np.zeros(mu.shape[0], Ez_arr.shape[0])
    for i in range(mu.shape[0]):
        for j in range(Ez_arr.shape[0]):
            if trivial: #passed even number of phsae boundaries
                top_arr[i,j] = 1
            if topological:#passed odd number of phsae boundaries
                top_arr[i,j] = -1
    """
    plt.xlabel(r'$E_Z$ (meV)')
    plt.ylabel(r'$\mu$ (meV)')
    title = r'Lx = {} nm, lx = {} nm, W1 = {} nm, W2 = {} nm, Vj = {} meV'.format(Lx*.1, Nod_widthx, Junc_width, Junc_width-2*Nod_widthy, Vj)
    plt.title(title, loc = 'center', wrap = True)

    #plt.xlim(0, 4.2)
    #plt.ylim(-2, 15.2)

    plt.show()
