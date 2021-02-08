import sys
import os
import gc

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker

import majoranaJJ.modules.plots as plots #plotting functions
import majoranaJJ.modules.finders as fndrs
import majoranaJJ.modules.SNRG as SNRG
import majoranaJJ.modules.fig_params as params
###################################################
#Defining System
ax = 50 #lattice spacing in x-direction: [A]
ay = 50 #lattice spacing in y-direction: [A]
Nx = 3 #Number of lattice sites along x-direction
Wj = 1000 #Junction region [A]
cutx = 0 #width of nodule
cuty = 0 #height of nodule
Lx = Nx*ax #Angstrom
Junc_width = Wj*.10 #nm
Nod_widthx = cutx*ax*.1 #nm
Nod_widthy = cuty*ay*.1 #nm

print("Nodule Width in x-direction = ", Nod_widthx, "(nm)")
print("Nodule Width in y-direction = ", Nod_widthy, "(nm)")
print("Junction Width = ", Junc_width, "(nm)")
###################################################
#Defining Hamiltonian parameters
alpha = 200 #Spin-Orbit Coupling constant: [meV*A]
phi = 0*np.pi #SC phase difference
delta = 0.3 #Superconducting Gap: [meV]
Vj = 0 #junction potential: [meV]

mu_i = -2
mu_f = 20
res = 0.1
delta_mu = mu_f - mu_i
mu_steps = int(delta_mu/res)
mu = np.linspace(mu_i, mu_f, mu_steps) #Chemical Potential: [meV]
#mu = np.linspace(mu[75], mu_f, mu_steps-75)
print("alpha = ", alpha)
print("Mu_i = ", mu_i)
print("Mu_f = ", mu_f)
print("Vj = ", Vj)

gi = 0
gf = 5
num_bound = 4
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
        gx = fndrs.SNRG_gam_finder(ax, ay, mu[i], gi, gf, Wj=Wj, Lx=Lx, cutx=cutx, cuty=cuty, Vj=Vj, alpha=alpha, delta=delta, phi=phi, k=20, tol=5e-7)
        for j in range(num_bound):
            if j >= gx.size:
                boundary[i, j] = None
            else:
                boundary[i, j] = gx[j]

        np.save("%s/boundary Lx = %.1f Wj = %.1f nodx = %.1f nody = %.1f Vj = %.1f alpha = %.1f delta = %.2f phi = %.3f mu_i = %.1f mu_f = %.1f.npy" % (dirS, Lx*.1, Junc_width, Nod_widthx,  Nod_widthy, Vj, alpha, delta, phi, mu_i, mu_f), np.array(boundary))
        gc.collect()

    sys.exit()
else:
    boundary_pi = np.load("%s/boundary Lx = %.1f Wj = %.1f nodx = %.1f nody = %.1f Vj = %.1f alpha = %.1f delta = %.2f phi = %.3f mu_i = %.1f mu_f = %.1f.npy" % (dirS, Lx*.1, Junc_width, Nod_widthx,  Nod_widthy, Vj, alpha, delta, np.pi, mu_i, mu_f))
    boundary_0 = np.load("%s/boundary Lx = %.1f Wj = %.1f nodx = %.1f nody = %.1f Vj = %.1f alpha = %.1f delta = %.2f phi = %.3f mu_i = %.1f mu_f = %.1f.npy" % (dirS, Lx*.1, Junc_width, Nod_widthx,  Nod_widthy, Vj, alpha, delta, 0, mu_i, mu_f))
    mu = np.linspace(mu_i, mu_f, boundary.shape[0])

    fig, axs = plt.subplots(2)
    #axs.set_facecolor("w")
    #axs.set_yticks([0, 5, 10, 15, 20])
    for i in range(boundary_0.shape[1]):
        axs[0].plot(boundary_0[:, i], mu, c='k', linewidth=2)
        axs[1].plot(boundary_pi[:, i], mu, c='k', linewidth=2)

    axs[0].fill_betweenx(mu, boundary_0[:, 0], boundary_0[:, 1], visible = True, alpha=0.5, color='steelblue')
    axs[1].fill_betweenx(mu, boundary_pi[:, 0], boundary_pi[:, 1], visible = True, alpha=0.5, color='steelblue')

    for ax in axs.flat:
        ax.set(xlabel=r'$E_Z$ (meV)', ylabel=r'$\mu$ (mev)')
    for ax in fig.get_axes():
        ax.label_outer()

    plt.subplots_adjust(top=0.95, left=0.25, bottom=0.15)
    axs[1].text(0.05, 12.8, '(b)', fontdict=None, fontsize=12)
    axs[0].text(0.05, 12.8, '(a)', fontdict=None, fontsize=12)

    cut1 = np.linspace(-2, 12, 100)
    cut2 = np.linspace(0, 3, 100)
    axs[0].plot(cut1*0+1, cut1, c='b', lw=2)
    axs[0].plot(cut2, cut2*0, c='r', lw=2)
    axs[1].plot(cut1*0+1, cut1, c='b', lw=2)
    axs[1].plot(cut2, cut2*0+10, c='r', lw=2)

    axs[0].set_xlim([0, 4.5])
    axs[0].set_ylim([-2, 15])
    axs[1].set_xlim([0, 4.5])
    axs[1].set_ylim([-2, 15])
    axs[0].xaxis.set_major_locator(ticker.MultipleLocator(1))
    axs[1].xaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.savefig('boundary juncwidth = {} nodwidthx = {} nodwidthy = {} phi = {} Vj = {} mu_i = {} mu_f = {}.png'.format(Junc_width, Nod_widthx, Nod_widthy, delta, alpha, phi, Vj, mu_i, mu_f), dpi=700)
    plt.show()
