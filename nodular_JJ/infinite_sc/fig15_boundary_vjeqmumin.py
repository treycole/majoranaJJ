import sys
import os
import gc

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
import matplotlib.colors as colors

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

cutxT = cutx
cutxB = cutx
cutyT = cuty
cutyB = cuty
Lx = Nx*ax #Angstrom
Junc_width = Wj*.1 #nm
cutxT_width = cutxT*ax*.1 #nm
cutyT_width = cutyT*ax*.1 #nm
cutxB_width = cutxB*ax*.1 #nm
cutyB_width = cutyB*ax*.1 #nm

print("Lx = ", Lx*.1, "(nm)" )
print("Top Nodule Width in x-direction = ", cutxT_width, "(nm)")
print("Bottom Nodule Width in x-direction = ", cutxB_width, "(nm)")
print("Top Nodule Width in y-direction = ", cutyT_width, "(nm)")
print("Bottom Nodule Width in y-direction = ", cutyB_width, "(nm)")
print("Junction Width = ", Junc_width, "(nm)")
###################################################
#Defining Hamiltonian parameters
#Defining Hamiltonian parameters
alpha = 200 #Spin-Orbit Coupling constant: [meV*A]
phi = 0*np.pi #SC phase difference
delta = 0.3 #Superconducting Gap: [meV]

mu_i = -5
mu_f = 15
res = 0.005
delta_mu = mu_f - mu_i
mu_steps = int(delta_mu/res)
mu = np.linspace(mu_i, mu_f, mu_steps) #Chemical Potential: [meV]

gi = 0
gf = 5.0
num_bound = 10
boundary = np.zeros((mu_steps, num_bound))
###################################################
#phase diagram mu vs gam
dirS = 'boundary_data'
Vj_i = 0 #Junction potential: [meV]
Vj_f = 40
min_Ez = np.load("%s/min_EZfxVj Lx = %.1f Wj = %.1f cutxT = %.1f cutyT = %.1f cutxB = %.1f cutyB = %.1f phi = %.3f Vj_i = %.1f Vj_f = %.1f.npy" % (dirS, Lx*.1, Junc_width, cutxT_width, cutyT_width, cutxB_width, cutyB_width, phi, Vj_i, Vj_f))
min_mu = np.load("%s/min_mufxVj Lx = %.1f Wj = %.1f cutxT = %.1f cutyT = %.1f cutxB = %.1f cutyB = %.1f phi = %.3f Vj_i = %.1f Vj_f = %.1f.npy" % (dirS, Lx*.1, Junc_width, cutxT_width, cutyT_width, cutxB_width, cutyB_width, phi, Vj_i, Vj_f))

Vj = np.linspace(Vj_i, Vj_f, min_mu.shape[0])
x = np.linspace(mu_i, mu_f, mu_steps)
Vj_for_min_Ez  = np.zeros(x.shape[0])
for i in range(x.shape[0]):
    Vj_for_min_Ez [i] = fndrs.Lagrange_interp(x[i], min_mu, Vj)

if not os.path.exists(dirS):
    os.makedirs(dirS)
try:
    PLOT = str(sys.argv[1])
except:
    PLOT = 'F'
if PLOT != 'P':
    for i in range(0, mu.shape[0]):
        print(mu.shape[0]-i, "| mu =", mu[i])
        gx = fndrs.SNRG_gam_finder(ax, ay, mu[i], gi, gf, Wj=Wj, Lx=Lx, cutxT=cutxT, cutyT=cutyT, cutxB=cutxB, cutyB=cutyB, Vj=Vj_for_min_Ez[i], alpha=alpha, delta=delta, phi=phi, k=20, tol=1e-5, PLOT=False, plot_junction=False)
        for j in range(num_bound):
            if j >= gx.size:
                boundary[i, j] = None
            else:
                boundary[i, j] = gx[j]

        np.save("%s/boundaryvj=mu Lx = %.1f Wj = %.1f cutxT = %.1f cutyT = %.1f cutxB = %.1f cutyB = %.1f phi = %.3f mu_i = %.1f mu_f = %.1f.npy" % (dirS, Lx*.1, Junc_width, cutxT_width, cutyT_width, cutxB_width, cutyB_width, phi, mu_i, mu_f), boundary)

        gc.collect()
else:
    boundary = np.load("%s/boundaryvj=mu Lx = %.1f Wj = %.1f cutxT = %.1f cutyT = %.1f cutxB = %.1f cutyB = %.1f phi = %.3f mu_i = %.1f mu_f = %.1f.npy" % (dirS, Lx*.1, Junc_width, cutxT_width, cutyT_width, cutxB_width, cutyB_width, phi, mu_i, mu_f))
    mu = np.linspace(mu_i, mu_f, boundary.shape[0])

    fig, axs = plt.subplots(1, gridspec_kw={'hspace':0.1, 'wspace':0.1})
    axs.set_yticks([0, 5, 10])
    axs.label_outer()

    for i in range(mu.shape[0]-1):
        for j in range(int(boundary.shape[1]/2)):
            if np.isnan(boundary[i,2*j+1]) and not np.isnan(boundary[i,2*j]):
                boundary[i,2*j+1] = 5
                break

    dist_arr = np.zeros((mu.shape[0], num_bound))
    for i in range(int(mu.shape[0])-1):
        for j in range(num_bound-1):
            if np.isnan(boundary[i, j]) or np.isnan(boundary[i+1, j]):
                dist_arr[i,j] = 100000
            else:
                dist_arr[i,j] = abs(boundary[i, j] - boundary[i+1, j])
            if dist_arr[i,j]>0.1:
                boundary[i, j:] = None
                pass

    for i in range(1, mu.shape[0]-1):
        for j in range(num_bound):
            if np.isnan(boundary[i+1,j]) and np.isnan(boundary[i-1, j]):
                boundary[i,j]=None

    color = colors.colorConverter.to_rgba('lightblue', alpha=1)
    for i in range(int(num_bound/2)):
        art = axs.fill_betweenx(mu, boundary[:, 2*i], boundary[:, 2*i+1], visible = True, fc=color, ec='k', lw=4, where=dist_arr[:,i]<0.1, zorder=1)
    for i in range(int(num_bound/2)):
        art = axs.fill_betweenx(mu, boundary[:, 2*i], boundary[:, 2*i+1], visible = True, fc=color, ec='face', lw=1, where=dist_arr[:,i]<0.1, zorder=1.2)


    axs.set_xlabel(r'$E_Z$ (meV)', size=9)
    axs.set_ylabel(r'$\mu$ (mev)', size=9)

    axs.set_ylim([-3, 13])
    axs.set_xlim([0, 4.2])

    plt.subplots_adjust(top=0.95, left=0.15, bottom=0.26, right=0.98)

    axs.tick_params(axis='x', labelsize=9)
    axs.tick_params(axis='y', labelsize=9)
    axs.xaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.savefig('FIG15', dpi=700)
    plt.show()
