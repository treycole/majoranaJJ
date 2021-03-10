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
import majoranaJJ.modules.distance as distance
import matplotlib.colors as colors
###################################################
#Defining System
ax = 50 #lattice spacing in x-direction: [A]
ay = 50 #lattice spacing in y-direction: [A]
Nx = 12 #Number of lattice sites along x-direction
Wj = 1000 #Junction region [A]
cutx = 4 #width of nodule
cuty = 8 #height of nodule

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
alpha = 200 #Spin-Orbit Coupling constant: [meV*A]
phi = 0*np.pi #SC phase difference
delta = 0.3 #Superconducting Gap: [meV]
Vj = -40 #junction potential: [meV]

mu_i = -5
mu_f = 15
res = 0.005
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
        gx = fndrs.SNRG_gam_finder(ax, ay, mu[i], gi, gf, Wj=Wj, Lx=Lx, cutxT=cutxT, cutyT=cutyT, cutxB=cutxB, cutyB=cutyB, Vj=Vj, alpha=alpha, delta=delta, phi=phi, k=50, tol = 1e-5, PLOT=False)
        for j in range(num_bound):
            if j >= gx.size:
                boundary[i, j] = None
            else:
                boundary[i, j] = gx[j]

        np.save("%s/boundary Lx = %.1f Wj = %.1f cutxT = %.1f cutyT = %.1f, cutxB = %.1f cutyB = %.1f, Vj = %.1f alpha = %.1f delta = %.2f phi = %.3f mu_i = %.1f mu_f = %.1f.npy" % (dirS, Lx*.1, Junc_width, cutxT_width,  cutyT_width, cutxB_width, cutyB_width, Vj, alpha, delta, phi, mu_i, mu_f), np.array(boundary))
        gc.collect()

    sys.exit()
else:
    boundary = np.load("%s/boundary Lx = %.1f Wj = %.1f cutxT = %.1f cutyT = %.1f, cutxB = %.1f cutyB = %.1f, Vj = %.1f alpha = %.1f delta = %.2f phi = %.3f mu_i = %.1f mu_f = %.1f.npy" % (dirS, Lx*.1, Junc_width, cutxT_width,  cutyT_width, cutxB_width, cutyB_width, Vj, alpha, delta, phi, mu_i, mu_f))
    mu = np.linspace(mu_i, mu_f, boundary.shape[0])

    fig, axs = plt.subplots(1, gridspec_kw={'hspace':0.1, 'wspace':0.1})
    axs.set_yticks([ 0, 5, 10])
    axs.label_outer()
    for i in range(mu.shape[0]-1):
        for j in range(int(boundary.shape[1]/2)):
            if np.isnan(boundary[i,2*j+1]) and not np.isnan(boundary[i,2*j]):
                boundary[i,2*j+1] = 5
                boundary[i+1,2*j+1] = 5
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

    color = colors.colorConverter.to_rgba('lightcyan', alpha=1)
    color = list(color)
    color[0] = 0.85
    for i in range(int(num_bound/2)):
        art = axs.fill_betweenx(mu, boundary[:, 2*i], boundary[:, 2*i+1], visible = True, alpha=1, fc=color, ec='k', lw=2.8, where=dist_arr[:,i]<0.1, zorder=1)
    for i in range(int(num_bound/2)):
        art = axs.fill_betweenx(mu, boundary[:, 2*i], boundary[:, 2*i+1], visible = True, alpha=1, fc=color, ec='face', lw=1, where=dist_arr[:,i]<0.1, zorder=1.2)

    for i in range(num_bound):
        #axs.scatter(boundary[:, i], mu, s=1, c='k', zorder=2)
        pass
    plt.subplots_adjust(top=0.9, left=0.1, bottom=0.1, right=0.97)
    axs.set_xlabel(r'$E_Z$ (meV)', size=9)
    axs.set_ylabel(r'$\mu$ (meV)', size=9)

    axs.set_xlim([0, 4.2])
    axs.set_ylim([-3, 13])
    axs.plot([1,1], [-2,12], c='b', lw=1.5, mec='k', zorder=4)
    axs.plot([0,3], [9.2,9.2], c='b', lw=1.5, mec='k', zorder=4)
    axs.plot([0,3], [11.1213,11.1213], c='b', lw=1.5, mec='k', zorder=4)
    axs.plot([0,3], [7.36,7.36], c='b', lw=1.5, mec='k', zorder=4)
    axs.tick_params(axis='x', labelsize=9)
    axs.tick_params(axis='y', labelsize=9)
    axs.xaxis.set_major_locator(ticker.MultipleLocator(1))
    title = r'Lx = {} nm, lxT = {} nm, lxB = {} nm, W1 = {} nm, W2 = {} nm, Vj = {} meV'.format(Lx*.1, cutxT_width, cutxB_width, Junc_width, Junc_width-(cutyT_width+cutyB_width), Vj)
    plt.title(title, loc = 'center', wrap = True)

    plt.show()
