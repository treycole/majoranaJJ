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
import majoranaJJ.operators.potentials as pot
import majoranaJJ.lattice.shapes as shps
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
alpha = 200 #Spin-Orbit Coupling constant: [meV*A]
phi = 0*np.pi #SC phase difference
delta = 0.3 #Superconducting Gap: [meV]
Vj =  10 #meV, Junction potential

mu_i = 5
mu_f = 25
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
#coor = shps.square(Nx, int(Wj/ay)+2)
#plots.potential_profile(coor, pot.Vjj(coor=coor, Wj=int(Wj/ay), Vsc=0, Vj=Vj, cutxT=cutxT, cutyT=cutyT, cutxB=cutxB, cutyB=cutyB))
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
    boundary = np.load("%s/boundary Lx = %.1f Wj = %.1f cutxT = %.1f cutyT = %.1f cutxB = %.1f cutyB = %.1f Vj = %.1f phi = %.3f mu_i = %.1f mu_f = %.1f.npy" % (dirS, Lx*.1, Junc_width, cutxT_width,  cutyT_width, cutxB_width, cutyB_width, Vj, phi, mu_i, 15))
    boundary_new = np.zeros((int(mu_steps/2), num_bound))
    boundary = np.concatenate((boundary, boundary_new))
    #mu = np.load("%s/mu Lx = %.1f Wj = %.1f cutxT = %.1f cutyT = %.1f cutxB = %.1f cutyB = %.1f Vj = %.1f phi = %.3f mu_i = %.1f mu_f = %.1f.npy" % (dirS, Lx*.1, Junc_width, cutxT_width,  cutyT_width, cutxB_width, cutyB_width, Vj, phi, mu_i, 15))
    #print(boundary[2853, 0], boundary[2853:2855, 0], mu[2853:2855])
    #cutoff = 2853
    #res = res/5
    #delta_mu = mu_f - mu[cutoff]
    #mu_steps = int(delta_mu/res)
    #mu_new = np.linspace(mu[cutoff], mu_f, mu_steps)
    #mu = np.concatenate((mu[0:cutoff], mu_new), axis=None)
    #num_bound = 10
    #boundary_new = np.zeros((mu.shape[0], num_bound))
    #for i in range(num_bound):
    #    boundary_new[:, i] = np.concatenate((boundary[0:cutoff, i], np.zeros(mu_new.shape)), axis=None)
    np.save("%s/mu Lx = %.1f Wj = %.1f cutxT = %.1f cutyT = %.1f cutxB = %.1f cutyB = %.1f Vj = %.1f phi = %.3f mu_i = %.1f mu_f = %.1f.npy" % (dirS, Lx*.1, Junc_width, cutxT_width, cutyT_width, cutxB_width, cutyB_width, Vj, phi, mu_i, mu_f), mu)
    for i in range(0, mu.shape[0]):
        print(mu.shape[0]-i, "| mu =", mu[i])
        gx = fndrs.SNRG_gam_finder(ax, ay, mu[i], gi, gf, Wj=Wj, Lx=Lx, cutxT=cutxT, cutyT=cutyT, cutxB=cutxB, cutyB=cutyB, Vj=Vj, alpha=alpha, delta=delta, phi=phi, k=20, tol = 1e-5, PLOT=False, plot_junction=False)
        for j in range(num_bound):
            if j >= gx.size:
                boundary[i, j] = None
            else:
                boundary[i, j] = gx[j]

        np.save("%s/boundary Lx = %.1f Wj = %.1f cutxT = %.1f cutyT = %.1f cutxB = %.1f cutyB = %.1f Vj = %.1f phi = %.3f mu_i = %.1f mu_f = %.1f.npy" % (dirS, Lx*.1, Junc_width, cutxT_width,  cutyT_width, cutxB_width, cutyB_width, Vj, phi, mu_i, mu_f), boundary)

        gc.collect()

    sys.exit()
else:
    boundary = np.load("%s/boundary Lx = %.1f Wj = %.1f cutxT = %.1f cutyT = %.1f cutxB = %.1f cutyB = %.1f Vj = %.1f phi = %.3f mu_i = %.1f mu_f = %.1f.npy" % (dirS, Lx*.1, Junc_width, cutxT_width,  cutyT_width, cutxB_width, cutyB_width, Vj, phi, mu_i, mu_f))
    mu = np.load("%s/mu Lx = %.1f Wj = %.1f cutxT = %.1f cutyT = %.1f cutxB = %.1f cutyB = %.1f Vj = %.1f phi = %.3f mu_i = %.1f mu_f = %.1f.npy" % (dirS, Lx*.1, Junc_width, cutxT_width,  cutyT_width, cutxB_width, cutyB_width, Vj, phi, mu_i, mu_f))

    fig, axs = plt.subplots(1, gridspec_kw={'hspace':0.1, 'wspace':0.1})
    axs.set_yticks([ 0, 5, 10])
    axs.label_outer()
    c = ['g', 'r', 'b', 'y', 'pink', 'orange']
    for i in range(len(c)):
        #axs.scatter(boundary[:, i], mu, s=10, c=c[i], zorder=0)
        pass

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
        art = axs.fill_betweenx(mu, boundary[:, 2*i], boundary[:, 2*i+1], visible = True, alpha=1, fc=color, ec='k', lw=4, where=dist_arr[:,i]<0.1, zorder=1)
    for i in range(int(num_bound/2)):
        art = axs.fill_betweenx(mu, boundary[:, 2*i], boundary[:, 2*i+1], visible = True, alpha=1, fc=color, ec='face', lw=0.5, where=dist_arr[:,i]<0.1, zorder=1.2)

    c = ['g', 'r', 'b', 'y', 'pink', 'orange']
    for i in range(num_bound):
        #axs.scatter(boundary[:, i], mu, c='k', zorder=0)
        pass
    plt.subplots_adjust(top=0.9, left=0.1, bottom=0.1, right=0.97)
    axs.set_xlabel(r'$E_Z$ (meV)', size=9)
    axs.set_ylabel(r'$\mu$ (meV)', size=9)

    axs.set_xlim([gi, gf])
    axs.set_ylim([mu_i, mu_f])
    #axs.plot([0.6, 0.6], [-2, 13.2], c='r', lw=1.5, mec='k', zorder=4)
    #axs.plot([0, 3], [10.30, 10.30], c='r', lw=1.5, mec='k', zorder=4)
    #axs.plot([0, 3], [6.28, 6.28], c='r', lw=1.5, mec='k', zorder=4)
    #axs.plot([0, 3], [2.37, 2.37], c='r', lw=1.5, mec='k', zorder=4)
    axs.tick_params(axis='x', labelsize=9)
    axs.tick_params(axis='y', labelsize=9)
    axs.xaxis.set_major_locator(ticker.MultipleLocator(1))
    title = r'Lx = %.1f nm, lxT = %.1f nm, lxB = %.1f nm, Nody_T = %.1f nm, Nody_B = %.1f nm, Vj = %.1f meV, phi = %.2f' % (Lx*.1, cutxT_width, cutxB_width, cutyT_width, cutyB_width, Vj, phi)
    plt.title(title, loc = 'center', wrap = True)

    plt.show()
