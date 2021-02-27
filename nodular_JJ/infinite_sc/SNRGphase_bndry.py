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
phi = np.pi #SC phase difference
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
    boundary = np.load("%s/boundary Lx = %.1f Wj = %.1f nodx = %.1f nody = %.1f Vj = %.1f alpha = %.1f delta = %.2f phi = %.3f mu_i = %.1f mu_f = %.1f.npy" % (dirS, Lx*.1, Junc_width, Nod_widthx,  Nod_widthy, Vj, alpha, delta, phi, mu_i, mu_f))
    for i in range( mu_steps):
        print(mu_steps-i, "| mu =", mu[i])
        gx = fndrs.SNRG_gam_finder(ax, ay, mu[i], gi, gf, Wj=Wj, Lx=Lx, cutx=cutx, cuty=cuty, Vj=Vj, alpha=alpha, delta=delta, phi=phi, k=20, tol = 1e-5, PLOT=False)
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
                boundary[i, j+1:] = None
                boundary[i+1, j+1:] = None
                pass

    for i in range(num_bound):
        #plt.scatter(boundary[:, i], mu, s=1, c='k', zorder=2)
        pass
    color = colors.colorConverter.to_rgba('lightcyan', alpha=1)
    for i in range(int(num_bound/2)):
        art = plt.fill_betweenx(mu, boundary[:, 2*i], boundary[:, 2*i+1], visible = True, alpha=1, fc=color, ec='k', lw=2.8, where=dist_arr[:,i]<0.1, zorder=0)
    for i in range(int(num_bound/2)):
        art = plt.fill_betweenx(mu, boundary[:, 2*i], boundary[:, 2*i+1], visible = True, alpha=1, fc=color, ec='face', lw=1, where=dist_arr[:,i]<0.1, zorder=2.5)

    plt.xlabel(r'$E_Z$ (meV)')
    plt.ylabel(r'$\mu$ (meV)')
    title = r'Lx = {} nm, lx = {} nm, W1 = {} nm, W2 = {} nm, Vj = {} meV'.format(Lx*.1, Nod_widthx, Junc_width, Junc_width-2*Nod_widthy, Vj)
    plt.title(title, loc = 'center', wrap = True)

    plt.xlim(-0.2, 4.2)
    plt.ylim(-2.2, 15.5)

    plt.show()
