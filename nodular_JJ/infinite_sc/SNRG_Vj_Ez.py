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
Wj = 1000 #Junction region [A]
cutx = 4 #width of nodule
cuty = 5 #height of nodule

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
phi = 0*np.pi #SC phase difference
delta = 1 #Superconducting Gap: [meV]
mu = 30 #junction potential: [meV]

Vj_i = -100
Vj_f = 50
res = 0.05
delta_Vj = Vj_f - Vj_i
Vj_steps = int(delta_Vj/res)
Vj = np.linspace(Vj_i, Vj_f, Vj_steps) #Chemical Potential: [meV]

print("alpha = ", alpha)
print("Vj_i = ", Vj_i)
print("Vj_f = ", Vj_f)

gi = 0
gf = 6
num_bound = 6
boundary = np.zeros((Vj_steps, num_bound))
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
    for i in range(Vj_steps):
        print(Vj_steps-i, "| Vj =", Vj[i])
        gx = fndrs.SNRG_gam_finder(ax, ay, mu, gi, gf, Wj=Wj, Lx=Lx, cutx=cutx, cuty=cuty, Vj=Vj[i], alpha=alpha, delta=delta, phi=phi, k=50)
        for j in range(num_bound):
            if j >= gx.size:
                boundary[i, j] = None
            else:
                boundary[i, j] = gx[j]

        np.save("%s/boundary Lx = %.1f Wj = %.1f nodx = %.1f nody = %.1f mu = %.1f alpha = %.1f delta = %.2f phi = %.3f Vj_i = %.1f Vj_f = %.1f.npy" % (dirS, Lx*.1, Junc_width, Nod_widthx, Nod_widthy, mu, alpha, delta, phi, Vj_i, Vj_f), np.array(boundary))
        gc.collect()

    sys.exit()
else:
    boundary = np.load("%s/boundary Lx = %.1f Wj = %.1f nodx = %.1f nody = %.1f mu = %.1f alpha = %.1f delta = %.2f phi = %.3f Vj_i = %.1f Vj_f = %.1f.npy" % (dirS, Lx*.1, Junc_width, Nod_widthx, Nod_widthy, mu, alpha, delta, phi, Vj_i, Vj_f))

    Vj = np.linspace(Vj_i, Vj_f, boundary.shape[0])
    for i in range(boundary.shape[1]):
        plt.scatter(boundary[:, i], Vj, c='r', s = 2)
        #plt.plot(boundary[:, i], mu, c='r')
    plt.grid()
    plt.xlabel(r'$E_z$ (meV)')
    plt.ylabel(r'$V_J$ (meV)')
    plt.xlim(gi, gf)
    #plt.ylim(-2,2)
    title = r"$L_x$ = %.1f nm, $W_j$ = %.1f nm, $nodule_x$ = %.1f nm, $nodule_y$ = %.1f nm, $\mu$ = %.1f meV, $\phi$ = %.2f " % (Lx*.1, Junc_width, Nod_widthx, Nod_widthy, mu, phi)
    plt.title(title, loc = 'center', wrap = True)
    plt.subplots_adjust(top=0.85)
    #plt.savefig('boundary juncwidth = {} nodwidthx = {} nodwidthy = {} phi = {} Vj = {} mu_i = {} mu_f = {}.png'.format(Junc_width, Nod_widthx, Nod_widthy, delta, alpha, phi, Vj, mu_i, mu_f))
    plt.show()
