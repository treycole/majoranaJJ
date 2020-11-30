import sys
import time
import os
import gc

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.signal import argrelextrema
import scipy.linalg as LA
import scipy.sparse.linalg as spLA

import majoranaJJ.operators.sparse_operators as spop #sparse operators
import majoranaJJ.modules.constants as params
import majoranaJJ.modules.self_energy as slfNRG
from majoranaJJ.operators.potentials import Vjj #potential JJ
import majoranaJJ.lattice.nbrs as nb #neighbor arrays
import majoranaJJ.lattice.shapes as shps #lattice shapes
import majoranaJJ.modules.plots as plots #plotting functions
import majoranaJJ.modules.gamfinder as gamfinder
from majoranaJJ.modules.checkers import boundary_check as bc
import majoranaJJ.modules.checkers as check
###################################################
#Defining System
ax = 50 #lattice spacing in x-direction: [A]
ay = 50 #lattice spacing in y-direction: [A]
Wj = 1000 #Junction width: [A]
cutx = 0 #width of nodule
cuty = 0 #height of nodule

Junc_width = Wj*.1 #nm
Nod_widthx = cutx*.1 #nm
Nod_widthy = cuty*.1 #nm
print("Nodule Width in x-direction = ", Nod_widthx, "(nm)")
print("Nodule Width in y-direction = ", Nod_widthy, "(nm)")
print("Junction Width = ", Junc_width, "(nm)")
###################################################
#Defining Hamiltonian parameters
alpha = 200 #Spin-Orbit Coupling constant: [meV*A]
phi = np.pi #SC phase difference
delta = 1 #Superconducting Gap: [meV]
Vsc = 0 #SC potential: [meV]
Vj = 0 #Junction potential: [meV]

mui = 0
muf = 20
res = 0.1
steps_mu = int((muf-mui)/(res)) + 1
mu = np.linspace(mui, muf, steps_mu) #meV

gi = 0
gf = 5.0
res = 0.01
steps_gam = int((gf - gi)/(res)) + 1
gx = np.linspace(gi, gf, steps_gam)

omega = np.linspace(0, delta, 500)

k = 4
tol = 1e-3
###################################################
def gap(gam, mu, Vj, Wj, alpha, delta, phi, tol):
    q_steps = 10
    if Vj < 0:
        VVJ = Vj
    else:
        VVJ = 0
    qmax = np.sqrt(2*(muf-VVJ)/params.xi)*1.5
    #print(qmax, np.pi/ax)
    qx = np.linspace(0, qmax, q_steps) #kx in the first Brillouin zone
    omega0_bands = np.zeros(qx.shape[0])
    for q in range(qx.shape[0]):
        #print(qx.shape[0]-q)
        H = slfNRG.Junc_eff_Ham_gen(omega=0,W=Wj,ay_targ=ay,kx=qx[q],m_eff=0.023,alp_l=alpha,alp_t=alpha,mu=mu,V_J=Vj,Gam=gam,Gam_SC_factor=0,Delta=delta,phi=phi,iter=50,eta=0)

        eigs, vecs = spLA.eigsh(H, k=k, sigma=0, which='LM')
        idx_sort = np.argsort(eigs)
        eigs = eigs[idx_sort]
        omega0_bands[q] = eigs[int(k/2)]
    #plt.plot(qx, omega0_bands, c='k')
    #plt.show()

    local_min_idx = np.array(argrelextrema(omega0_bands, np.less)[0])
    local_min_idx = np.concatenate((np.array([0]), local_min_idx))
    abs_min =  omega0_bands[local_min_idx[0]]
    idx_absmin = 0
    for n in range(local_min_idx.shape[0]):
        abs_min_new = omega0_bands[local_min_idx[n]]
        if abs_min_new < abs_min:
            abs_min = abs_min_new
            idx_absmin = n

    kx_of_absmin = qx[local_min_idx[idx_absmin]]
    idx_of_absmin = local_min_idx[idx_absmin]
    #print("kx at absolute minimum", kx_of_absmin)
    #print("gap of omega0", omega0_bands[idx_of_absmin] )
    true_eig, counter = self_consistency_finder_faster(gam, mu, Wj, Vj, alpha, delta, phi, kx_of_absmin, omega0_bands[idx_of_absmin], tol)
    true_eig2 = self_consistency_finder(gam, mu, Wj, Vj, alpha, delta, phi, kx_of_absmin, omega0_bands[idx_of_absmin], tol)
    print("faster gap", true_eig)
    print("slower gap", true_eig2)
    print(counter)
    sys.exit()
    return true_eig, kx_of_absmin, idx_of_absmin

def self_consistency_finder(gam, mu, Wj, Vj, alpha, delta, phi, kx, eigs_omega0, tol):
    true_eig = None
    delta_omega = eigs_omega0
    steps = int(eigs_omega0/tol) + 1
    omega = np.linspace(0, eigs_omega0, int(steps))
    omega_bands = np.zeros(omega.shape[0])
    for w in range(omega.shape[0]):
        #print(omega.shape[0]-w)
        H = slfNRG.Junc_eff_Ham_gen(omega=omega[w],W=Wj,ay_targ=ay,kx=kx,m_eff=0.023,alp_l=alpha,alp_t=alpha,mu=mu,V_J=Vj,Gam=gam,Gam_SC_factor=0,Delta=delta,phi=phi,iter=50,eta=0)
        eigs, vecs = spLA.eigsh(H, k=k, sigma=0, which='LM')
        idx_sort = np.argsort(eigs)
        eigs = eigs[idx_sort]
        omega_bands[w] = eigs[int(k/2)]
        #print(omega[w], abs(eigs[int(k/2)] - omega[w]))
        if abs(eigs[int(k/2)] - omega[w]) < tol:
            true_eig = eigs[int(k/2)]
            break
    #plt.plot(omega, omega_bands-omega, c='k')
    #plt.plot(omega, omega, c='b')
    #plt.show()
    return true_eig

def self_consistency_finder_faster(gam, mu, Wj, Vj, alpha, delta, phi, kx, eigs_omega0, tol):
    delta_omega = eigs_omega0
    steps = int(eigs_omega0/tol) + 1
    omega = np.linspace(0, eigs_omega0, int(steps))
    omega_bands = np.zeros(omega.shape[0])

    y1 = eigs_omega0
    x1 = 0
    x2 = y1/50
    counter = 0
    while True:
        counter+=1
        H = slfNRG.Junc_eff_Ham_gen(omega=x2,W=Wj,ay_targ=ay,kx=kx,m_eff=0.023,alp_l=alpha,alp_t=alpha,mu=mu,V_J=Vj,Gam=gam,Gam_SC_factor=0,Delta=delta,phi=phi,iter=50,eta=0)
        eigs, vecs = spLA.eigsh(H, k=k, sigma=0, which='LM')
        idx_sort = np.argsort(eigs)
        eigs = eigs[idx_sort]
        y2 = eigs[int(k/2)] - x2

        if abs(y2) < tol:
            return x2, counter

        m = (y2-y1)/(x2-x1)
        b = y1-m*x1
        omega_c = -b/m

        y1=y2
        x1=x2
        x2 = omega_c
        #print("counter", counter)

dirS = 'gap_data'
if not os.path.exists(dirS):
    os.makedirs(dirS)
try:
    PLOT = str(sys.argv[1])
except:
    PLOT = 'F'
if PLOT != 'P':

    #for each point in parameter space we now have the kx value of the absolute minimum of the band structure
    #now, around this kx value we know that the true minimum is close
    #omega needs to be scanned from 0 to the eigenvalue at that k value and omega=0

    for i in range(gx.shape[0]):
        for j in range(mu.shape[0]):
            true_eig = gap(gx[i], mu[j], Vj, Wj, alpha, delta, phi, tol)

    #np.save("%s/gap Wj = %.1f nodx = %.1f nody = %.1f Vj = %.1f Vsc = %.1f alpha = %.1f delta = %.2f phi = %.3f.npy" % (dirS, Junc_width, Nod_widthx,  Nod_widthy, Vj, Vsc, alpha, delta, phi), gap_k0)
    #gc.collect()

    sys.exit()
else:
    gap = np.load("%s/gap Wj = %.1f nodx = %.1f nody = %.1f Vj = %.1f Vsc = %.1f alpha = %.1f delta = %.2f phi = %.3f.npy" % (dirS, Junc_width, Nod_widthx,  Nod_widthy, Vj, Vsc, alpha, delta, phi))

    print(gap.shape, mu.shape, gx.shape)
    gap = gap/delta
    gap = np.transpose(gap)

    plt.contourf(gx, mu, gap, 500, vmin = 0, vmax = max(gap.flatten()), cmap = 'hot')
    cbar = plt.colorbar()
    cbar.set_label(r'$E_{gap}/\Delta$')

    plt.xlabel(r'$\Gamma_x$ (meV)')
    plt.ylabel(r'$\mu$ (meV)')

    plt.xlim(gi, gf)
    title = r"$W_j$ = %.1f nm, $nodule_x$ = %.1f nm, $nodule_y$ = %.1f nm, $V_j$ = %.1f meV, $V_{SC}$ = %.1f meV, $\phi$ = %.2f " % (Junc_width, Nod_widthx, Nod_widthy, Vj, Vsc, phi)
    #title = r"$L_x =$ {} nm, $L_y =$ {} nm, SC width = {} nm, $W_j =$ {} nm, $nodule_x = ${} nm, $nodule_y = ${} nm, $\alpha = $ {} meV*A, $\phi =$ {} ".format(Lx*.1, Ly*.1, SC_width, Junc_width, Nod_widthx, Nod_widthy, alpha, phi)
    plt.title(title, loc = 'center', wrap = True)
    plt.subplots_adjust(top=0.85)
    plt.savefig('gap juncwidth = {} nodwidthx = {} nodwidthy = {} phi = {} Vj = {} Vsc = {}.png'.format(Junc_width, Nod_widthx, Nod_widthy, delta, alpha, phi, Vj, Vsc))
    plt.show()

    sys.exit()
