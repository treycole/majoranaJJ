import sys
import time
import os
import gc

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import majoranaJJ.modules.SNRG as SNRG
import majoranaJJ.modules.finders as finders
import majoranaJJ.modules.constants as const
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
cutyT = 2*cuty
cutyB = 0
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
#########################################
#Defining Hamiltonian parameters
alpha = 200 #Spin-Orbit Coupling constant: [meV*A]
phi = np.pi #SC phase difference
delta = 0.3 #Superconducting Gap: [meV]
mu = 10 #Junction potential: [meV]
gx = 1 #mev

Vj_i = -11
Vj_f = 11
delta_Vj= Vj_f - Vj_i
res = 0.1
steps = int(abs(delta_Vj/res))+1
Vj = np.linspace(Vj_i, Vj_f, steps) #meV

print("alpha = ", alpha)
print("Vj_i = ", Vj_i)
print("Vj_f = ", Vj_f)
print("Gamma_x = ", gx)

gap_Vj = np.zeros(Vj.shape[0])
kx_of_gap = np.zeros(Vj.shape[0])

del_k = 1e-5
res_gap = 0.0005
M = res_gap/del_k
targ_steps = finders.targ_step_finder(res_gap, M, np.pi/Lx)
print(targ_steps)
###################################################
dirS = 'gap_data'
if not os.path.exists(dirS):
    os.makedirs(dirS)
try:
    PLOT = str(sys.argv[1])
except:
    PLOT = 'F'
if PLOT != 'P':
    gap_Vj = np.load("%s/gapfxVj Wj = %.1f Lx = %.1f cutxT = %.1f cutyT = %.1f cutxB = %.1f cutyB = %.1f mu = %.1f phi = %.3f Vj_i = %.1f Vj_f = %.1f gx = %.2f.npy" % (dirS, Junc_width, Lx*.1, cutxT_width, cutyT_width, cutxB_width, cutyB_width, mu, phi, Vj_i, Vj_f, gx))
    for i in range(0, Vj.shape[0]):
        print(steps-i, "| Vj =", Vj[i])
        GAP, KX = SNRG.gap(Wj=Wj, Lx=Lx, cutxT=cutxT, cutyT=cutyT, cutxB=cutxB, cutyB=cutyB, ax=ax, ay=ay, gam=gx, mu=mu, Vj=Vj[i], alpha=alpha, delta=delta, phi=phi, targ_steps=50000, n_avg=10, muf=mu, PLOT=False, tol=1e-9)
        gap_Vj[i] = GAP
        kx_of_gap[i] = KX

        np.save("%s/gapfxVj Wj = %.1f Lx = %.1f cutxT = %.1f cutyT = %.1f cutxB = %.1f cutyB = %.1f mu = %.1f phi = %.3f Vj_i = %.1f Vj_f = %.1f gx = %.2f.npy" % (dirS, Junc_width, Lx*.1, cutxT_width, cutyT_width, cutxB_width, cutyB_width, mu, phi, Vj_i, Vj_f, gx), gap_Vj)
        np.save("%s/kxofgapfxVj Wj = %.1f Lx = %.1f cutxT = %.1f cutyT = %.1f cutxB = %.1f cutyB = %.1f mu = %.1f phi = %.3f Vj_i = %.1f Vj_f = %.1f gx = %.2f.npy" % (dirS, Junc_width, Lx*.1, cutxT_width, cutyT_width, cutxB_width, cutyB_width, mu, phi, Vj_i, Vj_f, gx), kx_of_gap)
        gc.collect()

    sys.exit()
else:
    gap = np.load("%s/gapfxVj Wj = %.1f nodx = %.1f nody = %.1f mu = %.1f alpha = %.1f delta = %.2f phi = %.3f Vj_i = %.1f Vj_f = %.1f gx = %.2f.npy" % (dirS, Junc_width, Nod_widthx,  Nod_widthy, mu, alpha, delta, phi, Vj_i, Vj_f, gx))
    Vj = np.load("%s/Vj Wj = %.1f nodx = %.1f nody = %.1f mu = %.1f alpha = %.1f delta = %.2f phi = %.3f Vj_i = %.1f Vj_f = %.1f gx = %.2f.npy" % (dirS, Junc_width, Nod_widthx,  Nod_widthy, mu, alpha, delta, phi, Vj_i, Vj_f, gx))

    plt.plot(Vj, gap/delta)
    plt.grid()
    plt.xlabel(r'$V_j$ (meV)')
    plt.ylabel(r'$E_{gap}$ (meV)')
    #plt.xlim(0, 2)
    plt.ylim(0, 0.4)
    title = r"SNRG $E_Z$ = %.2f meV $W_j$ = %.1f nm, $nodule_x$ = %.1f nm, $nodule_y$ = %.1f nm, $\mu$ = %.1f meV, $\phi$ = %.2f " % (gx, Junc_width, Nod_widthx, Nod_widthy, mu, phi)

    plt.title(title, loc = 'center', wrap = True)
    plt.subplots_adjust(top=0.85)
    #plt.savefig('gapfxmu juncwidth = {} nodwidthx = {} nodwidthy = {} alpha = {} phi = {} Vj = {}.png'.format(Junc_width, Nod_widthx, Nod_widthy, alpha, phi, Vj))
    plt.show()

    sys.exit()
