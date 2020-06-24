"""
    Comparing the low-energy normal spectrum to the full Hamiltonian
"""

import sys
import os
dir = os.getcwd()
os.chdir('..')
home_dir = os.getcwd()
sys.path.append('%s/Build_Files' % (home_dir) )   # Adding Build_Files to system path
os.chdir(dir)
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import LineCollection
import Junction_Model_Class as JMC
import parameters as par
np.set_printoptions(linewidth = 500)

### parameters ###
Lx = 50. * 10.    # Length of supercell in the x-direction (in Angstroms)
W_sc = 500. * 10.  # how wide the superconductor regions are in the y-direction
W_j = 40. * 10.    # the width of the junction regino in the y-direction
W_c1 = 13. * 10.    # The width of the bottom cut of the nodule in the y-direction
W_c2 = 7. * 10.    # The width of the top cut of the nodule in the y-direction
L_c = 25. * 10.     # Length of the nodule in the x-direction
m_eff = 0.023      # effective mass
alpha = 250.       # Rashba spin-orbit coupling strength in meV * Angstrom

a_SC = 2. * 10.    # target mesh spacing in the y-direction near the junction and x-direction throughout the entire device
a_J = 1. * 10.     # target mesh spacing in the y-direction
W_sc_buffer = 10. * 10.  # width of buffer region of the superconductor where the mesh spacing is a_sc
ay_extended_targ = 10. * 10. # mesh spacing in the y-direction outside of the buffer region

V_j = 0.        # potential in the junction region
V_sc = 100.    # potential in the SC regions (in meV)
mu = 0.         # chemical potential (in meV), which shifts the spectrum uniformly throughout the entire device
Gam = 1.e-4     # Zeeman energy for a magnetic field oriented along the length of the junction (x-direction)

### Creating instance of the Junction model
system = JMC.Junction_Model(Lx,W_sc,W_j,W_c1,W_c2,L_c,a_SC,a_J,m_eff,alpha,W_sc_buffer = W_sc_buffer, ay_extended_targ= ay_extended_targ)
system.MESH.PLOT.plot_elements2() # Plots the mesh elements

### Create a low-energy subband space for the normal state Hamiltonian
qx_knot = .01 * np.pi/Lx            # small qx value that we use to generate low-energy subspace (to avoid any degeneracy issues)
num = 400                            # number of states to include in the low-energy subspace
lNRG = system.HAM.generate_lNRG_subspace(qx_knot,V_j,V_sc,num)  # generating the low-energy subspace

### Looping through qx values to obtain spectrum from low-energy basis
qx = np.linspace(0.,np.pi/Lx * 1.,501)
eig_arrL = np.zeros((qx.size,num))
for i in range(qx.size):
    if i % 100 == 0:
        print ("%d qx values left" % (qx.size - i))
    HamL = lNRG.compile_Ham(qx[i],mu,Gam,V_j,V_sc)  # generate the Hamiltonian in the low-energy basis at qx[i]
    eig_arrL[i,:],U = lNRG.solve_Ham(HamL)          # Diagonalize the low-energy Hamiltonian

### Plotting spectrum
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
for i in range(num):
    if i == 0:
        ax.plot(qx,eig_arrL[:,i],c = 'k',label = 'Low-energy basis')
    else:
        ax.plot(qx,eig_arrL[:,i],c = 'k')
    ax.plot(-qx,eig_arrL[:,i],c = 'k')
ax.legend()
ax.grid()
ax.set_ylim(ymin = 0.)
ax.set_xlabel(r"$q_x$ ($A^{-1}$)",fontsize = 12)
ax.set_ylabel(r"$E$ (mev)",fontsize = 12)
plt.show()
